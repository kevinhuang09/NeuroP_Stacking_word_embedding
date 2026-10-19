#!/usr/bin/env python3
"""
依照「參考序列（例如 pos）的長度分布」，從 UniProt FASTA 隨機抽出指定數量的序列當作 negative。

兩種模式：
  whole（預設）：直接抽「整條」UniProt 序列，長度分布與參考檔一致。
                 適合蛋白質等級的長度（數十到數百 aa 以上）。
  crop        ：從 UniProt 蛋白中「隨機截取片段」，片段長度照參考分布抽。
                 適合短胜肽（UniProt 中很短的完整蛋白很少，整條抽常會不夠）。

UniProt 資料（建議先下載到本機）：
  Swiss-Prot（已審核，約 57 萬條，~90 MB）：
    https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
  可直接用 .gz，不必解壓。

用法範例：
  # 依 MainDatasetPos 的長度分布抽 2000 條 neg
  python sample_uniprot_neg.py -r MainDatasetPos.fasta -u uniprot_sprot.fasta.gz -n 2000 -o neg_2000.fasta

  # 參考多個檔案、排除所有既有正負樣本、短胜肽用 crop 模式
  python sample_uniprot_neg.py -r MainDatasetPos.fasta DS_IndpPos.fasta \\
      -u uniprot_sprot.fasta.gz -n 5000 --mode crop \\
      --exclude MainDatasetPos.fasta DS_IndpPos.fasta MainDatasetNeg.fasta DS_IndpNeg.fasta \\
      -o neg_5000.fasta --seed 42

輸出：
  <out>.fasta                抽出的 negative 序列
  <out>_length_compare.png   參考 vs 抽樣的長度分布比較圖
  螢幕上印出兩邊的長度統計（若有 scipy 會加上 KS 檢定）

需要套件：matplotlib（scipy 可選）
"""

import argparse
import gzip
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------- FASTA I/O
def iter_fasta(path):
    """逐條產生 (header, sequence)，支援多行序列與 .gz。"""
    opener = gzip.open if path.endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header, chunks = line[1:], []
            else:
                chunks.append(line.replace(" ", ""))
    if header is not None:
        yield header, "".join(chunks).upper()


def write_fasta(records, path, width=60):
    with open(path, "w") as fh:
        for h, s in records:
            fh.write(f">{h}\n")
            for i in range(0, len(s), width):
                fh.write(s[i:i + width] + "\n")


def uniprot_acc(header):
    """sp|P12345|NAME_HUMAN ... → P12345；其他格式取第一個欄位。"""
    m = re.match(r"^(?:sp|tr)\|([^|]+)\|", header)
    return m.group(1) if m else header.split()[0]


# ---------------------------------------------------------------- 配額計算
def allocate(n_total, weights):
    """依權重把 n_total 分配到各 key（最大餘數法，總和剛好等於 n_total）。"""
    s = sum(weights.values())
    raw = {k: n_total * w / s for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    rest = n_total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rest]:
        alloc[k] += 1
    return {k: v for k, v in alloc.items() if v > 0}


def bin_of(length, width):
    return (length // width) * width


# ---------------------------------------------------------------- 篩選
def passes_filter(seq, args, exclude_seqs):
    if not seq:
        return False
    if args.standard_only and not set(seq) <= STANDARD_AA:
        return False
    if seq in exclude_seqs:
        return False
    return True


# ---------------------------------------------------------------- whole 模式
def sample_whole(args, ref_lengths, exclude_seqs, exclude_ids, rng):
    weights = Counter(bin_of(L, args.bin_width) for L in ref_lengths)
    target = allocate(args.n, weights)
    cap = {b: t * args.oversample for b, t in target.items()}   # 多存一些給缺額的鄰近 bin 補
    reservoir = defaultdict(list)
    seen = defaultdict(int)
    n_scanned = 0

    for h, s in iter_fasta(args.uniprot):
        n_scanned += 1
        b = bin_of(len(s), args.bin_width)
        if b not in cap:
            continue
        if uniprot_acc(h) in exclude_ids or not passes_filter(s, args, exclude_seqs):
            continue
        # 每個長度 bin 各自做 reservoir sampling（只掃一遍、記憶體固定）
        seen[b] += 1
        if len(reservoir[b]) < cap[b]:
            reservoir[b].append((h, s))
        else:
            j = rng.randrange(seen[b])
            if j < cap[b]:
                reservoir[b][j] = (h, s)
        if n_scanned % 100000 == 0:
            print(f"  已掃描 {n_scanned:,} 條...", file=sys.stderr)

    print(f"  共掃描 UniProt {n_scanned:,} 條", file=sys.stderr)

    picked, used, deficit = [], set(), {}
    for b, t in target.items():
        pool = reservoir[b]
        rng.shuffle(pool)
        take = []
        for h, s in pool:
            if len(take) == t:
                break
            if s not in used:
                take.append((h, s)); used.add(s)
        picked.extend(take)
        if len(take) < t:
            deficit[b] = t - len(take)

    # 缺額：從最接近的 bin 剩餘序列補
    if deficit:
        print(f"  警告：{len(deficit)} 個長度區間 UniProt 數量不足，"
              f"共缺 {sum(deficit.values())} 條，改從鄰近長度補齊。", file=sys.stderr)
        leftovers = {b: [r for r in reservoir[b] if r[1] not in used] for b in reservoir}
        for b, need in deficit.items():
            for nb in sorted(leftovers, key=lambda x: abs(x - b)):
                while need and leftovers[nb]:
                    h, s = leftovers[nb].pop()
                    if s not in used:
                        picked.append((h, s)); used.add(s); need -= 1
                if not need:
                    break
    if len(picked) < args.n:
        print(f"  警告：只抽到 {len(picked):,} / {args.n:,} 條。UniProt 中這個長度範圍的完整序列不夠，"
              f"建議改用 --mode crop（從長蛋白截取片段），或加大 --bin-width / --oversample。",
              file=sys.stderr)
    return picked


# ---------------------------------------------------------------- crop 模式
def sample_crop(args, ref_lengths, exclude_seqs, exclude_ids, rng):
    min_len = min(ref_lengths)
    pool_size = args.n * args.pool_factor
    pool, seen, n_scanned = [], 0, 0

    # 先從 UniProt 用 reservoir sampling 抽一個蛋白池（長度 >= 參考最短長度）
    for h, s in iter_fasta(args.uniprot):
        n_scanned += 1
        if len(s) < min_len:
            continue
        if uniprot_acc(h) in exclude_ids or not passes_filter(s, args, set()):
            continue
        seen += 1
        if len(pool) < pool_size:
            pool.append((h, s))
        else:
            j = rng.randrange(seen)
            if j < pool_size:
                pool[j] = (h, s)
        if n_scanned % 100000 == 0:
            print(f"  已掃描 {n_scanned:,} 條...", file=sys.stderr)
    print(f"  共掃描 UniProt {n_scanned:,} 條，蛋白池 {len(pool):,} 條", file=sys.stderr)

    # 目標長度：直接從參考長度中重抽，分布自然一致
    targets = [rng.choice(ref_lengths) for _ in range(args.n)]
    targets.sort(reverse=True)                     # 先處理長的，比較不會找不到夠長的蛋白
    rng.shuffle(pool)
    pool.sort(key=lambda r: len(r[1]))
    lens = [len(s) for _, s in pool]
    available = list(range(len(pool)))

    import bisect
    picked, used, used_frags = [], set(), set()
    for L in targets:
        ok = False
        for _ in range(50):
            lo = bisect.bisect_left(lens, L)
            if lo >= len(pool):
                break
            idx = rng.randrange(lo, len(pool))
            h, s = pool[idx]
            if args.unique_protein and idx in used:
                continue
            start = rng.randrange(len(s) - L + 1)
            frag = s[start:start + L]
            if frag in exclude_seqs or frag in used_frags:   # 與參考完全相同或重複就跳過
                continue
            if args.no_overlap_with_ref and any_overlap(frag, exclude_seqs):
                continue
            used.add(idx)
            used_frags.add(frag)
            picked.append((f"{uniprot_acc(h)}|crop:{start + 1}-{start + L}|{h}", frag))
            ok = True
            break
        if not ok:
            print(f"  警告：找不到長度 {L} 的可用片段", file=sys.stderr)
    return picked


def any_overlap(frag, ref_seqs):
    """片段是否包含任何參考序列，或被參考序列包含（慢，只在 --no-overlap-with-ref 時使用）。"""
    return any(frag in r or r in frag for r in ref_seqs)


# ---------------------------------------------------------------- 比較圖
def compare_plot(ref_lengths, neg_lengths, out_png, bins):
    lo, hi = min(ref_lengths + neg_lengths), max(ref_lengths + neg_lengths)
    width = max(1, -(-(hi - lo + 1) // bins))
    edges = [lo - 0.5 + k * width for k in range((hi - lo + 1) // width + 2)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(ref_lengths, bins=edges, density=True, alpha=0.55, color="#d1495b",
            label=f"reference (n={len(ref_lengths):,})", edgecolor="white", linewidth=0.4)
    ax.hist(neg_lengths, bins=edges, density=True, alpha=0.55, color="#2e86ab",
            label=f"sampled neg (n={len(neg_lengths):,})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Density")
    ax.set_title("Length distribution: reference vs sampled negatives")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def describe(name, L):
    return (f"{name:<12} n={len(L):>7,}  min={min(L):>5}  max={max(L):>6}  "
            f"mean={statistics.mean(L):>8.2f}  median={statistics.median(L):>7.1f}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="依長度分布從 UniProt 隨機抽 negative 序列")
    ap.add_argument("-r", "--ref", nargs="+", required=True, help="參考長度分布的 FASTA（例如 pos）")
    ap.add_argument("-u", "--uniprot", required=True, help="UniProt FASTA（可 .gz）")
    ap.add_argument("-n", type=int, required=True, help="要抽幾條")
    ap.add_argument("-o", "--out", default="uniprot_neg.fasta", help="輸出 FASTA")
    ap.add_argument("--mode", choices=["whole", "crop"], default="whole")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="要排除的序列檔（預設 = --ref 的檔案）；序列或 UniProt accession 相同者不抽")
    ap.add_argument("--bin-width", type=int, default=1,
                    help="whole 模式的長度區間寬度（1 = 逐一長度精確配對）")
    ap.add_argument("--oversample", type=int, default=3,
                    help="whole 模式每區間多存幾倍候選，用於補缺額")
    ap.add_argument("--pool-factor", type=int, default=20,
                    help="crop 模式蛋白池大小 = n × 此值")
    ap.add_argument("--unique-protein", action="store_true", default=True,
                    help="crop 模式每個蛋白最多取一段（預設開）")
    ap.add_argument("--allow-same-protein", dest="unique_protein", action="store_false",
                    help="crop 模式允許同一蛋白取多段")
    ap.add_argument("--no-overlap-with-ref", action="store_true",
                    help="crop 模式排除與參考序列互相包含的片段（較慢）")
    ap.add_argument("--keep-nonstandard", dest="standard_only", action="store_false",
                    help="保留含 X/B/Z/U/O 等非標準胺基酸的序列（預設排除）")
    ap.add_argument("--bins", type=int, default=50, help="比較圖的 bin 數")
    ap.add_argument("--seed", type=int, default=None, help="隨機種子（填了結果可重現）")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ref_lengths = [len(s) for f in args.ref for _, s in iter_fasta(f)]
    if not ref_lengths:
        sys.exit("錯誤：參考檔沒有序列")
    print(describe("reference", ref_lengths))

    exclude_files = args.exclude if args.exclude is not None else args.ref
    exclude_seqs, exclude_ids = set(), set()
    for f in exclude_files:
        for h, s in iter_fasta(f):
            exclude_seqs.add(s)
            exclude_ids.add(uniprot_acc(h))
    print(f"排除序列 {len(exclude_seqs):,} 條（來自 {len(exclude_files)} 個檔案）")

    print(f"從 {args.uniprot} 抽樣中（mode={args.mode}）...")
    if args.mode == "whole":
        picked = sample_whole(args, ref_lengths, exclude_seqs, exclude_ids, rng)
    else:
        picked = sample_crop(args, ref_lengths, exclude_seqs, exclude_ids, rng)

    if not picked:
        sys.exit("錯誤：沒有抽到任何序列，請確認 UniProt 檔與長度範圍。")
    rng.shuffle(picked)
    write_fasta(picked, args.out)

    neg_lengths = [len(s) for _, s in picked]
    print(describe("sampled neg", neg_lengths))
    try:
        from scipy.stats import ks_2samp
        ks = ks_2samp(ref_lengths, neg_lengths)
        print(f"KS 檢定：D = {ks.statistic:.4f}, p = {ks.pvalue:.3g}"
              "（D 越小代表分布越接近）")
    except ImportError:
        pass

    out_png = os.path.splitext(args.out)[0] + "_length_compare.png"
    compare_plot(ref_lengths, neg_lengths, out_png, args.bins)
    print(f"\n已輸出：{args.out}（{len(picked):,} 條）, {out_png}")


if __name__ == "__main__":
    main()
