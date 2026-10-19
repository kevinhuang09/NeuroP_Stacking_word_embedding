#!/usr/bin/env python3
"""
比較 Positive / Negative FASTA 的序列長度分布。

預設讀取同資料夾下的四個檔案：
    MainDatasetPos.fasta, MainDatasetNeg.fasta,
    DS_IndpPos.fasta,     DS_IndpNeg.fasta
檔名含 "Pos" 歸為 pos、含 "Neg" 歸為 neg。

用法：
    python pos_neg_length_dist.py                    # 用預設四個檔案
    python pos_neg_length_dist.py a_Pos.fasta b_Neg.fasta ...
    python pos_neg_length_dist.py --bins 40 --logy

輸出：
    length_summary.csv        每個檔案 / 每組的統計（數量、最短、最長、平均、中位數...）
    length_counts_pos_neg.csv 每個長度在各檔案中的出現次數
    length_dist_pos_neg.png   各資料集 pos vs neg 的直方圖 + 盒鬚圖

需要套件：matplotlib（pip install matplotlib）
"""

import argparse
import csv
import gzip
import os
import statistics
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FILES = [
    "MainDatasetPos.fasta", "MainDatasetNeg.fasta",
    "DS_IndpPos.fasta", "DS_IndpNeg.fasta",
]
COLORS = {"pos": "#d1495b", "neg": "#2e86ab"}


def read_lengths(path):
    """回傳每條序列的長度 list（支援多行序列與 .gz）。"""
    opener = gzip.open if path.endswith(".gz") else open
    lengths, cur = [], None
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    lengths.append(cur)
                cur = 0
            elif cur is not None:
                cur += len(line.replace(" ", ""))
    if cur is not None:
        lengths.append(cur)
    return lengths


def label_of(path):
    name = os.path.basename(path).lower()
    if "pos" in name:
        return "pos"
    if "neg" in name:
        return "neg"
    return "other"


def dataset_of(path):
    """去掉 Pos/Neg 與副檔名，取得資料集名稱（如 MainDataset、DS_Indp）。"""
    name = os.path.basename(path).split(".")[0]
    for tag in ("Pos", "Neg", "pos", "neg", "POS", "NEG"):
        name = name.replace(tag, "")
    return name.strip("_-") or "dataset"


def summarize(lengths):
    return {
        "n": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 2),
        "median": statistics.median(lengths),
        "stdev": round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0,
        "q1": statistics.quantiles(lengths, n=4)[0] if len(lengths) > 1 else lengths[0],
        "q3": statistics.quantiles(lengths, n=4)[2] if len(lengths) > 1 else lengths[0],
    }


def main():
    ap = argparse.ArgumentParser(description="比較 pos/neg FASTA 長度分布")
    ap.add_argument("files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--logy", action="store_true", help="直方圖 Y 軸取 log")
    ap.add_argument("-o", "--outdir", default=".")
    args = ap.parse_args()

    missing = [f for f in args.files if not os.path.exists(f)]
    if missing:
        sys.exit("找不到檔案：" + ", ".join(missing))

    # data[dataset][label] = lengths
    data, per_file = {}, {}
    for f in args.files:
        L = read_lengths(f)
        if not L:
            print(f"警告：{f} 沒有序列，略過")
            continue
        per_file[f] = L
        data.setdefault(dataset_of(f), {}).setdefault(label_of(f), []).extend(L)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 統計表 ----
    rows = []
    for f, L in per_file.items():
        rows.append({"group": os.path.basename(f), **summarize(L)})
    for lab in ("pos", "neg"):
        allL = [x for ds in data.values() for x in ds.get(lab, [])]
        if allL:
            rows.append({"group": f"ALL_{lab}", **summarize(allL)})

    print(f"{'group':<26}{'n':>8}{'min':>7}{'max':>7}{'mean':>9}{'median':>8}{'stdev':>9}")
    for r in rows:
        print(f"{r['group']:<26}{r['n']:>8,}{r['min']:>7}{r['max']:>7}"
              f"{r['mean']:>9}{r['median']:>8}{r['stdev']:>9}")

    with open(os.path.join(args.outdir, "length_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 每個長度在每個檔案的次數
    counters = {os.path.basename(f): Counter(L) for f, L in per_file.items()}
    all_len = sorted(set().union(*[c.keys() for c in counters.values()]))
    with open(os.path.join(args.outdir, "length_counts_pos_neg.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["length"] + list(counters))
        for L in all_len:
            w.writerow([L] + [counters[k].get(L, 0) for k in counters])

    # ---- 畫圖：每個資料集一列，左直方圖、右盒鬚圖 ----
    datasets = list(data)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 4.2 * len(datasets)),
                             gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    for i, ds in enumerate(datasets):
        groups = data[ds]
        ax_h, ax_b = axes[i]
        allL = [x for v in groups.values() for x in v]
        lo, hi = min(allL), max(allL)
        # 長度是整數 → bin 邊界對齊整數，避免每格包含的長度數不一致而出現假尖峰
        width = max(1, -(-(hi - lo + 1) // args.bins))
        bins = [lo - 0.5 + k * width for k in range((hi - lo + 1) // width + 2)]

        labs = [l for l in ("pos", "neg") if l in groups]
        for lab in labs:
            L = groups[lab]
            ax_h.hist(L, bins=bins, alpha=0.55, color=COLORS[lab],
                      label=f"{lab} (n={len(L):,}, median={statistics.median(L):g})",
                      edgecolor="white", linewidth=0.4)
            ax_h.axvline(statistics.median(L), color=COLORS[lab], ls="--", lw=1.4)
        if args.logy:
            ax_h.set_yscale("log")
        ax_h.set_title(f"{ds}: sequence length distribution")
        ax_h.set_xlabel("Sequence length")
        ax_h.set_ylabel("Number of sequences")
        ax_h.legend()
        ax_h.grid(axis="y", alpha=0.3)

        bp = ax_b.boxplot([groups[l] for l in labs], patch_artist=True, widths=0.55)
        ax_b.set_xticks(range(1, len(labs) + 1))
        ax_b.set_xticklabels(labs)   # 相容新舊版 matplotlib
        for patch, lab in zip(bp["boxes"], labs):
            patch.set_facecolor(COLORS[lab])
            patch.set_alpha(0.6)
        ax_b.set_title(f"{ds}: box plot")
        ax_b.set_ylabel("Sequence length")
        ax_b.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.outdir, "length_dist_pos_neg.png")
    fig.savefig(out_png, dpi=200)
    print(f"\n已輸出：length_summary.csv, length_counts_pos_neg.csv, {out_png}")


if __name__ == "__main__":
    main()
