#!/usr/bin/env python3
"""
統計 DS_Train / DS_Val / DS_Indp 三個資料集的序列長度分布與正負樣本(label)分布。

資料來源：featureStat/DS_Train.csv、DS_Val.csv、DS_Indp.csv
（由 mainProgram/Main_FeatureStk(_emb).py 產生，欄位為 name, sequence, label；
 label: 1 = 正樣本 (pos)，0 = 負樣本 (neg)）

用法：
    python DS_split_stat.py
    python DS_split_stat.py --bins 40 --logy
    python DS_split_stat.py -i ../data/featureStat -o .

輸出（存到 --outdir）：
    DS_split_length_summary.csv   每個 DS_* × label 的長度統計（數量、最短、最長、平均、中位數...）
    DS_split_label_count.csv      每個 DS_* 的正負樣本數量與比例
    DS_split_length_dist.png      合併為一張圖：最上面一列正負樣本數量長條圖，
                                  下面每個資料集一列，各自的序列長度直方圖 + 盒鬚圖
"""

import argparse
import os
import statistics

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_NAMES = ["DS_Train", "DS_Val", "DS_Indp"]
LABEL_NAME = {1: "pos", 0: "neg"}
COLORS = {"pos": "#d1495b", "neg": "#2e86ab"}


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
    ap = argparse.ArgumentParser(description="統計 DS_Train/DS_Val/DS_Indp 序列長度與正負樣本分布")
    ap.add_argument("-i", "--indir", default="featureStat", help="DS_Train.csv 等檔案所在資料夾")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--logy", action="store_true", help="直方圖 Y 軸取 log")
    args = ap.parse_args()

    dfs = {}
    for name in DATASET_NAMES:
        path = os.path.join(args.indir, f"{name}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"找不到檔案：{path}")
        df = pd.read_csv(path)
        df["length"] = df["sequence"].str.len()
        dfs[name] = df

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 正負樣本數量/比例 ----
    labelRows = []
    for name, df in dfs.items():
        counts = df["label"].value_counts()
        nPos, nNeg = counts.get(1, 0), counts.get(0, 0)
        total = nPos + nNeg
        labelRows.append({
            "dataset": name, "n_pos": nPos, "n_neg": nNeg, "n_total": total,
            "pos_ratio": round(nPos / total, 4) if total else 0,
            "neg_ratio": round(nNeg / total, 4) if total else 0,
        })
    labelDf = pd.DataFrame(labelRows)
    labelCsvPath = os.path.join(args.outdir, "DS_split_label_count.csv")
    labelDf.to_csv(labelCsvPath, index=False)
    print("[正負樣本分布]")
    print(labelDf.to_string(index=False))
    print(f"已輸出：{labelCsvPath}\n")

    # ---- 序列長度分布 ----
    lengthRows = []
    for name, df in dfs.items():
        lengthRows.append({"group": f"{name}_all", **summarize(df["length"].tolist())})
        for labelValue, labelName in LABEL_NAME.items():
            lengths = df.loc[df["label"] == labelValue, "length"].tolist()
            if lengths:
                lengthRows.append({"group": f"{name}_{labelName}", **summarize(lengths)})
    lengthDf = pd.DataFrame(lengthRows)
    lengthCsvPath = os.path.join(args.outdir, "DS_split_length_summary.csv")
    lengthDf.to_csv(lengthCsvPath, index=False)
    print("[序列長度分布]")
    print(lengthDf.to_string(index=False))
    print(f"已輸出：{lengthCsvPath}\n")

    # ---- 畫圖：合併成一張圖。第一列是正負樣本數量長條圖，下面每個資料集一列，左直方圖、右盒鬚圖 ----
    fig = plt.figure(figsize=(13, 3.6 + 4.2 * len(DATASET_NAMES)))
    gs = fig.add_gridspec(len(DATASET_NAMES) + 1, 2,
                          height_ratios=[1] + [1.2] * len(DATASET_NAMES),
                          width_ratios=[3, 1])

    ax_bar = fig.add_subplot(gs[0, :])
    barWidth = 0.35
    xPos = range(len(DATASET_NAMES))
    posCounts = [labelDf.loc[labelDf["dataset"] == n, "n_pos"].iloc[0] for n in DATASET_NAMES]
    negCounts = [labelDf.loc[labelDf["dataset"] == n, "n_neg"].iloc[0] for n in DATASET_NAMES]
    barsPos = ax_bar.bar([x - barWidth / 2 for x in xPos], posCounts, barWidth, label="pos", color=COLORS["pos"])
    barsNeg = ax_bar.bar([x + barWidth / 2 for x in xPos], negCounts, barWidth, label="neg", color=COLORS["neg"])
    ax_bar.bar_label(barsPos, padding=2)
    ax_bar.bar_label(barsNeg, padding=2)
    ax_bar.set_xticks(list(xPos))
    ax_bar.set_xticklabels(DATASET_NAMES)
    ax_bar.set_ylabel("Number of sequences")
    ax_bar.set_title("Positive / negative sample count per dataset")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    for i, name in enumerate(DATASET_NAMES):
        df = dfs[name]
        ax_h = fig.add_subplot(gs[i + 1, 0])
        ax_b = fig.add_subplot(gs[i + 1, 1])
        allLengths = df["length"].tolist()
        lo, hi = min(allLengths), max(allLengths)
        width = max(1, -(-(hi - lo + 1) // args.bins))
        bins = [lo - 0.5 + k * width for k in range((hi - lo + 1) // width + 2)]

        groupsByLabel = {LABEL_NAME[v]: df.loc[df["label"] == v, "length"].tolist()
                         for v in LABEL_NAME if (df["label"] == v).any()}
        labs = [l for l in ("pos", "neg") if l in groupsByLabel]
        for lab in labs:
            L = groupsByLabel[lab]
            ax_h.hist(L, bins=bins, alpha=0.55, color=COLORS[lab],
                      label=f"{lab} (n={len(L):,}, median={statistics.median(L):g})",
                      edgecolor="white", linewidth=0.4)
            ax_h.axvline(statistics.median(L), color=COLORS[lab], ls="--", lw=1.4)
        if args.logy:
            ax_h.set_yscale("log")
        ax_h.set_title(f"{name}: sequence length distribution")
        ax_h.set_xlabel("Sequence length")
        ax_h.set_ylabel("Number of sequences")
        ax_h.legend()
        ax_h.grid(axis="y", alpha=0.3)

        bp = ax_b.boxplot([groupsByLabel[l] for l in labs], patch_artist=True, widths=0.55)
        ax_b.set_xticks(range(1, len(labs) + 1))
        ax_b.set_xticklabels(labs)
        for patch, lab in zip(bp["boxes"], labs):
            patch.set_facecolor(COLORS[lab])
            patch.set_alpha(0.6)
        ax_b.set_title(f"{name}: box plot")
        ax_b.set_ylabel("Sequence length")
        ax_b.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    outPng = os.path.join(args.outdir, "DS_split_length_dist.png")
    fig.savefig(outPng, dpi=200)
    print(f"已輸出：{outPng}")


if __name__ == "__main__":
    main()
