import pandas as pd


def csv_seq_only_to_fasta(csv_path, fasta_path, prefix="seq"):
    """將只有 seq 欄位的 CSV 轉為 FASTA 格式

    :param csv_path: 輸入的 CSV 檔案路徑
    :param fasta_path: 輸出的 FASTA 檔案路徑
    :param prefix: 自動生成的 ID 前綴 (預設為 seq_1, seq_2...)
    """
    df = pd.read_csv(csv_path)

    with open(fasta_path, "w", encoding="utf-8") as fasta_file:
        for idx, row in enumerate(df["seq"], start=1):
            # 確保內容為字串並移除前後空白
            sequence = str(row).strip()

            # 如果序列不是空值才寫入
            if sequence and sequence.lower() != "nan":
                # 自動生成 ID，如 >seq_1, >seq_2...
                header = f">{prefix}_{idx}"
                fasta_file.write(f"{header}\n{sequence}\n")

    print(f"轉換完成！已寫入 {len(df):,} 筆序列至 {fasta_path}")


# ================= 使用範例 =================
csv_seq_only_to_fasta(
    csv_path="NeuroPosTrain.csv",  # 你的 CSV 檔名
    fasta_path="NeuroPosTrain.fasta",  # 輸出的 FASTA 檔名
    prefix="seq",  # ID 前綴，輸出會變成 >seq_1, >seq_2 ...
)