import os


def count_multiple_fasta(file_list):
    """統計多個 FASTA 檔案的序列數量"""
    total_all_files = 0
    results = {}

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"警告: 找不到檔案 {file_path}")
            continue

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    count += 1

        results[file_path] = count
        total_all_files += count

    return results, total_all_files


# 填入你 4 個 FASTA 檔案的路徑
fasta_files = ["NeuroPosTest.fasta", "NeuroNegTest.fasta", "NeuroPosTrain.fasta", "NeuroNegTrain.fasta"]

# 執行統計
file_counts, total = count_multiple_fasta(fasta_files)

# 印出結果
print("=== FASTA 序列統計結果 ===")
for file, count in file_counts.items():
    print(f"{file}: {count:,} 條序列")

print("-" * 30)
print(f"所有檔案總計: {total:,} 條序列")