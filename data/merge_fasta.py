def merge_fasta(file1_path, file2_path, output_path):
    """合併兩個 FASTA 檔案並輸出至新的檔案"""
    with open(output_path, "w", encoding="utf-8") as outfile:
        # 讀取並寫入第一個 FASTA 檔
        with open(file1_path, "r", encoding="utf-8") as f1:
            outfile.write(f1.read())

        # 確保兩個檔案內容之間有換行，避免序列標頭黏在一起
        outfile.write("\n")

        # 讀取並寫入第二個 FASTA 檔
        with open(file2_path, "r", encoding="utf-8") as f2:
            outfile.write(f2.read())


# 使用範例
merge_fasta("MainDatasetPos.fasta", "DS_IndpPos.fasta", "NeuroPos.fasta")