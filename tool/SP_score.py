from sympy import symbols, solve, Eq, sqrt


def calculate_specificity(accuracy, precision, recall, mcc, N):
    # 定義TP, TN, FP, FN的符號
    TP, TN, FP, FN = symbols('TP TN FP FN')

    # 根據給定的值定義方程
    eq_accuracy = Eq((TP + TN) / N, accuracy)
    eq_precision = Eq(TP / (TP + FP), precision)
    eq_recall = Eq(TP / (TP + FN), recall)
    eq_mcc = Eq((TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), mcc)

    # 解方程組
    solution = solve((eq_accuracy, eq_precision, eq_recall, eq_mcc), (TP, TN, FP, FN))

    # 提取解
    TP_val = solution[0][0]
    TN_val = solution[0][1]
    FP_val = solution[0][2]
    FN_val = solution[0][3]

    # 計算特異度
    specificity_val = TN_val / (TN_val + FP_val)

    return TP_val, TN_val, FP_val, FN_val, specificity_val


# 給定數據集的新值
accuracy_new = 0.709125475
precision_new =0.569014085
recall_new = 1
mcc_new = 0.548008202
N_new = 1000

# 計算特異度和混淆矩陣值
TP_val, TN_val, FP_val, FN_val, specificity_val = calculate_specificity(accuracy_new, precision_new, recall_new,
                                                                        mcc_new, N_new)

# 輸出結果
print(f"TP: {TP_val}")
print(f"TN: {TN_val}")
print(f"FP: {FP_val}")
print(f"FN: {FN_val}")
print(f"Specificity: {specificity_val}")
"""
import re
import os
import sys

# from iFeature
def ReadFasta(file, output_fasta):
    if os.path.exists(file) is False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) is None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('\n>')
    records[0] = records[0].split('>')[1]

    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name = array[0]
        sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    # 将 myFasta 写入输出文件
    with open(output_fasta, 'w') as f_out:
        for item in myFasta:
            f_out.write(f'>{item[0]}\n{item[1]}\n')

    return myFasta

# 使用方法
input_fasta = "C:/Users/Jonathan Lin/Downloads/uniprotkb_go_0005488_AND_organism_name_2024_09_02.fasta"  # 输入的 FASTA 文件路径
output_fasta = "C:/Users/Jonathan Lin/Downloads/output.fasta"  # 输出的 FASTA 文件路径

ReadFasta(input_fasta, output_fasta)"""


