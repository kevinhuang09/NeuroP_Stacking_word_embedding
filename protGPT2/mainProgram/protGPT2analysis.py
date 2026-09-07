import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd

class Protgpt2Analysis:
    def __init__(self, outputPath, foldNum, datasetName, outputSuffix, lengthBinList, dpcAnalysisFileName, lengthAnalysisFileName):
        self.outputPath = outputPath
        self.foldNum = foldNum
        self.datasetName = datasetName
        self.aminoList = []
        self.dipeptideList = []
        self.outputSuffix = outputSuffix
        self.lengthBinList = lengthBinList
        self.dpcAnalysisFileName = dpcAnalysisFileName
        self.lengthAnalysisFileName = lengthAnalysisFileName

    def countAminoAcidComposition(self, result):
        aminoAcidCounter = Counter()
        for sequence in result:
            aminoAcidCounter.update(sequence)
        for aminoAcid in "ACDEFGHIKLMNPQRSTVWY":
            if aminoAcid not in aminoAcidCounter:
                aminoAcidCounter[aminoAcid] = 0
        sortedAminoAcidCounterDict = {aa: aminoAcidCounter[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"}
        self.aminoList.append(list(sortedAminoAcidCounterDict.values()))
        return sortedAminoAcidCounterDict

    def countDipeptideComposition(self, result):
        dipeptideCounter = Counter()
        for sequence in result:
            for i in range(len(sequence) - 1):
                dipeptide = sequence[i:i + 2]
                dipeptideCounter.update([dipeptide])
        dpcList = [f"{aa1}{aa2}" for aa1 in 'ACDEFGHIKLMNPQRSTVWY' for aa2 in 'ACDEFGHIKLMNPQRSTVWY']
        dpcDict = {dipeptide: dipeptideCounter[dipeptide] for dipeptide in dpcList}   #檢察預設補0的功能

        # 補0缺失的dipeptide
        for dipeptide in dpcList:
            if dipeptide not in dpcDict:
                dpcDict[dipeptide] = 0

        self.dipeptideList.append(list(dpcDict.values()))
        return dpcDict

    def pccHeatmap(self, data):
        # 交叉計算PCC並存儲結果
        pcc_matrix = np.zeros((self.foldNum, self.foldNum))  # 初始化一個9x9的矩陣來存儲PCC值
        for i in range(self.foldNum):
            for j in range(self.foldNum):
                pcc_matrix[i, j] = np.corrcoef(data[i], data[j])[0, 1]

        # 繪製熱度圖
        plt.figure(figsize=(10, 8))
        plt.imshow(pcc_matrix, cmap='coolwarm', interpolation='nearest')
        cbar = plt.colorbar()  # 添加顏色條
        cbar.ax.tick_params(labelsize=20)  # 更改colorbar刻度數字大小
        plt.clim(0.5, 1)
        plt.title(f'{self.datasetName} PCC', fontsize = 25)
        plt.xlabel('GPT Group', fontsize = 25)
        plt.ylabel('GPT Group', fontsize = 25)
        plt.xticks(np.arange(self.foldNum), np.arange(1, self.foldNum + 1), fontsize = 20)  # x軸刻度
        plt.yticks(np.arange(self.foldNum), np.arange(1, self.foldNum + 1), fontsize = 20)  # y軸刻度
        fig = plt.gcf()
        fig.savefig(f"{self.outputPath}Analysis/PCC.png")
        plt.show()

    def aacHeatmap(self, data):
        # 交叉計算PCC並存儲結果
        pcc_matrix = np.zeros((self.foldNum, self.foldNum))  # 初始化一個9x9的矩陣來存儲PCC值
        for i in range(self.foldNum):
            for j in range(self.foldNum):
                pcc_matrix[i, j] = np.corrcoef(data[i], data[j])[0, 1]
        # 繪製熱度圖
        plt.figure(figsize=(10, 8))
        plt.imshow(pcc_matrix, cmap='coolwarm', interpolation='nearest')
        cbar = plt.colorbar()  # 添加顏色條
        cbar.ax.tick_params(labelsize=20)  # 更改colorbar刻度數字大小
        plt.clim(0.5, 1) #溫度的上限下限
        plt.title(f'{self.datasetName} AAC', fontsize = 25)
        plt.xlabel('GPT Group', fontsize = 25)
        plt.ylabel('GPT Group', fontsize = 25)
        plt.xticks(np.arange(self.foldNum), np.arange(1, self.foldNum + 1), fontsize = 20)  # x軸刻度
        plt.yticks(np.arange(self.foldNum), np.arange(1, self.foldNum + 1), fontsize = 20)  # y軸刻度
        fig = plt.gcf()
        fig.savefig(f"{self.outputPath}Analysis/AAC.png")
        plt.show()

    def fastaToSeqList(self, path):
        sequences = []
        for record in SeqIO.parse(path, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    def lengthCount(self, pepList):
        pepLengthDict = dict(Counter(len(item) for item in pepList))
        pepLengthDict = dict(sorted(pepLengthDict.items(), key=lambda x: x[0]))
        return pepLengthDict

    def binLength(self, originalDict, bins):
        scaleNames = [f"{bin[0]}-{bin[1]}" for bin in bins]
        groupedDict = {scaleName: 0 for scaleName in scaleNames}  # 初始化分组后的字典，所有值为0
        for length, count in originalDict.items():
            binKey = None
            for idx, binRange in enumerate(bins):
                if binRange[0] <= length <= binRange[1]:
                    binKey = idx
                    break
            if binKey is None:
                binKey = len(bins)
            groupName = scaleNames[binKey]  # 根据索引获取刻度名称
            groupedDict[groupName] += count  # 更新对应刻度的计数
        return groupedDict

    def aac_dpcAnalysis(self):
        with pd.ExcelWriter(f'{self.outputPath}/Analysis/{self.dpcAnalysisFileName}.xlsx') as writer:   #def的
            for i in range(1, self.foldNum + 1):
                foldPepList = self.fastaToSeqList(f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_gpt.fasta")
                foldAAC_Dict = self.countAminoAcidComposition(foldPepList)
                foldDPC_Dict = self.countDipeptideComposition(foldPepList)
                AACDf = pd.DataFrame(foldAAC_Dict.items(), columns=['Amino Acid', 'Count'])
                DPCDf = pd.DataFrame(foldDPC_Dict.items(), columns=['Dipeptide', 'Count'])
                AACDf.to_excel(writer, sheet_name=f'fold{i}', index=False)
                DPCDf.to_excel(writer, sheet_name=f'fold{i}', startrow=len(foldAAC_Dict) + 2, index=False)
        loadedAminoList = self.aminoList
        loadedDepeptideList = self.dipeptideList
        self.aacHeatmap(loadedAminoList)
        self.pccHeatmap(loadedDepeptideList)

    def lengthAnalysis(self):
        with pd.ExcelWriter(f'{self.outputPath}Analysis/{self.lengthAnalysisFileName}.xlsx') as writer: #外部輸入 加name suffix
            origPepList = self.fastaToSeqList(f"{self.outputPath}{self.datasetName}{self.outputSuffix}_orig.fasta")
            origPepLengthDict = self.lengthCount(origPepList)
            binPepLengthDict = self.binLength(origPepLengthDict, self.lengthBinList)
            origLengthDf = pd.DataFrame(origPepLengthDict.items(), columns=['Length', 'Count'])
            binOrigLengthDf = pd.DataFrame(binPepLengthDict.items(), columns=['Bin', 'Count'])
            origLengthDf.to_excel(writer, sheet_name=f'orig', index=False)
            binOrigLengthDf.to_excel(writer, sheet_name=f'orig', startrow=len(origPepLengthDict) + 2, index=False)
            for i in range(1, self.foldNum + 1):
                foldPepList = self.fastaToSeqList(f"{self.outputPath}{self.datasetName}x{i}{self.outputSuffix}_gpt.fasta")
                pepLengthDict = self.lengthCount(foldPepList)
                binLengthDict = self.binLength(pepLengthDict, self.lengthBinList)
                lengthDf = pd.DataFrame(pepLengthDict.items(), columns=['Length', 'Count'])
                binLengthDf = pd.DataFrame(binLengthDict.items(), columns=['Bin', 'Count'])
                lengthDf.to_excel(writer, sheet_name=f'fold{i}', index=False)
                binLengthDf.to_excel(writer, sheet_name=f'fold{i}', startrow=len(pepLengthDict) + 2, index=False)

    def main(self):
        self.aac_dpcAnalysis()
        self.lengthAnalysis()

Protgpt2Analysis = Protgpt2Analysis(outputPath="../data/IL13/",
                                    foldNum= 4,
                                    datasetName="IL13",
                                    outputSuffix="pos",
                                    lengthBinList=[[8,10], [11,15], [16,20], [21, 30], [31,38]],
                                    dpcAnalysisFileName="foldAAC_DPC",
                                    lengthAnalysisFileName="foldLengthAnalysis"
                                    )
Protgpt2Analysis.main()
