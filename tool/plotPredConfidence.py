import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PredConfidence:
    def __init__(self):
        pass

    @staticmethod
    def plotPredConfidence(predictionsList, trueLabelsList, numBins=10, modelNameList=None, outputExcel=None,
                           figSave=False, figSavePath=None):

        with pd.ExcelWriter(outputExcel, engine='openpyxl', mode='w') as writer:
            data = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            numPositiveData = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            totalSamplesData = {
                'Prediction Probability': [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)]}

            for i, predictions in enumerate(predictionsList):
                # 將預測值轉換為NumPy數組
                predictionsArr = np.array(predictions)

                # 將真實標籤轉換為NumPy數組
                trueLabelsArr = np.array(trueLabelsList)

                # 初始化y軸數據為0
                yValues = np.zeros(numBins)
                numPositiveList = []
                totalSamplesList = []

                # 遍歷每個區間
                for j in range(numBins):
                    # 獲取當前區間的上下界
                    lowerBound = j / numBins
                    upperBound = (j + 1) / numBins

                    # 將預測值為1.0的記錄在0.9-1.0區間
                    if upperBound == 1.0:
                        mask = (predictionsArr >= lowerBound) & (predictionsArr <= upperBound)
                    else:
                        mask = (predictionsArr >= lowerBound) & (predictionsArr < upperBound)

                    # 在該區間內計算真實positive的比例
                    numPositive = np.sum(trueLabelsArr[mask] == 1)
                    totalSamples = np.sum(mask)

                    # 計算並記錄y軸數據
                    if totalSamples > 0:
                        yValues[j] = numPositive / totalSamples
                        numPositiveList.append(numPositive)
                        totalSamplesList.append(totalSamples)

                data[modelNameList[i]] = yValues
                numPositiveData[modelNameList[i]] = numPositiveList
                totalSamplesData[modelNameList[i]] = totalSamplesList

                # 繪製圖形
                xValues = np.arange(numBins)

                plt.figure(figsize=(12, 9), dpi=300)
                plt.plot(xValues, yValues, marker='o')
                plt.xlabel('Prediction Probability')
                plt.ylabel('True Positive Rate')
                plt.title(f'Confidence Analysis - {modelNameList[i]}')

                plt.xticks(xValues, [f'{i / numBins:.1f}-{(i + 1) / numBins:.1f}' for i in range(numBins)])  # 設定x軸刻度

                if figSave:
                    plt.savefig(figSavePath + f'plotPredConfidence_{modelNameList[i]}.png')

                plt.show()

            if outputExcel:
                df1 = pd.DataFrame(data)
                df2 = pd.DataFrame(numPositiveData)
                df3 = pd.DataFrame(totalSamplesData)
                sheetData = {'plotPredConfidence': df1,
                             'numPositive': df2,
                             'totalSamples': df3}

                for sheet_name, df in sheetData.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)


import random

# 假設有一些預測值和真實標籤
predictions = [[round(random.uniform(0.00, 1.00), 2) for _ in range(100)]]

trueLabels = [random.choice([0, 1]) for _ in range(100)]
modelNameList = ['lightgbm']

# 呼叫函式進行畫圖分析
PredConfidence().plotPredConfidence(predictionsList=predictions, trueLabelsList=trueLabels, numBins=10,
                                    modelNameList=modelNameList, outputExcel="outputExcel.xlsx")
