import pandas as pd
import random
import numpy as np


class DataAugmentation:  # class input 1.trainData 2.能control正=1 負=0 正負都要=2 3.default = [0.85 - 0.65 step=0.5] or [0.8 0.7]
    def __init__(self, data, target, weightList, changeNumber, changeRatio):
        """
        changeNumber 與 changeRatio 如果都有值(不是 None 的話)會以 changeRatio 為優先
        """
        self.data = data
        self.target = target
        self.weightList = weightList
        self.changeNumber = changeNumber  # 4-1. 增加%數 (augRatio)or 4-2.幾條 (augNumber)給兩個變數
        self.changeRatio = changeRatio

    def run(self, savePath, outPutBinary=False):
        posIndex = list(self.data.loc[self.data['y'] == 1].index[:])
        if self.changeNumber & self.changeRatio is not None:
            self.changeNumber = None
        if self.changeRatio is not None:
            posNum = int(np.ceil(len(posIndex) * self.changeRatio))
            randomPosIndexList = random.sample(posIndex, posNum)   # random 要可以挑重複 ratio < 1 用sample >1 用 choices
        if self.changeNumber is not None:
            posNum = self.changeNumber
            randomPosIndexList = random.choices(posIndex, k=posNum)
        negIndex = list(self.data.loc[self.data['y'] == 0].index[:])
        data_X = self.data.drop(['y'], axis=1)
        resultSeriesList = []
        cutoffList = []
        if self.target == 0:
            newWeightList = list(filter(lambda x: x < 0.5, self.weightList))
        elif self.target == 1:
            newWeightList = list(filter(lambda x: x > 0.5, self.weightList))
        elif self.target == 2:
            newWeightList = self.weightList
        for randomPosIndex in randomPosIndexList:
            randomPosSeries = data_X.loc[randomPosIndex]
            cutoff = random.choice(newWeightList)  # list 裡面挑一個數值回傳 不要回傳list
            randomNegIndex = random.choice(negIndex)
            randomNegSeries = data_X.loc[randomNegIndex]
            posSeries = randomPosSeries * cutoff
            negSeries = randomNegSeries * (1-cutoff)   # 四段if變一段
            resultSeries = posSeries + negSeries
            resultSeriesList.append(resultSeries)
            cutoffList.append(cutoff)
        resultNumList = list(range(0, len(resultSeriesList)))
        resultStrList = ['result'] * len(resultSeriesList)
        modifiedList = list(map(lambda x, y: x + '_' + str(y), resultStrList, resultNumList))
        resultDf = pd.DataFrame(resultSeriesList, index=modifiedList)
        if outPutBinary:
            for i in range(len(cutoffList)):
                if cutoffList[i] < 0.5:
                    cutoffList[i] = 0
                else:
                    cutoffList[i] = 1
        resultDf['y'] = cutoffList
        newData = pd.concat([self.data, resultDf])
        newData.to_csv(savePath)

        return newData
