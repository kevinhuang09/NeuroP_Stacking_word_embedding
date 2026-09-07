from sklearn.metrics import accuracy_score, matthews_corrcoef
import pandas as pd
import matplotlib.pyplot as plt
from userPackage.LoadDataset import LoadDataset
import os


class PepPropertyAnalysis:
    def __init__(self, predResultDf, answerDf, modelNameList, ppRatioOutputDir, ppRatioOutputPicDir, ratioRangeDict, proGroupDict, propertyAccFileName, propertyMccFileName):
        self.modelNameList = []
        if modelNameList is None:
            modelNameList = ['rbfsvm', 'gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm',
                             'rf', 'xgboost', 'gpc', 'mlp', 'dt', 'svm', 'qda']
        for modelName in modelNameList:
            self.modelNameList.append(modelName.lower())
        self.answerDf = answerDf
        self.predResultDf = predResultDf
        self.aaRatioPartDict = {}
        self.ratioRangeDict = ratioRangeDict
        self.proGroupDict = proGroupDict
        self.ratioPath = ppRatioOutputDir
        if not os.path.isdir(self.ratioPath):
            os.mkdir(self.ratioPath)
        self.ratioPicPath = ppRatioOutputPicDir
        if not os.path.isdir(self.ratioPicPath):
            os.makedirs(self.ratioPicPath)
        self.accExcel = pd.ExcelWriter(self.ratioPath + propertyAccFileName, engine='openpyxl', mode='w')
        self.mccExcel = pd.ExcelWriter(self.ratioPath + propertyMccFileName, engine='openpyxl', mode='w')

    def closeExcel(self):
        self.accExcel.close()
        self.mccExcel.close()

    def addSeqAndLabelToDf(self, indpNegSeqDic=None, indpPosSeqDic=None):  # 添加true Label
        self.predResultDf = pd.concat([self.predResultDf, self.answerDf], axis=1)
        indpList = []
        if indpNegSeqDic is not None:
            for name in indpNegSeqDic.values():
                indpList.append(name)
        if indpPosSeqDic is not None:
            for name in indpPosSeqDic.values():
                indpList.append(name)
        self.predResultDf.insert(0, 'seq', indpList)
        return

    def calRatioInPep(self, protGroup):
        aaRatioList = []
        ratioList = self.ratioRangeDict[protGroup]
        ratioRangeList = self.calParNumParList(ratioList)  # 算區間
        aaNumPartDcit = {}
        for part in ratioRangeList:
            self.aaRatioPartDict[protGroup + part] = []
            aaNumPartDcit[part] = []
        for pepName, row in self.predResultDf.iterrows():
            aaNum = 0
            seq = self.predResultDf.at[pepName, 'seq']  # 增加seq名
            for word in seq:
                if word not in self.proGroupDict[protGroup]:
                    continue
                aaNum += 1
            aaRatio = aaNum / len(seq)  # 計算Ratio
            aaRatioList.append(aaRatio)
            # 分區間
            aaNumPartDcit = self.parRatioScore(protGroup, aaRatio, ratioList, ratioRangeList, pepName,aaNumPartDcit)
        self.predResultDf[protGroup + 'ratio'] = aaRatioList
        if os.path.isfile(self.ratioPath + 'pred_ResultDf.csv'):
            print(f"Overriding{self.ratioPath + 'pred_ResultDf.csv'}")
        self.predResultDf.to_csv(self.ratioPath + 'pred_ResultDf.csv')
        parRatioDf = self.calBinScoing(protGroup, ratioRangeList,aaNumPartDcit)
        return parRatioDf

    def calParNumParList(self, aaPartScoreList):  # 計算區間aaParScoreList有幾個
        parNum = len(aaPartScoreList)
        aaPartList = []
        for par in range(parNum):
            if par == 0:
                continue
            aaPartList.append(str(aaPartScoreList[par - 1]) + '~' + str(aaPartScoreList[par]))
        return aaPartList

    ###改成依aaPartScoreList區間，將aaNumPartDict存seq,aaRatioParDict存index
    def parRatioScore(self, proGroup, aaRatio, ratioRangeList, aaPartList, pepName,aaNumPartDcit):
        for par in range(len(ratioRangeList)):
            if aaRatio == ratioRangeList[0]:
                self.aaRatioPartDict[proGroup + aaPartList[0]].append(pepName)
                aaNumPartDcit[aaPartList[0]].append(pepName)
                break
            if aaRatio == ratioRangeList[-1]:
                self.aaRatioPartDict[proGroup + aaPartList[-1]].append(pepName)
                aaNumPartDcit[aaPartList[-1]].append(pepName)
                break
            if aaRatio < ratioRangeList[par]:
                self.aaRatioPartDict[proGroup + aaPartList[par - 1]].append(pepName)
                aaNumPartDcit[aaPartList[par - 1]].append(pepName)
                break
        return aaNumPartDcit

    def calBinScoing(self, aaGroup, ratioRangeList,aaNumPartDcit):
        parRatioDf = pd.DataFrame()
        for name in self.modelNameList:
            accList = []
            mccList = []
            for part in ratioRangeList:
                ans = []
                pred = []
                for index in self.aaRatioPartDict[aaGroup + part]:  # 抓間格內的某model的ans,pred
                    pred.append(self.predResultDf.at[index, name])
                    ans.append((self.predResultDf.at[index, 'y']))
                acc, mcc = self.doScoring(ans, pred)
                accList.append(acc)
                mccList.append(mcc)
            parRatioDf[aaGroup + '_' + name + '_' + 'acc'] = accList
            parRatioDf[aaGroup + '_' + name + '_' + 'mcc'] = mccList
        geneTypeAccDf = self.writeAccXls(aaGroup, ratioRangeList, parRatioDf,aaNumPartDcit)
        geneTypeMccDf = self.writeMccXls(aaGroup, ratioRangeList, parRatioDf,aaNumPartDcit)
        self.drawDataPlot('acc', aaGroup, geneTypeAccDf, ratioRangeList)
        self.drawDataPlot('mcc', aaGroup, geneTypeMccDf, ratioRangeList)
        parRatioDf.index = [ratioRangeList]
        return parRatioDf

    def doScoring(self, ans, pred):  # 計算acc,mcc
        acc = accuracy_score(ans, pred)
        mcc = matthews_corrcoef(ans, pred)
        return acc, mcc

    def writeAccXls(self, aaGroup, aaPartList, parRatioDf,aaNumPartDcit):  # 寫excel檔
        # acc
        accDf = pd.DataFrame()
        numList = []
        for model in self.modelNameList:
            valueList = []
            for value in parRatioDf[aaGroup + '_' + model + '_' + 'acc']:
                valueList.append(value)
            accDf[model] = valueList
        for num in aaNumPartDcit.values():
            numList.append(len(num))
        accDf['seqNumber'] = numList
        accDf.index = aaPartList
        accDf.to_excel(self.accExcel, sheet_name=f'{aaGroup}_acc')
        # self.accExcel.close()
        return accDf

    def writeMccXls(self, aaGroup, aaPartList, parRatioDf,aaNumPartDcit):  # 寫excel檔
        # mcc
        mccDf = pd.DataFrame()
        numList = []
        for model in self.modelNameList:
            valueList = []
            for value in parRatioDf[aaGroup + '_' + model + '_' + 'mcc']:
                valueList.append(value)
            mccDf[model] = valueList
        for num in aaNumPartDcit.values():
            numList.append(len(num))
        mccDf['seqNumber'] = numList
        mccDf.index = aaPartList
        mccDf.index = aaPartList
        mccDf.to_excel(self.mccExcel, sheet_name=f'{aaGroup}_mcc')
        # self.mccExcel.close()
        return mccDf

    def drawDataPlot(self, score, proGroup, geneTypeDf, aaPartList):  # 畫圖
        for model in self.modelNameList:
            listx = aaPartList
            listy2 = geneTypeDf.loc[aaPartList, model]  # ratio
            listy1 = []
            for par in aaPartList:
                numList = []
                for num in self.aaRatioPartDict[proGroup + par]:
                    numList.append(num)
                listy1.append(len(numList))
            self.drawPlot(model, score, proGroup, listx, listy1, listy2)
        return

    # acc跟number反過來
    def drawPlot(self, model, score, protGroup, listx, listy1, listy2):
        plt.figure(figsize=(10, 6))
        plt.tick_params(axis='both', labelsize=13)
        plt.plot(listx, listy2, label=score, color='red', marker="o", markersize=8)
        plt.xlabel('')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left', fontsize=15)
        plt.title(f'{score} of {protGroup}_{model}', fontsize=22)
        ax2 = plt.twinx()
        ax2.bar(listx, listy1, label='Number', alpha=0.5)
        ax2.set_ylabel('Number')
        plt.legend(loc='upper right', fontsize=15)
        plt.grid(False)  # 關閉格線
        if os.path.isfile(self.ratioPicPath + f"PepProperty of {protGroup}_{model}_{score}.png"):
            print(f'即將覆蓋{self.ratioPicPath + f"PepProperty of {protGroup}_{model}_{score}.png"}')
        plt.savefig(self.ratioPicPath + f"PepProperty of {protGroup}_{model}_{score}.png")  # 儲存圖片
        plt.show()
        return


#modelNameList = ['catboost', 'gbc', 'et']
modelNameList = ['lr','dt','knn']
outputDir = '../data/PeptideProperty/'
outputPicDir = "../data/PeptideProperty/Pic/"
mlDataPath = "../data/mlData/"   # 內含 feature selection 後的 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
dataIndp = pd.read_csv(mlDataPath + f'/50/' + "indp_F" + str(50) + ".csv",
                       index_col=[0])  # 例如檔名為Indp_F190.csv
dataIndpX_df = dataIndp.drop(["y"], axis=1)
dataIndpY_df = dataIndp[["y"]]

aaGroupDict = {'Hydrophobic': ['V', 'I', 'L', 'M', 'F', "W", "C"],
                 'Hydrophilic': ["R", "D", 'E', 'H', 'K', 'N', 'Q', "T"],
                 'Charged': ["R", "D", 'E', 'H', 'K'],
               'PosCharged': ["R", 'H', 'K'],
               'NegCharged': ["D", 'E'],
               'Polar': ["C", "Q", 'N', 'Y', 'T', 'S'],
               'NonPolar': ["G", "A", 'V', 'L', 'I', "M", 'F', 'W', 'P']}



ratioRangeDict = {'Hydrophobic': [0, 0.3, 0.5, 1],
                  'Hydrophilic': [0, 0.3, 0.5, 0.7, 1],
                  'Charged': [0, 0.3, 0.5, 0.7, 1],
                  'PosCharged': [0, 0.1, 0.3, 1],
                  'NegCharged': [0, 0.08, 0.1, 1],
                  'Polar': [0, 0.15, 0.35, 1],
                  'NonPolar': [0, 0.3, 0.7, 1]}
bestPredVectorDf = pd.read_csv('../data/predResultDf.csv', index_col=[0])
ldObj = LoadDataset(minSeqLength=5)
negSeqDict = ldObj.readFasta("../data/HemoPI_1_neg_val20%.fasta")
posSeqDict = ldObj.readFasta("../data/HemoPI_1_pos_val20%.fasta")

ppaObj = PepPropertyAnalysis(predResultDf=bestPredVectorDf, answerDf=dataIndpY_df, modelNameList=modelNameList,
                             ppRatioOutputDir=outputDir, ppRatioOutputPicDir=outputPicDir, ratioRangeDict=ratioRangeDict,
                             proGroupDict=aaGroupDict, propertyAccFileName='propertyAcc.xlsx', propertyMccFileName='propertyMcc.xlsx')
# 加True Label,seq
ppaObj.addSeqAndLabelToDf(negSeqDict, posSeqDict)  # 不用回傳
# 計算分數並分區間
for aaGroup in aaGroupDict.keys():
    analysisResultDf = ppaObj.calRatioInPep(aaGroup)
    if os.path.isfile(outputDir + f'{aaGroup}_RangeScore.csv'):
        print(f"即將覆蓋{outputDir + f'{aaGroup}_RangeScore.csv'}")
    analysisResultDf.to_csv(outputDir + f'{aaGroup}_RangeScore.csv')
ppaObj.closeExcel()
