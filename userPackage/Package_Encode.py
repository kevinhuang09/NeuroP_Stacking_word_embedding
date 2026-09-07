from devPackage.PackageModelAmp import EncodeModelAmp
from devPackage.PackageiFeature import iFeature
from devPackage.PackagePFeature import PFeature
from devPackage.OVP import OVP
from devPackage.PackageBoruta import BorutaPackage
from devPackage.Normalization import Normalization
import pandas as pd
import pickle
import json
import os
from MLProcess.PycaretWrapper import PycaretWrapper
from devPackage.PackageCenterGDP import centerGDP
from userPackage.FeatureStat_word2Vec import WordEmbeddingFeature
from userPackage.FeatureStat_FastText import FastTextFeature


class EncodeAllFeatures:
    def __init__(self):
        self.skipFeatureList = None
        self.featureDict = None

    def dataEncodeSetup(self, saveFeatureDict=None, saveJsonPath=None, loadJsonPath=None, b_loadJson=False):
        if b_loadJson:
            path = loadJsonPath
            with open(path) as f:
                self.featureDict = json.load(f)
        else:
            path = saveJsonPath
            with open(path, 'w') as f:
                json.dump(saveFeatureDict, f)
                self.featureDict = saveFeatureDict

    def dataEncodeOutPut(self, dataDict):
        encodedDfList = []
        for label in [0, 1, -1]:
            inputData = dataDict[label]
            if inputData is not None:
                eifObj = iFeature(inputData, self.featureDict['iFeature'])
                epfObj = PFeature(inputData, self.featureDict['pFeature'])
                emaObj = EncodeModelAmp(inputData, self.featureDict['ampFeature'])  # windows 拉出去dict
                eovpObj = OVP(inputData, self.featureDict['ovpFeature'])
                eigObj = centerGDP(inputData, self.featureDict['centerGDPFeature'])
                ewordEmObj = WordEmbeddingFeature(inputData, self.featureDict['wordEmFeature'])
                efastTextObj = FastTextFeature(inputData, self.featureDict['fastTextFeature'])
                a = eifObj.getOutputDf()
                b = epfObj.getOutputDf()
                c = emaObj.getOutputDf()
                d = eovpObj.getOutputDf()
                g = eigObj.getOutputDf()
                h = ewordEmObj.getOutputDf()
                i = efastTextObj.getOutputDf()
                encodedDf = pd.concat([a, b], axis=1)
                encodedDf = pd.concat([encodedDf, c], axis=1)
                encodedDf = pd.concat([encodedDf, d], axis=1)
                encodedDf = pd.concat([encodedDf, g], axis=1)
                encodedDf = pd.concat([encodedDf, h], axis=1)
                encodedDf = pd.concat([encodedDf, i], axis=1)
                encodedDf.insert(encodedDf.shape[1], 'y', label)
                encodedDfList.append(encodedDf)
            else:
                pass
        if len(encodedDfList) == 2:
            outputDf = pd.concat([encodedDfList[0], encodedDfList[1]])
        else:
            outputDf = encodedDfList[0]

        return outputDf

    @staticmethod
    def dataNormalization(encodeTrainDf=None, encodeIndpDf=None, normalization='standard',
                          saveNmlzScalerPklPath='./data/', loadNmlzScalerPklPath='./data/', b_loadPkl=False):
        nmlzObj = Normalization(trainDf=encodeTrainDf, testDf=encodeIndpDf)
        if normalization.lower() == 'standard':
            if b_loadPkl:
                trainNmlzedDf = None
                indpNmlzedDf = nmlzObj.standardTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                indpNmlzedDf = None
                trainNmlzedDf, standardSca = nmlzObj.standard()
                path = saveNmlzScalerPklPath
                with open(path, 'wb') as f:
                    pickle.dump(standardSca, f)
        elif normalization.lower() == 'minmax':
            if b_loadPkl:
                trainNmlzedDf = None
                indpNmlzedDf = nmlzObj.minMaxTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                indpNmlzedDf = None
                trainNmlzedDf, minMaxSca = nmlzObj.minMax()
                path = saveNmlzScalerPklPath
                with open(path, 'wb') as f:
                    pickle.dump(minMaxSca, f)
        elif normalization.lower() == 'robust':
            if b_loadPkl:
                trainNmlzedDf = None
                indpNmlzedDf = nmlzObj.robustTest(loadParams=True, loadNmlzParamsPklPath=loadNmlzScalerPklPath)
            else:
                indpNmlzedDf = None
                trainNmlzedDf, robustSca = nmlzObj.robust()
                path = saveNmlzScalerPklPath
                with open(path, 'wb') as f:
                    pickle.dump(robustSca, f)

        if b_loadPkl:
            indpNmlzedDf = indpNmlzedDf.round(5)
            return indpNmlzedDf
        else:
            trainNmlzedDf = trainNmlzedDf.round(5)
            return trainNmlzedDf

    def dataBoruta(self, trainDf, borutaMethod='XGB', runBoruta=True, featRankPath="./data/", skipFeatureList=None):
        self.skipFeatureList = skipFeatureList
        if runBoruta:
            brtTrainDf = trainDf.drop(columns=self.skipFeatureList)
            brtObj = BorutaPackage(dataDf=brtTrainDf, modelName=borutaMethod.upper(), runBoruta=runBoruta,
                                        featRankPath=featRankPath + 'Boruta-featureRank-{}.csv'.format(borutaMethod))
        else:
            brtObj = BorutaPackage(dataDf=None, modelName=borutaMethod.upper(), runBoruta=runBoruta,
                                   featRankPath=featRankPath)
        return brtObj

    def dataEvalFeatureNum(self, startNum=50, endNum=500, step=20, featNumScorePath='../data/', saveCsvPath='../data/',
                           trainDf=None, indpDf=None, brtObj=None,foldNum=5,session = None):##添加foldNum
        bestScoreDf_mcc = pd.DataFrame()
        bestScoreDf_auc = pd.DataFrame()
        for featureNum in range(startNum, endNum, step):
            os.mkdir(saveCsvPath + str(featureNum))
            featureList = brtObj.numberRanks(featureNum) + self.skipFeatureList + ["y"]
            trainCSV = trainDf[featureList]
            indpCSV = indpDf[featureList]
            trainCSV.to_csv(saveCsvPath + str(featureNum) + "/train_F" + str(featureNum) + ".csv")
            indpCSV.to_csv(saveCsvPath + str(featureNum) + "/indp_F" + str(featureNum) + ".csv")
            trainData = pd.read_csv(saveCsvPath + str(featureNum) + "/train_F" + str(featureNum) + ".csv",
                                    index_col=[0])
            pycObj = PycaretWrapper()
            setupDf = pycObj.doSetup(trainData,sessionID = session)
            defaultModelParamList, scoreRank_mcc = pycObj.doCompareModel(fold = foldNum,
                includeModelList=['gbc', 'ridge', 'lr', 'catboost', 'lda', 'ada', 'knn', 'nb', 'et', 'lightgbm', 'rf',
                                  'xgboost', 'mlp', 'dt', 'svm', 'qda'])
            scoreRank_auc = scoreRank_mcc.sort_values(by='AUC', ascending=False)
            bestScoreDf_mcc.insert(loc=len(bestScoreDf_mcc.columns.tolist()), value=scoreRank_mcc.iloc[0],
                                   column=featureNum)
            bestScoreDf_auc.insert(loc=len(bestScoreDf_auc.columns.tolist()), value=scoreRank_auc.iloc[0],
                                   column=featureNum)
            scoreRank_mcc.to_csv(saveCsvPath + str(featureNum) + "/CV_" + str(featureNum) + "_byMcc.csv")
            scoreRank_auc.to_csv(saveCsvPath + str(featureNum) + "/CV_" + str(featureNum) + "_byAuc.csv")
        bestScoreDf_mcc.T.to_csv(featNumScorePath + 'bestScore_mcc.csv')
        bestScoreDf_auc.T.to_csv(featNumScorePath + 'bestScore_auc.csv')
        # bestScoreDf_mcc.T.to_excel(featNumScorePath + '_mcc.xlsx')  # pip install openpyxl
        # bestScoreDf_auc.T.to_excel(featNumScorePath + '_auc.xlsx')  # pip install openpyxl

    def dataDecidedFeatureNum(self, featureNum, saveCsvPath="./data/", brtObj=None, trainDf=None, indpDf=None):
        featureList = brtObj.numberRanks(featureNum) + self.skipFeatureList + ['y']  # list 包含 feature 名稱
        if trainDf is not None:
            trainCSV = trainDf[featureList]
            trainCSV.to_csv(saveCsvPath + "/train_F" + str(featureNum) + ".csv")
            indpCSV = indpDf[featureList]
            indpCSV.to_csv(saveCsvPath + "/indp_F" + str(featureNum) + ".csv")
        else:
            indpCSV = indpDf[featureList]
            indpCSV.to_csv(saveCsvPath + "/indp_F" + str(featureNum) + ".csv")
