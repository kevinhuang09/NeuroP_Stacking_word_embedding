# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
# b_saveFastaSeqCountStat = True  # 是否印出並儲存 fasta 序列數量統計
# ======================================================================================================================

import matplotlib

def setMatplotlibBackend(disablePopup):
    """disablePopup=True: 使用Agg backend，禁止彈出圖表視窗；disablePopup=False: 維持預設backend，正常跳出視窗"""
    if disablePopup:
        matplotlib.use('Agg')

setMatplotlibBackend(disablePlotPopup)

import sys, os

def setupParentPath(enableVscodeParentPath):
    """
    enableVscodeParentPath=True:
    使用Pycharm, Vscode進行編譯時，把上一層目錄加入sys.path，方便import套件
    """
    if enableVscodeParentPath:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.join(current_dir, "..")
        sys.path.append(parent)

setupParentPath(useVscodeParentPath)


import json

from sklearn.model_selection import train_test_split

from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat

#  移到外面再把舊的加進空的dict裡面
ifeatureDict = {"AAC": True,
                "AAINDEX": False,  # 注意!!需等長
                "CKSAAGP": [True, 5],
                "CTDC": True,
                "CTDD": True,
                "CTDT": True,
                "CTriad": True,  # 會產生feature 343個
                "DDE": True,
                "DPC": True,
                "GAAC": True,
                "GDPC": True,
                "GTPC": True,
                "KSCTriad": [True, 0],  # 會產生feature 343個
                "QSOrder": [False, 30, 0.1],
                "TPC": False,  # 會有8000個feature 不建議開啟
                "SOCN": [False, 3],
                "APAAC": [True, 3, 0.05],
                "Geary": [True, 3],
                "Moran": [True, 3],
                "NMBroto": [True, 3],
                "CKSAAP": [True, 3],  # 會產生feature 1600個
                "BINARY": False,  # 注意!!需等長
                "PAAC": [True, 3, 0.05]
                }

PfeatureDict = {"DDOR": True,
                "RRI": True,
                "SER": True,
                "SEP": True,
                "SE": True,
                "QSO": [True, 3, 0.1]
                }  # gap=3 w=0.1

AMPfeatureDict = {"length": True,
                  "calculate_mw": [True, True],  # [0] = on(True)/off [1] = amide
                  "calculate_charge": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "charge_density": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "isoelectric_point": [True, True],  # [0] = on(True)/off [1] = amide
                  "instability_index": True,
                  "aromaticity": True,
                  "aliphatic_index": True,
                  "hydrophobic": True,
                  "aasi": True,
                  "abhprk": [True, 5],  # [0] = on(True)/off [1] = window
                  "argos": True,
                  "bulkiness": True,
                  "charge_phys": True,
                  "charge_acid": True,
                  "cougar": [True, 5],  # [0] = on(True)/off [1] = window
                  "ez": [True, 5],  # [0] = on(True)/off [1] = window
                  "flexibility": True,
                  "gravy": True,
                  "levitt_alpha": True,
                  "mss": True,
                  "msw": [True, 5],  # [0] = on(True)/off [1] = window
                  "pepArc": True,
                  "polarity": True,
                  "refractivity": True,
                  "tm_tend": True,
                  "z3": [True, 5],  # [0] = on(True)/off [1] = window
                  "z5": [True, 5],  # [0] = on(True)/off [1] = window
                  "formula": True,  # C,H,N,O,S atom composition
                  "boman_index": True,
                  "eisenberg": True,
                  "hopp_woods": True,
                  "janin": True,
                  "kytedoolittle": True
                  }

OVPfeatureDict = {"OVPC": True,
                  "OVP": [True, 4, 4]  # 兩個數為 N, C 端胺基酸數目  N= C=
                  }

MotifBitVecfeatureDict = {"Usage": True,
                          "motifList": ['FKK', 'LKL', 'KKLL', 'KWK', 'VLK',
                                        'CY'
                                        ''
                                        'CR', 'CRR', 'RFC', 'RRR', 'LKKL']
                          }

centerGDPDict = {"Usage": False, "UseGap": False, "gap_size": -1}  # 若是預測每個 amino acid 的 label 在使用

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'motifBitVecFeature': MotifBitVecfeatureDict,
               'centerGDPFeature': centerGDPDict}

mlDataPath = "../data/mlData/"  # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
normalizeMethod = 'standard'
dataName = 'NeuroP_1'

# DataSet載入區（與 Main_FeatureStk.py 相同資料集）
MainDatasetNegFastaPath = "../data/MainDatasetNeg.fasta"
MainDatasetPosFastaPath = "../data/MainDatasetPos.fasta"
DS_IndpNegFastaPath = "../data/DS_IndpNeg.fasta"
DS_IndpPosFastaPath = "../data/DS_IndpPos.fasta"

splitTestSize = 0.1
splitRandomState = 42

def splitSeqDict(seqDict, test_size, random_state):
    keys = list(seqDict.keys())
    trainKeys, valKeys = train_test_split(keys, test_size=test_size, random_state=random_state)
    return {k: seqDict[k] for k in trainKeys}, {k: seqDict[k] for k in valKeys}

ldObj = LoadDataset(minSeqLength=5)
MainDatasetNegSeqDict = ldObj.readFasta(MainDatasetNegFastaPath)
MainDatasetPosSeqDict = ldObj.readFasta(MainDatasetPosFastaPath)
indpNegSeqDict2 = ldObj.readFasta(DS_IndpNegFastaPath)
indpPosSeqDict2 = ldObj.readFasta(DS_IndpPosFastaPath)

# 依 9:1 切分為 train / val（neg, pos 各自切分以維持類別比例）
trainNegSeqDict, indpNegSeqDict = splitSeqDict(MainDatasetNegSeqDict, splitTestSize, splitRandomState)
trainPosSeqDict, indpPosSeqDict = splitSeqDict(MainDatasetPosSeqDict, splitTestSize, splitRandomState)

trainDataDict = {0: trainNegSeqDict, 1: trainPosSeqDict, -1: None}
indpDataDict = {0: indpNegSeqDict, 1: indpPosSeqDict, -1: None}
indpDataDict2 = {0: indpNegSeqDict2, 1: indpPosSeqDict2, -1: None}  # 第二組 indp test data (DS_Indp，外部獨立測試集)

encodeObj = EncodeAllFeatures()

encodeObj.dataEncodeSetup(saveFeatureDict=featureDict,  # normalization 前傳出來
                          saveJsonPath=paramPath + f'{dataName}_featureTypeDict.json',  # 把 featureDict 存至 json 檔
                          loadJsonPath=None,  # 讀取 featureDict 的 pkl 檔
                          b_loadJson=False)  # True: 讀取 featureDict 的 pkl 檔 (loadJsonPath), False: 把 featureDict 存至 pkl 檔 (saveJsonPath)

encodeTrainDf = encodeObj.dataEncodeOutPut(dataDict=trainDataDict)
encodeIndpDf1 = encodeObj.dataEncodeOutPut(dataDict=indpDataDict)
encodeIndpDf2 = encodeObj.dataEncodeOutPut(dataDict=indpDataDict2)  # 第二組 indp test data

# ======================================================================================================================
# normalization
nmlzScalerPath = paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl'

delNanTrainDf = FeatureStat.delNan(data=encodeTrainDf, logPath="../data/mlData/delNanTrain.txt")
delNanIndpDf = FeatureStat.delNan(data=encodeIndpDf1, logPath="../data/mlData/delNanIndp.txt")
delNanIndpDf2 = FeatureStat.delNan(data=encodeIndpDf2, logPath="../data/mlData/delNanIndp2.txt")

trainNmlzDf = encodeObj.dataNormalization(encodeTrainDf=encodeTrainDf,
                                          encodeIndpDf=None,  # train scaler存起來 ，indp 另外做
                                          normalization=normalizeMethod,
                                          saveNmlzScalerPklPath=nmlzScalerPath,
                                          loadNmlzScalerPklPath=None,
                                          b_loadPkl=False)  # True: 讀取 NmlzScaler 的 pkl 檔 (loadNmlzScalerPklPath)
# False: 把 NmlzScaler 存至 pkl 檔 (saveNmlzScalerPklPath)

indpNmlzDf1 = encodeObj.dataNormalization(encodeTrainDf=None,
                                          encodeIndpDf=encodeIndpDf1,
                                          normalization=normalizeMethod,
                                          saveNmlzScalerPklPath=None,
                                          loadNmlzScalerPklPath=nmlzScalerPath,
                                          b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔

indpNmlzDf2 = encodeObj.dataNormalization(encodeTrainDf=None,
                                          encodeIndpDf=encodeIndpDf2,
                                          normalization=normalizeMethod,
                                          saveNmlzScalerPklPath=None,
                                          loadNmlzScalerPklPath=nmlzScalerPath,
                                          b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔

# ================================================================================================================
# 把全部 feature 做完 nmlz 的結果存成 csv 檔 (可檢查 feature cutoff 以及 nmlz 的結果)
featureStatPath = '../data/featureStat/'
trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'
featureAnalysisXlsxPath = featureStatPath + "featureAnalysis.xlsx"

trainNmlzDf.to_csv(trainNmlzCsvPath)  # 變數名稱不用ed
indpNmlzDf1.to_csv(indpNmlzCsvPath)

# Feature Stat 分析
featureStatObj = FeatureStat(dataPath=trainNmlzCsvPath)
# featureStatObj = FeatureStat(dataDf=trainNmlzDf)         # 也可 input dataframe
featureStatObj.sdAnalysis(
    saveFigPath=featureStatPath + f"sd_analysis_{dataName}_{normalizeMethod}.jpg")  # std deviation 分析結果 output 圖片
featureStatObj.featureValuePct_analysis(saveFinalExcel=featureAnalysisXlsxPath)  # 分析結果 output xlsx 檔
#featureStatObj.pepCompositionAnalysis(posFastaPath="../data/HemoPI_1_pos_main80%.fasta",
#                                      negFastaPath="../data/HemoPI_1_neg_main80%.fasta",
#                                      saveXlsxPath=featureStatPath + 'pepCompositionAnalysis.xlsx')  # 找出只含有 1, 2 or 3 個 amino acid 的 peptide
# output xlsx 檔

# filteredTrainNmlzDf 為篩選後的 nmlz dataframe, 跑後續 boruta 用  (紀錄剩下幾個 feature)
filteredTrainNmlzDf, removeList = featureStatObj.processData(xlsxPath=featureAnalysisXlsxPath, columnName='top1percent',
                                                             number='+0.98', protectFeatSubstringList=['MotifBitVec'])
featureStatObj.processDataLog(logPath='../data/mlData/')
filterTrainNmlzPath = featureStatPath + f'filtered_train_{dataName}_{normalizeMethod}.csv'  # 移除完 feature 後新的 nmlz dataset 檔, 跑後續 boruta 用
removeFeatureListPath = featureStatPath + f'remove_feature_list_{dataName}_{normalizeMethod}.json'  # 移除掉的 feature 會存在這個文字檔裡
filteredTrainNmlzDf.to_csv(filterTrainNmlzPath)
with open(removeFeatureListPath, 'w') as f:
    json.dump(removeList, f)
# ================================================================================================================
skipFeatureList = [s for s in filteredTrainNmlzDf.columns if s.__contains__("MotifBitVec")]
brtObj = encodeObj.dataBoruta(borutaMethod='XGB', runBoruta=True, featRankPath=mlDataPath,
                              trainDf=filteredTrainNmlzDf, skipFeatureList=skipFeatureList)

featureNumEnd = filteredTrainNmlzDf.shape[1] - 1 - len(skipFeatureList)  # 扣掉 y 欄位與 skip 的 feature，即實際可供 boruta 排序的 feature 數量

# encodeObj.dataEvalFeatureNum(startNum=50, endNum=featureNumEnd, step=20,
#                              featNumScorePath=mlDataPath, saveCsvPath=mlDataPath,
#                              trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf1, brtObj=brtObj, foldNum=5, session = None)   #sessionID可修改成任意整數，ex:1,4,10,15...



encodeObj.dataDecidedFeatureNum(featureNum=790, saveCsvPath=mlDataPath,
                                trainDf=filteredTrainNmlzDf, indpDf=indpNmlzDf1,
                                brtObj=brtObj)   # 決定好 feature 數字請開這個
