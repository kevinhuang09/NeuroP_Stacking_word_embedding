# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
useTrainValMeta = False  # True: 讀取 Lv1 predictOnTrainToo=True 存的 DS_Train+DS_Val Meta-Feature-Matrix（檔名多 _trainval）
                        # False: 沿用原本只有 DS_Val 的 Meta-Feature-Matrix 當 Lv2 訓練資料
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

import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring

# ======================================================================================================================
# 基本路徑：直接讀取 Main_MLStkLv1_debug_test.py 存的 Meta-Feature-Matrix（DS_Val 預測機率表）當作 Lv2 的訓練資料，
# 對每一欄 (feature type × model) 的機率做 Boruta 特徵排序，篩選出真正有用的 meta-feature 組合。
# 參考 main_Feature_v2.py 的 Boruta 流程（encodeObj.dataBoruta(...)），只做排序，不做 dataEvalFeatureNum 那段
# feature 數量掃描（那段需要同時有 train/indp 兩份資料，Lv1 debug_test 目前只產生 DS_Val 的 Meta-Feature-Matrix）。
mlDataPath = "../data/mlData/"  # Boruta 排序結果 Boruta-featureRank-{method}.csv、決定好的 train_F{N}/indp_F{N} 存這裡
mlScorePath = "../data/mlScore/"  # 內含 Main_MLStkLv1_debug_test.py 存的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
tuneModelPath = "../data/tuneModel/"  # 跟 Lv1 共用根目錄，Lv2 的 model 存在 Lv2/{normalizeMethod}/ 子資料夾，不會跟 Lv1 的 feature type 資料夾撞名
finalModelPath = "../data/finalModel/"  # Lv2 finalize 好的 model
dataName = 'NeuroP_1'

# 要跟 Main_FeatureStk_debug.py / Main_MLStkLv1_debug_test.py 的 normalizeMethodList 保持一致，
# 否則會去讀一份不存在的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
normalizeMethodList = ['robust']

borutaMethod = 'XGB'  # Boruta 底層估計器：'XGB' / 'RF' / 'LGB'，跟 main_Feature_v2.py 一致用 XGB
decidedFeatureNum = 160  # Boruta 排序後，決定拿前幾個 meta-feature 來訓練 Lv2 model

# Lv2 的 base learner 沿用 Lv1 debug 用過的 17 個 model
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

os.makedirs(mlDataPath, exist_ok=True)

encodeObj = EncodeAllFeatures()

for normalizeMethod in normalizeMethodList:
    metaFeatureMatrixSuffix = '_trainval' if useTrainValMeta else ''
    metaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
    metaFeatureMatrixDf = pd.read_csv(metaFeatureMatrixPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 Lv1 Meta-Feature-Matrix（{metaFeatureMatrixPath}）：{metaFeatureMatrixDf.shape}")

    # 每一欄都是某個 (feature type × model) 的預測機率，全部都要參與 Boruta 排序，沒有需要保護、跳過的欄位
    skipFeatureList = []
    featRankPrefix = mlDataPath + f'Lv2_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}_'
    brtObj = encodeObj.dataBoruta(borutaMethod=borutaMethod, runBoruta=True, featRankPath=featRankPrefix,
                                  trainDf=metaFeatureMatrixDf, skipFeatureList=skipFeatureList)

    featRankCsvPath = featRankPrefix + f'Boruta-featureRank-{borutaMethod}.csv'
    print(f"[{normalizeMethod}] Boruta 排序結果（{len(brtObj.feature_sort)} 個 meta-feature）已儲存到 {featRankCsvPath}")
    print(brtObj.feature_sort)

    # ==================================================================================================================
    # dataEvalFeatureNum：依 Boruta 排序，掃描不同 meta-feature 數量（startNum~endNum，每次間隔 step）分別做一次
    # CV 比較，把每個數量下最好的分數存到 bestScore_mcc.csv / bestScore_auc.csv，方便挑選最終要用幾個 meta-feature。
    # 需要同時有 train（metaFeatureMatrixDf）跟 indp 兩份 Meta-Feature-Matrix：indp 版本目前 Main_MLStkLv1_debug_test.py
    # 還沒有產生，要先讓 Lv1 也對 DS_Indp 做一次預測、存成 Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv，
    # 這裡才讀得到，否則下面這行會丟 FileNotFoundError。
    indpMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv'
    indpMetaFeatureMatrixDf = pd.read_csv(indpMetaFeatureMatrixPath, index_col=[0])

    # encodeObj.dataEvalFeatureNum(startNum=5, endNum=len(brtObj.feature_sort) + 1, step=5,
    #                              featNumScorePath=featRankPrefix, saveCsvPath=featRankPrefix,
    #                              trainDf=metaFeatureMatrixDf, indpDf=indpMetaFeatureMatrixDf,
    #                              brtObj=brtObj, foldNum=5, session=None)  # sessionID可修改成任意整數，ex:1,4,10,15...

    # dataDecidedFeatureNum 內部是用 saveCsvPath + "/train_F{N}.csv" 存檔，等同把 featRankPrefix 當資料夾用，
    # 所以要先把這個資料夾建出來，不然 to_csv 會因為資料夾不存在而丟 FileNotFoundError
    os.makedirs(featRankPrefix, exist_ok=True)
    encodeObj.dataDecidedFeatureNum(featureNum=decidedFeatureNum, saveCsvPath=featRankPrefix,
                                    trainDf=metaFeatureMatrixDf, indpDf=indpMetaFeatureMatrixDf,
                                    brtObj=brtObj)

    # ==================================================================================================================
    # 用 Boruta 決定好的 feature 數量（train_F{N}.csv / indp_F{N}.csv）訓練 Lv2 model：
    # base learner 沿用 Lv1 debug 用過的 17 個 model，一起丟給 PycaretWrapper 做 tune + CV 比較，
    # 再 finalize、存檔，最後對 DS_Indp 做 predict 並用真實 y 算分。
    decidedTrainCsvPath = featRankPrefix + f'/train_F{decidedFeatureNum}.csv'
    decidedIndpCsvPath = featRankPrefix + f'/indp_F{decidedFeatureNum}.csv'
    decidedTrainDf = pd.read_csv(decidedTrainCsvPath, index_col=[0])
    decidedIndpDf = pd.read_csv(decidedIndpCsvPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 Boruta 篩選後（{decidedFeatureNum} 個 meta-feature）的資料："
          f"train={decidedTrainDf.shape}, indp={decidedIndpDf.shape}")

    lv2PycObj = PycaretWrapper()
    lv2PycObj.doSetup(trainData=decidedTrainDf, sessionID=42)
    lv2PycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=5,
                         n_iter=10, early_stopping=False, customGridDict=None)

    lv2TuneSavePath = os.path.join(tuneModelPath, 'Lv2', normalizeMethod + metaFeatureMatrixSuffix)
    os.makedirs(lv2TuneSavePath, exist_ok=True)
    lv2PycObj.doSaveModel(lv2TuneSavePath, b_isFinalizedModel=False)  # 儲存 tune 好的 model

    _, lv2CvScoreRank = lv2PycObj.doCompareModel(fold=5, includeModelList=lv2PycObj.tunedModelList)
    lv2CvScoreCsvPath = mlScorePath + f'Lv2_cvScore_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
    lv2CvScoreRank.to_csv(lv2CvScoreCsvPath)
    print(f"[{normalizeMethod}] Lv2 17 個 base learner 的 CV 分數已儲存到 {lv2CvScoreCsvPath}")

    lv2PycObj.doFinalizeModel()  # train + self test 合併重新 fit
    lv2FinalSavePath = os.path.join(finalModelPath, 'Lv2', normalizeMethod + metaFeatureMatrixSuffix)
    os.makedirs(lv2FinalSavePath, exist_ok=True)
    lv2PycObj.doSaveModel(lv2FinalSavePath, b_isFinalizedModel=True)  # 儲存 finalize 好的 model
    lv2FinalModelList = lv2PycObj.finalModelList

    # 用 finalize 好的 model 對 DS_Indp（Boruta 篩選後的 meta-feature）做 predict，跟真實 y 算分
    decidedIndp_X = decidedIndpDf.drop(columns=['y'])
    decidedIndp_y = decidedIndpDf[['y']]
    lv2PredObjIndp = Predict(dataX=decidedIndp_X, modelList=lv2FinalModelList)
    lv2PredVectorListIndp, lv2ProbVectorListIndp = lv2PredObjIndp.doPredict()

    lv2ScoreObjIndp = Scoring(predVectorList=lv2PredVectorListIndp, probVectorList=lv2ProbVectorListIndp,
                             answerDf=decidedIndp_y, modelNameList=modelNameList)
    lv2IndpScoreCsvPath = mlScorePath + f'Lv2_indpScore_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
    lv2IndpScoreDf = lv2ScoreObjIndp.doScoring(b_optimizedMcc=False, path=lv2IndpScoreCsvPath, sortColumn='mcc')
    print(f"[{normalizeMethod}] Lv2 17 個 base learner 在 DS_Indp 上的分數已儲存到 {lv2IndpScoreCsvPath}")
    print(lv2IndpScoreDf)
