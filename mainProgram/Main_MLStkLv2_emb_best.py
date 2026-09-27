# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
useTrainValMetaForSort = False  # Boruta 排序依據：True 用 DS_Train+DS_Val 的 Meta-Feature-Matrix 算重要性排序，False 只用 DS_Val
useTrainValMetaForTrain = False  # 最終 Lv2 訓練資料（dataDecidedFeatureNum 篩出來的 train_F{N}.csv）：
                                 # True 用 DS_Train+DS_Val，False 只用 DS_Val
                                 # 兩者互相獨立：可以用 DS_Train+DS_Val 排序找重要 feature，但最終只拿 DS_Val 訓練，反之亦然
                                 # True 都是讀取 Lv1 predictOnTrainToo=True 存的 Meta-Feature-Matrix（檔名多 _trainval）

# 跟 Main_MLStkLv2_emb.py 最大的差異：tune_try 決定同一份 train/indp 資料要重複「tune + finalize + 對 DS_Indp 評分」
# 這一整套流程幾次，每次只換 sessionID(讓 pycaret 的 train/holdout 切分跟 tune_model 的搜尋過程都不一樣)，
# 藉此觀察 tuning 結果的變異程度。
#
# 這支程式支援「分好幾次執行、累積嘗試次數」：
#   - sessionID 不是每次執行都從 baseSessionID 重新開始，而是讀取已累積的總表(summaryCsv)，
#     接著歷史上用過的最大 sessionID 繼續往下編號，避免重跑到同一批 sessionID、浪費運算。
#   - 每次 try 自己表現最好(indp mcc 最高)的那個 model，會立刻存到 Lv2_try_history/{normalizeMethod}/ 底下
#     (檔名帶 sessionID)，不等到全部 tuneTry 跑完才決定要不要存，這樣不管這個 process 中途被中斷、
#     或是這次執行的結果比之前任何一次都差，之前任何一次 try 的 model 都不會遺失。
#   - 每次執行結束後，會把這次新增的結果 append 進 summaryCsv(不是覆蓋)，再從「所有歷史 + 這次新增」
#     的完整記錄裡重新挑出 indp mcc 全域最高的那一筆，把對應的 model 檔案複製到 Lv2_best/ 底下
#     (不需要重新訓練或重新載入，因為 Lv2_try_history 裡已經有現成的 model 檔案)。
tuneTry = 5  # 這次執行要新增幾次嘗試
baseSessionID = 42  # 從來沒有累積過任何結果時(summaryCsv 不存在)，第一次執行從這個 sessionID 開始
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
import shutil
import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring

# ======================================================================================================================
# 基本路徑：跟 Main_MLStkLv2_emb.py 共用同一份 Lv1 Meta-Feature-Matrix / Boruta 排序結果，
# 只有 tune 過程重複多次(tuneTry)這件事不一樣，所以 mlDataPath/mlScorePath 沿用同一份即可；
# 但輸出的 model／score 檔名都額外加上 best，避免跟 Main_MLStkLv2_emb.py 單次跑出來的結果互相覆蓋。
mlDataPath = "../data/mlData_emb/"  # Boruta 排序結果 Boruta-featureRank-{method}.csv、決定好的 train_F{N}/indp_F{N} 存這裡
mlScorePath = "../data/mlScore_emb/"  # 內含 Main_MLStkLv1_emb.py 存的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
finalModelPath = "../data/finalModel_emb/"  # 全域最佳 model 存這裡(子資料夾 Lv2_best，跟 Main_MLStkLv2_emb.py 的 Lv2 分開)
dataName = 'NeuroP_1'

# 要跟 Main_FeatureStk_emb.py / Main_MLStkLv1_emb.py 的 normalizeMethodList 保持一致，
# 否則會去讀一份不存在的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
normalizeMethodList = ['robust']

borutaMethod = 'XGB'  # Boruta 底層估計器：'XGB' / 'RF' / 'LGB'，跟 main_Feature_v2.py 一致用 XGB
decidedFeatureNum = 60  # Boruta 排序後，決定拿前幾個 meta-feature 來訓練 Lv2 model

# Lv2 的 base learner 沿用 Lv1 debug 用過的 17 個 model
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

os.makedirs(mlDataPath, exist_ok=True)

encodeObj = EncodeAllFeatures()

for normalizeMethod in normalizeMethodList:
    sortMetaSuffix = '_trainval' if useTrainValMetaForSort else ''
    trainMetaSuffix = '_trainval' if useTrainValMetaForTrain else ''

    sortMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}{sortMetaSuffix}.csv'
    sortMetaFeatureMatrixDf = pd.read_csv(sortMetaFeatureMatrixPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 Lv1 Meta-Feature-Matrix 當 Boruta 排序依據（{sortMetaFeatureMatrixPath}）：{sortMetaFeatureMatrixDf.shape}")

    if trainMetaSuffix == sortMetaSuffix:
        trainMetaFeatureMatrixDf = sortMetaFeatureMatrixDf
    else:
        trainMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}{trainMetaSuffix}.csv'
        trainMetaFeatureMatrixDf = pd.read_csv(trainMetaFeatureMatrixPath, index_col=[0])
        print(f"[{normalizeMethod}] 讀取 Lv1 Meta-Feature-Matrix 當最終 Lv2 訓練資料（{trainMetaFeatureMatrixPath}）：{trainMetaFeatureMatrixDf.shape}")

    # 排序依據跟最終訓練資料的開關不同時，檔名要能區分開來，避免互相覆蓋
    metaFeatureMatrixSuffix = sortMetaSuffix if sortMetaSuffix == trainMetaSuffix else \
        f"_sort{'TrainVal' if useTrainValMetaForSort else 'Val'}_train{'TrainVal' if useTrainValMetaForTrain else 'Val'}"

    # 每一欄都是某個 (feature type × model) 的預測機率，全部都要參與 Boruta 排序，沒有需要保護、跳過的欄位
    skipFeatureList = []
    featRankPrefix = mlDataPath + f'Lv2_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}_'
    brtObj = encodeObj.dataBoruta(borutaMethod=borutaMethod, runBoruta=True, featRankPath=featRankPrefix,
                                  trainDf=sortMetaFeatureMatrixDf, skipFeatureList=skipFeatureList)

    featRankCsvPath = featRankPrefix + f'Boruta-featureRank-{borutaMethod}.csv'
    print(f"[{normalizeMethod}] Boruta 排序結果（{len(brtObj.feature_sort)} 個 meta-feature）已儲存到 {featRankCsvPath}")
    print(brtObj.feature_sort)

    # ==================================================================================================================
    indpMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv'
    indpMetaFeatureMatrixDf = pd.read_csv(indpMetaFeatureMatrixPath, index_col=[0])

    # dataDecidedFeatureNum 內部是用 saveCsvPath + "/train_F{N}.csv" 存檔，等同把 featRankPrefix 當資料夾用，
    # 所以要先把這個資料夾建出來，不然 to_csv 會因為資料夾不存在而丟 FileNotFoundError
    os.makedirs(featRankPrefix, exist_ok=True)
    encodeObj.dataDecidedFeatureNum(featureNum=decidedFeatureNum, saveCsvPath=featRankPrefix,
                                    trainDf=trainMetaFeatureMatrixDf, indpDf=indpMetaFeatureMatrixDf,
                                    brtObj=brtObj)

    # ==================================================================================================================
    # 用 Boruta 決定好的 feature 數量（train_F{N}.csv / indp_F{N}.csv）訓練 Lv2 model，這份資料在 tuneTry 次嘗試中都固定不變，
    # 只有 sessionID(下面迴圈裡的 trySessionID) 每次嘗試都不一樣，讓 pycaret 的 train/holdout 切分跟 tune_model 搜尋過程各自獨立。
    decidedTrainCsvPath = featRankPrefix + f'/train_F{decidedFeatureNum}.csv'
    decidedIndpCsvPath = featRankPrefix + f'/indp_F{decidedFeatureNum}.csv'
    decidedTrainDf = pd.read_csv(decidedTrainCsvPath, index_col=[0])
    decidedIndpDf = pd.read_csv(decidedIndpCsvPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 Boruta 篩選後（{decidedFeatureNum} 個 meta-feature）的資料："
          f"train={decidedTrainDf.shape}, indp={decidedIndpDf.shape}")

    decidedIndp_X = decidedIndpDf.drop(columns=['y'])
    decidedIndp_y = decidedIndpDf[['y']]

    # 每個 try 自己最好的 model 存到這個資料夾(檔名帶 sessionID，跨執行也不會互相覆蓋)
    tryHistoryPath = os.path.join(finalModelPath, 'Lv2_try_history', normalizeMethod + metaFeatureMatrixSuffix)
    os.makedirs(tryHistoryPath, exist_ok=True)

    # 累積所有歷史執行結果的總表：存在就讀進來、接著編號；不存在代表這是第一次執行
    summaryCsvPath = mlScorePath + f'Lv2_indpScore_best_summary_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
    if os.path.exists(summaryCsvPath):
        historySummaryDf = pd.read_csv(summaryCsvPath)
        nextSessionID = int(historySummaryDf['sessionID'].max()) + 1
        print(f"[{normalizeMethod}] 讀到歷史累積結果（{summaryCsvPath}）共 {len(historySummaryDf)} 筆，"
              f"這次執行從 sessionID={nextSessionID} 繼續")
    else:
        historySummaryDf = pd.DataFrame()
        nextSessionID = baseSessionID
        print(f"[{normalizeMethod}] 沒有歷史累積結果，這次執行從 sessionID={nextSessionID} 開始")

    # combinedSummaryDf 從歷史紀錄開始，每跑完一次 try 就立刻 append 一列、立刻寫回 csv、
    # 立刻重新判斷要不要更新 Lv2_best，而不是等 tuneTry 次全部跑完才一次做完；
    # 這樣不管在第幾次 try 之後中斷，已經跑完的每一次都已經完整落地(summary.csv + Lv2_best 都同步)，不會遺失
    combinedSummaryDf = historySummaryDf.copy()

    def promoteGlobalBest(summaryDf):
        """依目前 summaryDf(累積到目前為止的所有 try)重新挑出 indp mcc 全域最高的一筆，
        把 Lv2_try_history 裡對應的 model 檔案複製到 Lv2_best/(直接覆蓋，不用重新訓練)"""
        globalBestRow = summaryDf.loc[summaryDf['mcc'].idxmax()]
        globalBestSessionID = int(globalBestRow['sessionID'])
        globalBestModelName = globalBestRow['modelName']

        lv2BestSavePath = os.path.join(finalModelPath, 'Lv2_best', normalizeMethod + metaFeatureMatrixSuffix)
        os.makedirs(lv2BestSavePath, exist_ok=True)
        srcModelPath = os.path.join(tryHistoryPath, f'session{globalBestSessionID}_{globalBestModelName}_final.pkl')
        dstModelPath = os.path.join(lv2BestSavePath, f'{globalBestModelName}_final.pkl')
        shutil.copy2(srcModelPath, dstModelPath)

        bestModelInfoPath = os.path.join(lv2BestSavePath, f'{globalBestModelName}_final_info.json')
        bestModelInfo = {
            'sessionID': globalBestSessionID,
            'modelName': globalBestModelName,
            'indpScore': {k: v for k, v in globalBestRow.items() if k not in ('modelName', 'sessionID')},
        }
        with open(bestModelInfoPath, 'w', encoding='utf-8') as f:
            json.dump(bestModelInfo, f, ensure_ascii=False, indent=2)

        print(f"[{normalizeMethod}] 目前累積 {len(summaryDf)} 次嘗試中的全域最佳（sessionID={globalBestSessionID}, "
              f"model={globalBestModelName}, indp mcc={globalBestRow['mcc']:.4f}）已同步到 {lv2BestSavePath}")

    for tryIdx in range(tuneTry):
        trySessionID = nextSessionID + tryIdx
        print(f"[{normalizeMethod}] ===== tune_try {tryIdx + 1}/{tuneTry}（sessionID={trySessionID}）=====")

        lv2PycObj = PycaretWrapper()
        lv2PycObj.doSetup(trainData=decidedTrainDf, sessionID=trySessionID)
        lv2PycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=5,
                             n_iter=100, early_stopping=False, customGridDict=None)

        _, lv2CvScoreRank = lv2PycObj.doCompareModel(fold=5, includeModelList=lv2PycObj.tunedModelList)
        lv2CvScoreCsvPath = mlScorePath + f'Lv2_cvScore_best_session{trySessionID}_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
        lv2CvScoreRank.to_csv(lv2CvScoreCsvPath)

        lv2PycObj.doFinalizeModel()  # train + self test 合併重新 fit
        lv2FinalModelList = lv2PycObj.finalModelList

        # 用這次嘗試 finalize 好的 model 對 DS_Indp（Boruta 篩選後的 meta-feature）做 predict，跟真實 y 算分
        lv2PredObjIndp = Predict(dataX=decidedIndp_X, modelList=lv2FinalModelList)
        lv2PredVectorListIndp, lv2ProbVectorListIndp = lv2PredObjIndp.doPredict()

        lv2ScoreObjIndp = Scoring(predVectorList=lv2PredVectorListIndp, probVectorList=lv2ProbVectorListIndp,
                                 answerDf=decidedIndp_y, modelNameList=modelNameList)
        lv2IndpScoreCsvPath = mlScorePath + f'Lv2_indpScore_best_session{trySessionID}_{dataName}_{normalizeMethod}{metaFeatureMatrixSuffix}.csv'
        lv2IndpScoreDf = lv2ScoreObjIndp.doScoring(b_optimizedMcc=False, path=lv2IndpScoreCsvPath, sortColumn='mcc')
        print(f"[{normalizeMethod}] tune_try {tryIdx + 1}/{tuneTry} 17 個 base learner 在 DS_Indp 上的分數已儲存到 {lv2IndpScoreCsvPath}")

        # doScoring 已經依 mcc 由大到小排序，第一列就是這次嘗試表現最好的 model
        tryBestModelName = lv2IndpScoreDf.index[0]
        tryBestRow = lv2IndpScoreDf.iloc[0].to_dict()
        tryBestRow['sessionID'] = trySessionID
        tryBestRow['modelName'] = tryBestModelName
        print(f"[{normalizeMethod}] tune_try {tryIdx + 1}/{tuneTry} 最佳 model：{tryBestModelName}，"
              f"indp mcc={tryBestRow['mcc']:.4f}")

        # 這個 try 自己最好的 model 立刻存檔(不等全部 tuneTry 跑完，也不等著跟其他 try 比較)
        bestModelIdx = lv2PycObj.modelNameList.index(tryBestModelName)
        tryFinalModelObj = lv2FinalModelList[bestModelIdx]
        PycaretWrapper.doSaveModelByName(model=tryFinalModelObj, modelName=f'session{trySessionID}_{tryBestModelName}',
                                         path=tryHistoryPath, b_isFinalizedModel=True)

        # 立刻把這一列併入總表、寫回 csv、重新判斷全域最佳並同步 Lv2_best，
        # 確保這個 try 一旦跑完，不管接下來是否被中斷，這一次的成果都已經完整記錄、不會遺失
        combinedSummaryDf = pd.concat([combinedSummaryDf, pd.DataFrame([tryBestRow])], ignore_index=True)
        combinedSummaryDf.to_csv(summaryCsvPath, index=False)
        promoteGlobalBest(combinedSummaryDf)

    print(f"[{normalizeMethod}] {tuneTry} 次 tune_try 全部完成，累積共 {len(combinedSummaryDf)} 筆結果，"
          f"總表：{summaryCsvPath}")
    print(combinedSummaryDf)
