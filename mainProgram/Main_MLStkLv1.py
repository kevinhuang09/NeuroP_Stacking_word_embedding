# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
tune_model = True  # True: 針對每個 normalizeMethod × feature type × model 做一對一 tune 並存檔；False: 跳過訓練，直接讀取已存的 finalized model 做 predict
predictOnTrainToo = False  # True: 除了對 DS_Val predict 外，也對 DS_Train predict，兩者 concat 成另一份 Meta-Feature-Matrix
                           # 注意：DS_Train 是 finalize 時模型看過的資料（in-sample），機率會偏樂觀，跟 DS_Val 的
                           # out-of-sample 機率意義不同，只是多存一份組法給 Lv2 比較用，不會取代原本 DS_Val-only 的檔案
# ======================================================================================================================
# 測試用開關：跟 Main_MLStkLv1_debug.py 功能完全相同，只差在把 feature type 與 model 數量都縮小方便快速測試
testFeatureTypeCount = 99  # 只取 5 個 feature type（兩個 mergedFeature 合併項目一定會被包含在內）
testModelCount = 99  # 只取 5 個 model（lightgbm、xgboost 一定會被包含在內）
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
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring

# ======================================================================================================================
# 基本路徑：直接接 Main_FeatureStk_debug.py 產生的結果，不重新推導 feature type/欄位
# Main_FeatureStk_debug.py 對每個 normalizeMethod 都會把 feature filter 完的資料依 feature type 拆成
# featureStatPath/featureType_csv_{dataName}_{normalizeMethod}_filtered/DS_Train(或 DS_Val)/{typeName}.csv，
# 檔名就是 feature type 名稱（mergedFeature 底下的合併項目已經是簡短名稱，如 mergedFeature.SingleValueCombine），
# 這裡直接列出資料夾裡的檔案逐一讀取、訓練，不用重新讀 featureTypeDict.json 或重算欄位對照表。
featureStatPath = '../data/featureStat/'
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案（測試結果會加上 _test 後綴，跟正式結果分開）
tuneModelPath = "../data/tuneModel_test/"  # 測試專用資料夾，避免覆蓋 Main_MLStkLv1_debug.py 存的正式 finalized model
dataName = 'NeuroP_1'

# Main_FeatureStk_debug.py 目前 normalizeMethodList 只有 'standard'，這裡要跟它保持一致，
# 否則會去讀一個 Main_FeatureStk_debug.py 根本沒產生的 featureType_csv_{dataName}_{normalizeMethod}_filtered 資料夾
normalizeMethodList = ['robust']

modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

# 測試用：只取 testModelCount 個 model，lightgbm 跟 xgboost 一定要包含在內
requiredModelNameList = ['lightgbm', 'xgboost']
otherModelNameList = [modelName for modelName in modelNameList if modelName not in requiredModelNameList]
modelNameList = (requiredModelNameList + otherModelNameList)[:testModelCount]
print(f"[test] 只使用 {len(modelNameList)} 個 model 進行測試: {modelNameList}")

os.makedirs(mlScorePath, exist_ok=True)
resultRows = []
valScoreRows = []  # 每個 combo 用 finalize/load 好的 model 對 DS_Val 做 predict 後，跟真實 y 算出來的分數

# ======================================================================================================================
# 每一個 normalize 方式 × 每一個 feature type × 每一個 model 做一對一訓練（Optuna TPE tune，5-fold CV，用 MCC 當優化目標）
for normalizeMethod in normalizeMethodList:
    filteredSplitDir = featureStatPath + f'featureType_csv_{dataName}_{normalizeMethod}_filtered/'
    trainSplitDir = filteredSplitDir + 'DS_Train/'
    valSplitDir = filteredSplitDir + 'DS_Val/'
    indpSplitDir = filteredSplitDir + 'DS_Indp/'

    # 檔名（去掉 .csv）就是 feature type 名稱，train / val / indp 三邊都要有才拿來訓練＋predict
    trainTypeFileSet = {f for f in os.listdir(trainSplitDir) if f.endswith('.csv')}
    valTypeFileSet = {f for f in os.listdir(valSplitDir) if f.endswith('.csv')}
    indpTypeFileSet = {f for f in os.listdir(indpSplitDir) if f.endswith('.csv')}
    typeFileNameList = sorted(trainTypeFileSet & valTypeFileSet & indpTypeFileSet)
    missingSet = (trainTypeFileSet | valTypeFileSet | indpTypeFileSet) - (trainTypeFileSet & valTypeFileSet & indpTypeFileSet)
    if missingSet:
        print(f"[提醒] {normalizeMethod}：{len(missingSet)} 個 feature type 沒有同時存在於 DS_Train/DS_Val/DS_Indp，已跳過: {sorted(missingSet)}")
    print(f"[{normalizeMethod}] 讀到 {len(typeFileNameList)} 個 feature type 的過濾後資料：{trainSplitDir}")

    # 測試用：只取 testFeatureTypeCount 個 feature type，兩個 mergedFeature 合併項目一定要包含在內
    mergedTypeFileNameList = [f for f in typeFileNameList if f.startswith('mergedFeature')]
    otherTypeFileNameList = [f for f in typeFileNameList if not f.startswith('mergedFeature')]
    typeFileNameList = (mergedTypeFileNameList + otherTypeFileNameList)[:testFeatureTypeCount]
    print(f"[test][{normalizeMethod}] 只使用 {len(typeFileNameList)} 個 feature type 進行測試: {typeFileNameList}")

    # 用任一份 DS_Val / DS_Indp 檔案先確定 index/y，組出 Meta-Feature-Matrix 骨架
    firstValDf = pd.read_csv(valSplitDir + typeFileNameList[0], index_col=[0])
    metaFeatureMatrix = pd.DataFrame(index=firstValDf.index)
    metaFeatureMatrix['y'] = firstValDf['y']

    firstIndpDf = pd.read_csv(indpSplitDir + typeFileNameList[0], index_col=[0])
    indpMetaFeatureMatrix = pd.DataFrame(index=firstIndpDf.index)
    indpMetaFeatureMatrix['y'] = firstIndpDf['y']

    if predictOnTrainToo:
        firstTrainDf = pd.read_csv(trainSplitDir + typeFileNameList[0], index_col=[0])
        trainMetaFeatureMatrix = pd.DataFrame(index=firstTrainDf.index)
        trainMetaFeatureMatrix['y'] = firstTrainDf['y']

    for typeFileName in typeFileNameList:  # 一次挑一種 Feature Type
        typeName = typeFileName[:-len('.csv')]

        subTrainDf = pd.read_csv(trainSplitDir + typeFileName, index_col=[0])
        subValDf = pd.read_csv(valSplitDir + typeFileName, index_col=[0])
        subVal_X = subValDf.drop(columns=['y'])  # DS_Val 同一組欄位，只留 X 給 predict 用
        subIndpDf = pd.read_csv(indpSplitDir + typeFileName, index_col=[0])
        subIndp_X = subIndpDf.drop(columns=['y'])  # DS_Indp 同一組欄位，只留 X 給 predict 用
        print(f"[{normalizeMethod}] {typeName}: train={subTrainDf.shape}, val_X={subVal_X.shape}, indp_X={subIndp_X.shape}")

        # 每個 normalizeMethod + featureType 的 model 各自存在獨立資料夾，避免互相覆蓋
        comboSavePath = os.path.join(tuneModelPath, normalizeMethod, typeName.replace('.', '_'))
        os.makedirs(comboSavePath, exist_ok=True)

        pycObj = PycaretWrapper()
        if tune_model:
            pycObj.doSetup(trainData=subTrainDf, sessionID=42)

        for modelName in modelNameList:  # 每個 model 逐一嘗試（feature type × model 一對一訓練）
            try:
                if tune_model:
                    # 進行一對一訓練
                    pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe',
                                       includeModelList=[modelName], foldNum=5,
                                       n_iter=10, early_stopping=False, customGridDict=None)
                    _, scoreRank = pycObj.doCompareModel(fold=5, includeModelList=pycObj.tunedModelList)  # 對 tune 好的 model 重新做一次 CV 拿分數表
                    mccScore = scoreRank['MCC'].iloc[0]
                    print(f"[完成] {normalizeMethod} + {typeName} + {modelName}: MCC = {mccScore}")
                    resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                       'model': modelName, 'mcc': mccScore})

                    pycObj.doFinalizeModel()  # train + self test 合併重新 fit，存起來給之後直接讀取用
                    pycObj.doSaveModel(comboSavePath, b_isFinalizedModel=True)  # 每個 model 各自存檔
                    predictModelList = pycObj.finalModelList
                else:
                    # tune_model=False：跳過訓練，直接讀取已存的 finalized model
                    predictModelList = pycObj.doLoadModel(comboSavePath, fileNameList=[modelName],
                                                           b_isFinalizedModel=True)

                # 用這個 model 對 DS_Val / DS_Indp 做 predict，分別寫進各自的 Meta-Feature-Matrix（機率表）
                predObj = Predict(dataX=subVal_X, modelList=predictModelList)
                predVectorList, probVectorList = predObj.doPredict()
                metaFeatureMatrix[f'{typeName}.{modelName}'] = probVectorList[0]

                if predictOnTrainToo:
                    # DS_Train 是 finalize 時模型看過的資料，predict 出來的機率屬於 in-sample，僅供跟 DS_Val 的
                    # out-of-sample 機率一起 concat 成另一種組法比較用
                    trainPredObj = Predict(dataX=subTrainDf.drop(columns=['y']), modelList=predictModelList)
                    _, trainProbVectorList = trainPredObj.doPredict()
                    trainMetaFeatureMatrix[f'{typeName}.{modelName}'] = trainProbVectorList[0]

                indpPredObj = Predict(dataX=subIndp_X, modelList=predictModelList)
                _, indpProbVectorList = indpPredObj.doPredict()
                indpMetaFeatureMatrix[f'{typeName}.{modelName}'] = indpProbVectorList[0]
                print(f"[predict] {normalizeMethod} + {typeName} + {modelName} 已寫入 Meta-Feature-Matrix（val + indp）")

                # DS_Val 是這個 model 訓練/調參時完全沒看過的資料，拿真實 y 對 predict 結果算分，
                # 比前面 doCompareModel 那個「DS_Train 內部 CV」的分數更能反映真正的泛化表現
                valScoreObj = Scoring(predVectorList=predVectorList, probVectorList=probVectorList,
                                     answerDf=subValDf[['y']], modelNameList=[modelName])
                valScoreDf = valScoreObj.doScoring(b_optimizedMcc=False, sortColumn=None)
                valScoreRow = {'normalizeMethod': normalizeMethod, 'featureType': typeName, 'model': modelName}
                valScoreRow.update(valScoreDf.iloc[0].to_dict())
                valScoreRows.append(valScoreRow)
                print(f"[Val分數] {normalizeMethod} + {typeName} + {modelName}: "
                      f"MCC={valScoreDf['mcc'].iloc[0]:.4f}, AUC={valScoreDf['auc'].iloc[0]:.4f}")
            except Exception as e:
                actionName = '訓練' if tune_model else '讀取/predict'
                print(f"[跳過] {normalizeMethod} + {typeName} + {modelName} {actionName}失敗: {e}")
                if tune_model:
                    resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                       'model': modelName, 'mcc': None, 'error': str(e)})
                valScoreRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                     'model': modelName, 'mcc': None, 'error': str(e)})

    metaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv'
    metaFeatureMatrix.to_csv(metaFeatureMatrixPath)
    print(f"[{normalizeMethod}] DS_Val 機率表 (Meta-Feature-Matrix) 已儲存到 {metaFeatureMatrixPath}")

    if predictOnTrainToo:
        # DS_Train（in-sample）+ DS_Val（out-of-sample）的機率表 row-wise concat，多存一份給 Lv2 比較用，
        # 不會覆蓋掉上面 DS_Val-only 的版本
        trainValMetaFeatureMatrix = pd.concat([trainMetaFeatureMatrix, metaFeatureMatrix])
        trainValMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}_trainval.csv'
        trainValMetaFeatureMatrix.to_csv(trainValMetaFeatureMatrixPath)
        print(f"[{normalizeMethod}] DS_Train({trainMetaFeatureMatrix.shape}) + DS_Val({metaFeatureMatrix.shape}) "
              f"機率表 (Meta-Feature-Matrix) 已儲存到 {trainValMetaFeatureMatrixPath}：{trainValMetaFeatureMatrix.shape}")

    indpMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv'
    indpMetaFeatureMatrix.to_csv(indpMetaFeatureMatrixPath)
    print(f"[{normalizeMethod}] DS_Indp 機率表 (Meta-Feature-Matrix) 已儲存到 {indpMetaFeatureMatrixPath}")

resultValScoreSuffix = '_trainval' if predictOnTrainToo else ''

if tune_model:
    resultDf = pd.DataFrame(resultRows)
    resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}_test{resultValScoreSuffix}.csv'
    resultDf.to_csv(resultCsvPath, index=False)
    print(f"每個 normalizeMethod × feature type × model 的 MCC 分數已儲存到 {resultCsvPath}")

valScoreDf = pd.DataFrame(valScoreRows)
valScoreCsvPath = mlScorePath + f'featureType_model_valScore_{dataName}_test{resultValScoreSuffix}.csv'
valScoreDf.to_csv(valScoreCsvPath, index=False)
print(f"每個 normalizeMethod × feature type × model 在 DS_Val 上的分數（accuracy/precision/recall/f1_score/auc/specificity/mcc）"
      f"已儲存到 {valScoreCsvPath}")
