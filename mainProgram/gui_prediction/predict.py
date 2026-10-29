"""
在 pycaret 環境(conda env: neuropredictor-tool)執行這支程式。

前提：predict.csv 裡所有序列的 LLM embedding 都已經用 precompute_embedding.py
(在有 torch 的環境，例如 conda env: base)算好、存進快取 csv，這支程式只會讀快取，
完全不會 import torch，可以在沒有 torch 的舊環境裡執行。

流程：
  1. 讀 predict.csv(欄位: id, sequence；id 可省略)
  2. 用跟訓練時完全相同的 iFeature/pFeature/ampFeature/ovpFeature 子特徵 +
     7 個 LLM embedding 算出原始數值，並用訓練時存好的 standardScaler 手動做
     跟訓練一致的 z-score normalization(只對用得到的欄位做，不需要重算全部欄位)
  3. 對每個 feature type，載入對應的 Lv1 finalized base learner，產生 70 個
     meta-feature(跟 Lv2_best/standard/rf_final.pkl 訓練時用的 70 個欄位一致)
  4. 載入分數最高的 Lv2 RandomForest stacking model
     (data/finalModel_emb/Lv2_best/standard/rf_final.pkl，session 42，
     對 DS_Indp 的表現：accuracy=0.9471, auc=0.9868, mcc=0.8943)做最終預測
  5. 輸出每條序列的機率與陽性(神經肽)/陰性判定

用法:
  conda run -n neuropredictor-tool python predict.py [輸入csv路徑] [輸出csv路徑]
  預設輸入: mainProgram/gui_prediction/predict.csv
  預設輸出: mainProgram/gui_prediction/predict_result.csv
"""
import sys
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# 把使用者傳進來的相對路徑先轉成絕對路徑：import config 會把工作目錄切到 mainProgram/
# (配合 devPackage/PackagePFeature.py 寫中間檔用的寫死相對路徑)，之後再解析會解析錯地方。
_origCwd = os.getcwd()
_cliArgList = [os.path.abspath(os.path.join(_origCwd, arg)) for arg in sys.argv[1:]]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict


def encodeAndNormalize(recordList):
    """算出 70 個 meta-feature 需要的所有原始欄位，並用訓練時存好的 scaler 做 normalization。"""
    seqDict = {name: seq for name, seq in recordList}
    blockSize = max(len(seq) for seq in seqDict.values())
    dataDict = {0: None, 1: None, -1: seqDict}

    encodeObj = EncodeAllFeatures()
    encodeObj.featureDict = config.buildMinimalFeatureDict(blockSize=blockSize)
    rawDf = encodeObj.dataEncodeOutPut(dataDict=dataDict)
    rawDf = rawDf.drop(columns=['y'])

    with open(config.STANDARD_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    # StandardScaler 是用 fit_transform(ndarray) 存的，沒有欄位名稱，只能靠「訓練時 encode 大表格的欄位順序」
    # 對應每個欄位在 scaler.mean_ / scaler.scale_ 裡的位置；z-score 每欄各自獨立計算，
    # 不需要重建訓練時的完整欄位集合(25191欄)也能算出跟訓練一致的結果。
    fullRawColumnList = config.readCsvHeaderColumns(config.ENCODE_TRAIN_CSV_PATH)
    colPosMap = {col: i for i, col in enumerate(fullRawColumnList)}

    missingColumnList = [c for c in rawDf.columns if c not in colPosMap]
    if missingColumnList:
        raise RuntimeError(
            f"有 {len(missingColumnList)} 個欄位在訓練時的 encode 表格裡找不到，"
            f"無法對齊 scaler: {missingColumnList[:10]}")

    normalizedDf = pd.DataFrame(index=rawDf.index)
    for col in rawDf.columns:
        pos = colPosMap[col]
        normalizedDf[col] = ((rawDf[col].astype(float) - scaler.mean_[pos]) / scaler.scale_[pos]).round(5)
    return normalizedDf


def buildMetaFeatureMatrix(normalizedDf):
    """對每個 feature type 載入對應的 Lv1 base learner，組出 70 欄的 meta-feature 矩陣。"""
    metaFeatureColumnOrder = config.readCsvHeaderColumns(config.LV2_TRAIN_F70_PATH)

    typeToModelList = {}
    for col in metaFeatureColumnOrder:
        typeName, modelName = col.rsplit('.', 1)
        typeToModelList.setdefault(typeName, []).append(modelName)

    metaFeatureMatrix = pd.DataFrame(index=normalizedDf.index)
    for typeName, modelNameList in typeToModelList.items():
        typeCsvPath = os.path.join(config.FILTERED_TRAIN_DIR, f'{typeName}.csv')
        typeColumnList = config.readCsvHeaderColumns(typeCsvPath)
        Xtype = normalizedDf[typeColumnList]

        modelFolder = os.path.join(config.TUNE_MODEL_EMB_DIR, typeName.replace('.', '_'))
        modelList = PycaretWrapper().doLoadModel(modelFolder, fileNameList=modelNameList, b_isFinalizedModel=True)

        predObj = Predict(dataX=Xtype, modelList=modelList)
        _, probVectorList = predObj.doPredict()
        for modelName, probVector in zip(modelNameList, probVectorList):
            metaFeatureMatrix[f'{typeName}.{modelName}'] = probVector
        print(f"[Lv1] {typeName}: 已用 {modelNameList} 產生 meta-feature")

    return metaFeatureMatrix[metaFeatureColumnOrder]


def predictWithLv2(metaFeatureMatrix):
    modelList = PycaretWrapper().doLoadModel(
        config.LV2_BEST_MODEL_DIR, fileNameList=[config.LV2_MODEL_NAME], b_isFinalizedModel=True)
    predObj = Predict(dataX=metaFeatureMatrix, modelList=modelList)
    _, probVectorList = predObj.doPredict()
    return probVectorList[0]


def main():
    guiPredictionDir = os.path.join(config.PROJECT_ROOT, 'mainProgram', 'gui_prediction')
    inputCsvPath = _cliArgList[0] if len(_cliArgList) > 0 else os.path.join(guiPredictionDir, 'predict.csv')
    outputCsvPath = _cliArgList[1] if len(_cliArgList) > 1 else os.path.join(guiPredictionDir, 'predict_result.csv')

    recordList = config.loadPredictCsv(inputCsvPath)
    if not recordList:
        print(f"{inputCsvPath} 沒有讀到任何有效序列，結束。")
        return

    print(f"讀到 {len(recordList)} 條序列，開始計算特徵...")
    normalizedDf = encodeAndNormalize(recordList)

    print("計算 Lv1 base learner 的 meta-feature...")
    metaFeatureMatrix = buildMetaFeatureMatrix(normalizedDf)

    print(f"用 Lv2 stacking '{config.LV2_MODEL_NAME}' 模型做最終預測...")
    probVector = predictWithLv2(metaFeatureMatrix)

    seqMap = dict(recordList)
    resultDf = pd.DataFrame({
        'id': metaFeatureMatrix.index,
        'sequence': [seqMap[name] for name in metaFeatureMatrix.index],
        'probability': probVector,
    })
    resultDf['prediction'] = resultDf['probability'].apply(
        lambda p: 'Positive (neuropeptide)' if p >= config.PREDICTION_THRESHOLD else 'Negative')

    resultDf.to_csv(outputCsvPath, index=False)
    print(f"預測完成，結果已輸出到 {outputCsvPath}")
    print(resultDf.to_string(index=False))


if __name__ == '__main__':
    main()
