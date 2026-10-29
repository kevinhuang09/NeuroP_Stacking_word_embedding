"""
Evaluate a Lv2 finalized model on one of DS_Train / DS_Val / DS_Indp, using the
per-feature-type normalized CSVs under
data/featureStat/featureType_csv_NeuroP_1_standard_filtered/<dataset>
(the same source the Lv1 base learners were trained on) to rebuild the 70
meta-features, then score against that dataset's labels.

Run in the pycaret env:
  conda run -n neuropredictor-tool python eval_lv2_model.py DS_Val
  conda run -n neuropredictor-tool python eval_lv2_model.py DS_Indp
  conda run -n neuropredictor-tool python eval_lv2_model.py DS_Train
"""
import os
import sys

import pandas as pd

GUI_PRED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GUI_PRED_DIR)
import config  # reuses TUNE_MODEL_EMB_DIR / LV2_BEST_MODEL_DIR / LV2_TRAIN_F70_PATH / readCsvHeaderColumns

from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict
from MLProcess.Scoring import Scoring


def buildMetaFeatureMatrix(datasetName):
    filteredDir = os.path.join(
        config.DATA_DIR, 'featureStat', 'featureType_csv_NeuroP_1_standard_filtered', datasetName)
    metaFeatureColumnOrder = config.readCsvHeaderColumns(config.LV2_TRAIN_F70_PATH)

    typeToModelList = {}
    for col in metaFeatureColumnOrder:
        typeName, modelName = col.rsplit('.', 1)
        typeToModelList.setdefault(typeName, []).append(modelName)

    yLabels = None
    metaFeatureMatrix = None
    for typeName, modelNameList in typeToModelList.items():
        typeDf = pd.read_csv(os.path.join(filteredDir, f'{typeName}.csv'), index_col=0)
        y = typeDf['y']
        X = typeDf.drop(columns=['y'])
        if yLabels is None:
            yLabels = y
            metaFeatureMatrix = pd.DataFrame(index=X.index)

        modelFolder = os.path.join(config.TUNE_MODEL_EMB_DIR, typeName.replace('.', '_'))
        modelList = PycaretWrapper().doLoadModel(modelFolder, fileNameList=modelNameList, b_isFinalizedModel=True)
        _, probVectorList = Predict(dataX=X, modelList=modelList).doPredict()
        for modelName, probVector in zip(modelNameList, probVectorList):
            metaFeatureMatrix[f'{typeName}.{modelName}'] = probVector

    return metaFeatureMatrix[metaFeatureColumnOrder], yLabels


def main():
    datasetName = sys.argv[1] if len(sys.argv) > 1 else 'DS_Indp'
    lv2ModelName = sys.argv[2] if len(sys.argv) > 2 else config.LV2_MODEL_NAME

    metaFeatureMatrix, yLabels = buildMetaFeatureMatrix(datasetName)

    lv2ModelList = PycaretWrapper().doLoadModel(
        config.LV2_BEST_MODEL_DIR, fileNameList=[lv2ModelName], b_isFinalizedModel=True)
    predVectorList, probVectorList = Predict(dataX=metaFeatureMatrix, modelList=lv2ModelList).doPredict()

    scoreDf = Scoring(predVectorList, probVectorList, yLabels, modelNameList=[lv2ModelName]).doScoring()
    print(f'\n{datasetName} / {lv2ModelName}:')
    print(scoreDf.to_string())


if __name__ == '__main__':
    main()
