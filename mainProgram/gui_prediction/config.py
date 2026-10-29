"""
gui_prediction 共用設定與工具函式。

這個資料夾要分兩個 conda 環境各自執行一支程式（跟專案現有的
precompute_llmEmbedding.py / Main_FeatureStk_emb.py 兩階段流程同一個理由）：
  1. precompute_embedding.py 在有 torch/transformers 的環境執行（例如 conda env: base），
     把 predict.csv 裡新序列缺的 LLM embedding 算好，寫進跟訓練共用的快取 csv。
  2. predict.py 在 pycaret 環境（conda env: neuropredictor-tool，Python 3.8）執行，
     讀取快取 + 跑完整個 Lv1 base learner -> Lv2 RandomForest stacking，
     輸出每條序列的預測機率與陽性(神經肽)/陰性判定。

目前接的是分數最高的 Lv2 RandomForest stacking model：
data/finalModel_emb/Lv2_best/standard/rf_final.pkl
(session 42，對 DS_Indp 的表現：accuracy=0.9471, auc=0.9868, mcc=0.8943，是目前存檔裡
 分數最高的一次 RandomForest 版本，記錄於同資料夾 rf_final_info.json)。
"""
import os
import re
import sys
import csv
import copy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEVPACKAGE_DIR = os.path.join(PROJECT_ROOT, 'devPackage')
MAINPROGRAM_DIR = os.path.join(PROJECT_ROOT, 'mainProgram')

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# userPackage/FeatureStat_llmEmbedding.py 用 `from main_T5_ESM import EmbeddingsFeature`
# 載入 mainProgram/main_T5_ESM.py；這支程式放在 gui_prediction/子資料夾，執行時
# sys.path[0] 預設是 gui_prediction/ 而非 mainProgram/，要手動補進去才能 import 到。
if MAINPROGRAM_DIR not in sys.path:
    sys.path.append(MAINPROGRAM_DIR)
# devPackage/PackageiFeature.py 的 Geary/Moran/NMBroto 在非 Windows 平台用 sys.path[0]
# 組出 iFeaturedata/AAidx.txt 的路徑(其餘特徵都用 __file__ 算絕對路徑，只有這三個沒有)，
# 固定把 sys.path[0] 指到 devPackage，讓這三個特徵在 Linux/WSL 也能正常讀到資料檔。
sys.path[0] = DEVPACKAGE_DIR

# devPackage/PackagePFeature.py(DDOR/RRI/SER/SEP/SE/QSO)寫中間檔用的是寫死的相對路徑
# "../devPackage/intermediate files/..."，假設執行時的工作目錄是 mainProgram/(跟專案
# 既有 Main_*.py 的執行慣例一致)，這裡主動切過去，避免因為這支程式放在 gui_prediction/
# 子資料夾而解析到錯誤的路徑。
os.chdir(os.path.join(PROJECT_ROOT, 'mainProgram'))

DATA_NAME = 'NeuroP_1'
NORMALIZE_METHOD = 'standard'
DECIDED_FEATURE_NUM = 70

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PARAM_DIR = os.path.join(DATA_DIR, 'param')
FEATURE_STAT_DIR = os.path.join(DATA_DIR, 'featureStat')
TUNE_MODEL_EMB_DIR = os.path.join(DATA_DIR, 'tuneModel_test_emb', NORMALIZE_METHOD)
LV2_BEST_MODEL_DIR = os.path.join(DATA_DIR, 'finalModel_emb', 'Lv2_best', NORMALIZE_METHOD)
LV2_TRAIN_F70_PATH = os.path.join(
    DATA_DIR, 'mlData_emb', f'Lv2_{DATA_NAME}_{NORMALIZE_METHOD}_{DECIDED_FEATURE_NUM}',
    f'train_F{DECIDED_FEATURE_NUM}.csv')
ENCODE_TRAIN_CSV_PATH = os.path.join(FEATURE_STAT_DIR, f'encode_{DATA_NAME}_DS_Train.csv')
FILTERED_TRAIN_DIR = os.path.join(
    FEATURE_STAT_DIR, f'featureType_csv_{DATA_NAME}_{NORMALIZE_METHOD}_filtered', 'DS_Train')
STANDARD_SCALER_PATH = os.path.join(PARAM_DIR, f'{DATA_NAME}_{NORMALIZE_METHOD}Scaler.pkl')
LLM_EMBEDDING_CACHE_PREFIX = os.path.join(PARAM_DIR, f'{DATA_NAME}_llmEmbedding')

LV2_MODEL_NAME = 'rf'
PREDICTION_THRESHOLD = 0.5

MIN_SEQ_LENGTH = 5
VALID_AA_PATTERN = re.compile('[^ARNDCQEGHILKMFPSTWYV-]')

# 這 10 個是 Lv2_best/standard/rf_final.pkl 實際用到的 70 個 meta-feature裡(data/mlData_emb/
# Lv2_NeuroP_1_standard_70/train_F70.csv 的欄位名稱)，會用到的全部 LLM embedding 類型
# （key 須跟 FeatureConfig.py 的 llmEmbeddingFeatureDict 一致；原本這裡只列了7個，漏了
#  ESM_150M/ESM_35M/T5_XL_Bfd/ProteinGLM_7B_clm 這4個，導致 predict.py 算 meta-feature 時
#  在這些 feature type 上 KeyError）
REQUIRED_LLM_EMBEDDING_DICT = {
    'ESM_150M': 'esm2_t30_150M_UR50D',
    'ESM_35M': 'esm2_t12_35M_UR50D',
    'ESM_650M': 'esm2_t33_650M_UR50D',
    'ESM_3B': 'esm2_t36_3B_UR50D',
    'T5_ProstT5': 'Rostlab/ProstT5',
    'T5_XL_UniRef50': 'Rostlab/prot_t5_xl_uniref50',
    'T5_XL_Bfd': 'Rostlab/prot_t5_xl_bfd',
    'Bert_ProtBert': 'Rostlab/prot_bert',
    'ProteinGLM_10B_mlm': 'biomap-research/proteinglm-10b-mlm',
    'ProteinGLM_7B_clm': 'proteinglm/proteinglm-7b-clm',
}

# 70 個 meta-feature裡，其餘 classical feature type 用到的子特徵開關，比照 train_F70.csv 實際欄位核對
# （iFeature.GAAC / ampFeature.formula / ovpFeature.OVPC 是 mergedFeature.OVPC_GAAC_formula 的原始組成，
#  雖然本身不是 70 個欄位之一，但沒開的話 mergedFeature.OVPC_GAAC_formula 就算不出來）
REQUIRED_IFEATURE_KEYS = ['APAAC', 'CKSAAGP', 'CKSAAP', 'CTDC', 'CTDD', 'CTDT',
                          'DDE', 'DPC', 'GDPC', 'GTPC', 'Moran', 'GAAC']
REQUIRED_PFEATURE_KEYS = ['DDOR', 'QSO', 'RRI', 'SEP']
REQUIRED_AMPFEATURE_KEYS = ['cougar', 'ez', 'z3', 'z5', 'formula']
REQUIRED_OVPFEATURE_KEYS = ['OVPC']


def buildLlmEmbeddingDict(blockSize=None):
    """組出 LLMEmbeddingFeature 需要的 dict：只開 REQUIRED_LLM_EMBEDDING_DICT 這 7 個方法。"""
    llmDict = {
        'Usage': True,
        'modelDirPath': LLM_EMBEDDING_CACHE_PREFIX,
        'blockSize': blockSize,
    }
    for key, modelName in REQUIRED_LLM_EMBEDDING_DICT.items():
        llmDict[key] = [True, modelName]
    return llmDict


def buildMinimalFeatureDict(blockSize=None):
    """
    以 FeatureConfig.featureDict 為底，只保留 70 個 Lv2 meta-feature 實際用得到的
    classical feature子特徵 + 7 個 LLM embedding，其餘全部關閉，避免算一堆用不到的特徵。
    """
    from userPackage.FeatureConfig import featureDict as defaultFeatureDict
    featureDict = copy.deepcopy(defaultFeatureDict)

    def turnAllOff(subDict):
        for key, value in subDict.items():
            if isinstance(value, list):
                value[0] = False
            else:
                subDict[key] = False

    def turnOn(subDict, keyList):
        for key in keyList:
            if isinstance(subDict[key], list):
                subDict[key][0] = True
            else:
                subDict[key] = True

    for groupName in ['iFeature', 'pFeature', 'ampFeature', 'ovpFeature']:
        turnAllOff(featureDict[groupName])
    turnOn(featureDict['iFeature'], REQUIRED_IFEATURE_KEYS)
    turnOn(featureDict['pFeature'], REQUIRED_PFEATURE_KEYS)
    turnOn(featureDict['ampFeature'], REQUIRED_AMPFEATURE_KEYS)
    turnOn(featureDict['ovpFeature'], REQUIRED_OVPFEATURE_KEYS)
    featureDict['centerGDPFeature']['Usage'] = False
    featureDict['llmEmbeddingFeature'] = buildLlmEmbeddingDict(blockSize=blockSize)
    return featureDict


def loadPredictCsv(csvPath):
    """
    讀取使用者輸入的 predict.csv，欄位需含 sequence（大小寫不拘），id 欄位可省略。
    回傳 [(id, sequence), ...]，序列統一轉大寫、非標準胺基酸字元轉成 '-'
    （跟 userPackage/LoadDataset.py 的 readFasta 處理方式一致）。
    長度小於 MIN_SEQ_LENGTH 的序列會被跳過並印出警告。
    """
    import pandas as pd
    df = pd.read_csv(csvPath, dtype=str)
    columnMap = {c.lower(): c for c in df.columns}
    if 'sequence' not in columnMap:
        raise ValueError(f"{csvPath} 缺少 'sequence' 欄位，目前欄位: {list(df.columns)}")
    seqCol = columnMap['sequence']
    idCol = columnMap.get('id')

    recordList = []
    usedNameSet = set()
    for i, row in df.iterrows():
        rawSeq = row[seqCol]
        if not isinstance(rawSeq, str) or rawSeq.strip() == '':
            continue
        seq = VALID_AA_PATTERN.sub('-', rawSeq.strip().upper())
        name = str(row[idCol]).strip() if idCol is not None and pd.notna(row[idCol]) else f'seq{i + 1}'
        if name in usedNameSet:
            name = f'{name}_{i + 1}'
        if len(seq) < MIN_SEQ_LENGTH:
            print(f"[跳過] {name}: 序列長度 {len(seq)} 小於最小長度 {MIN_SEQ_LENGTH}")
            continue
        usedNameSet.add(name)
        recordList.append((name, seq))
    return recordList


def readCsvHeaderColumns(csvPath):
    """只讀第一行表頭，避免大檔案整個載入記憶體；回傳欄位名稱 list（去掉最前面的 index 欄與最後的 y 欄）。"""
    with open(csvPath, newline='', encoding='utf-8') as f:
        header = next(csv.reader(f))
    assert header[-1] == 'y', f"{csvPath} 表頭最後一欄預期是 'y'，實際是 '{header[-1]}'"
    return header[1:-1]
