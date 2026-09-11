"""
在裝有 torch(且支援 RTX 5090 / sm_120)的新環境下執行，
先把 ESM / T5 的 embedding 算好、寫進快取 csv。

跑完這支程式後，Main_FeatureStk_emb.py 在舊的 Python 3.8 環境執行時，
所有序列都能直接命中快取，完全不會 import torch，也就不會再撞到
「ModuleNotFoundError: No module named 'torch'」。

modelDirPath / dataName 必須跟 FeatureConfig.py 裡動態設定的值一致，
快取檔名才會對得上（f'{modelDirPath}_{method小寫}_cache.csv'）。
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat_llmEmbedding import LLMEmbeddingFeature

dataName = 'NeuroP_1'
paramPath = "../data/param/"

MainDatasetNegFastaPath = "../data/MainDatasetNeg.fasta"
MainDatasetPosFastaPath = "../data/MainDatasetPos.fasta"
DS_IndpNegFastaPath = "../data/DS_IndpNeg.fasta"
DS_IndpPosFastaPath = "../data/DS_IndpPos.fasta"

# 只需要覆蓋到「會被用到的序列」即可，DS_Train/DS_Val 是從 MainDataset 切出來的子集，
# 不會產生 MainDataset 以外的新序列，所以不需要重現 Main_FeatureStk_emb.py 裡的切分邏輯
ldObj = LoadDataset(minSeqLength=5)
allSeqLi = []
for fastaPath in [MainDatasetNegFastaPath, MainDatasetPosFastaPath,
                   DS_IndpNegFastaPath, DS_IndpPosFastaPath]:
    allSeqLi.extend(ldObj.readFasta(fastaPath).values())
allSeqDict = {i: seq for i, seq in enumerate(allSeqLi)}

blockSize = max(len(seq) for seq in allSeqDict.values())

llmEmbeddingDict = {
    "Usage": True,
    "modelDirPath": paramPath + f'{dataName}_llmEmbedding',  # 要跟 FeatureConfig.py 動態設定的值一致
    "blockSize": blockSize,
    "ESM": [True, "esm2_t33_650M_UR50D"],
    "T5": [True, "Rostlab/ProstT5"],
}

print(f"共 {len(allSeqDict)} 條序列（含重複）需要計算 embedding，blockSize={blockSize}")
LLMEmbeddingFeature(allSeqDict, llmEmbeddingDict)
print("ESM/T5 embedding 快取已計算完成，可以切回 Python 3.8 環境執行 Main_FeatureStk_emb.py 了")
