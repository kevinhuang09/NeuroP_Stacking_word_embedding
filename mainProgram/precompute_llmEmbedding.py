"""
在裝有 torch(且支援 RTX 5090 / sm_120)的新環境下執行，
先把 ESM / T5 / Ankh 的 embedding 算好、寫進快取 csv。

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
    # key 須跟 FeatureConfig.py 的 llmEmbeddingFeatureDict 保持一致，快取檔名才會對得上；
    # 這裡全部開啟（True）先把要用到的模型都算好快取，之後在 FeatureConfig.py 裡開關哪些模型都不用重算
    "ESM_650M": [True, "esm2_t33_650M_UR50D"],
    "ESM_3B": [True, "esm2_t36_3B_UR50D"],
    "ESM_8M": [False, "esm2_t6_8M_UR50D"],
    "ESM_35M": [False, "esm2_t12_35M_UR50D"],
    "ESM_150M": [False, "esm2_t30_150M_UR50D"],
    "ESM_15B": [False, "esm2_t48_15B_UR50D"],  # 55GB+ 權重檔+需30GB+ VRAM，目前網路環境下載不穩，先關閉
    "ESM_1b": [False, "esm1b_t33_650M_UR50S"],
    "ESM_1v_1": [False, "esm1v_t33_650M_UR90S_1"],
    "ESM_1v_2": [False, "esm1v_t33_650M_UR90S_2"],
    "ESM_1v_3": [False, "esm1v_t33_650M_UR90S_3"],
    "ESM_1v_4": [False, "esm1v_t33_650M_UR90S_4"],
    "ESM_1v_5": [False, "esm1v_t33_650M_UR90S_5"],
    "T5_ProstT5": [True, "Rostlab/ProstT5"],
    "T5_XL_UniRef50": [True, "Rostlab/prot_t5_xl_uniref50"],
    "T5_XL_BFD": [False, "Rostlab/prot_t5_xl_bfd"],
    "T5_XXL_UniRef50": [False, "Rostlab/prot_t5_xxl_uniref50"],
    "T5_XXL_BFD": [False, "Rostlab/prot_t5_xxl_bfd"],
    "T5_Base_MT_UniRef50": [False, "Rostlab/prot_t5_base_mt_uniref50"],
    "Ankh_Base": [True, "ElnaggarLab/ankh-base"],
    "Ankh_Large": [True, "ElnaggarLab/ankh-large"],
    "Ankh2_Large": [False, "ElnaggarLab/ankh2-large"],
    "Ankh2_Ext1": [False, "ElnaggarLab/ankh2-ext1"],
    "Ankh2_Ext2": [False, "ElnaggarLab/ankh2-ext2"],
    "Ankh3_Large": [False, "ElnaggarLab/ankh3-large"],
    "Ankh3_XL": [False, "ElnaggarLab/ankh3-xl"],
    "Bert_ProtBert": [False, "Rostlab/prot_bert"],
    "Bert_ProtBert_BFD": [False, "Rostlab/prot_bert_bfd"],
    "ProteinGLM_100B_int4": [False, "biomap-research/proteinglm-100b-int4"],
    "ProteinGLM_10B_mlm": [True, "biomap-research/proteinglm-10b-mlm"]
}

print(f"共 {len(allSeqDict)} 條序列（含重複）需要計算 embedding，blockSize={blockSize}")
LLMEmbeddingFeature(allSeqDict, llmEmbeddingDict)
print("ESM/T5/Ankh embedding 快取已計算完成，可以切回 Python 3.8 環境執行 Main_FeatureStk_emb.py 了")
