"""
在有 torch/transformers 的環境(例如 conda env: base)執行這支程式，
把 predict.csv 裡每條序列缺少的 LLM embedding(ESM/T5/Bert/ProteinGLM)算好，
寫進跟訓練共用的快取 csv(data/param/NeuroP_1_llmEmbedding_*_cache.csv)，
以序列內容本身當 key，所以算過一次之後不論換到哪個 predict.csv 都能命中快取。

跑完這支程式後，才能切到 pycaret 環境(conda env: neuropredictor-tool)執行 predict.py；
predict.py 只會讀快取，完全不會 import torch。

用法:
  conda run -n base python precompute_embedding.py [predict.csv路徑，預設同資料夾 predict.csv]

注意：ProteinGLM_10B_mlm / ESM_3B 都是數十億參數的大型模型，第一次執行需要下載模型權重
(數十 GB)並用 GPU 推論，請確保網路與硬碟空間足夠、且目前 GPU 沒有被其他程式佔滿。
"""
import sys
import os

# 把使用者傳進來的相對路徑先轉成絕對路徑：import config 會把工作目錄切到 mainProgram/
# (配合 devPackage/PackagePFeature.py 寫中間檔用的寫死相對路徑)，之後再解析會解析錯地方。
_origCwd = os.getcwd()
_cliArgList = [os.path.abspath(os.path.join(_origCwd, arg)) for arg in sys.argv[1:]]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from userPackage.FeatureStat_llmEmbedding import LLMEmbeddingFeature


def main():
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    csvPath = _cliArgList[0] if len(_cliArgList) > 0 else os.path.join(scriptDir, 'predict.csv')

    recordList = config.loadPredictCsv(csvPath)
    if not recordList:
        print(f"{csvPath} 沒有讀到任何有效序列，結束。")
        return

    seqDict = {name: seq for name, seq in recordList}
    blockSize = max(len(seq) for seq in seqDict.values())
    print(f"共 {len(seqDict)} 條序列需要確認/計算 embedding，blockSize={blockSize}")
    print(f"需要的 embedding 類型: {list(config.REQUIRED_LLM_EMBEDDING_DICT.keys())}")

    llmEmbeddingDict = config.buildLlmEmbeddingDict(blockSize=blockSize)
    LLMEmbeddingFeature(seqDict, llmEmbeddingDict)

    print("embedding 快取已計算/確認完成，可以切到 pycaret 環境"
          "(conda env: neuropredictor-tool)執行 predict.py 了")


if __name__ == '__main__':
    main()
