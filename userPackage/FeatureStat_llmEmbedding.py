import os
import gc
import torch
import numpy as np
import pandas as pd


class LLMEmbeddingFeature:
    """
    把每條胺基酸序列丟入預訓練好的蛋白質語言模型(ESM / T5)，取得該序列的 embedding 向量，
    依 llmEmbeddingDict 裡 ESM、T5 各自的開關，各自算出一組固定維度的向量當作該序列的特徵；
    多個方法同時開啟時，各自的欄位會水平合併成一份輸出 DataFrame，可當成一種新的 feature type 併入
    既有的 featureDict 系統(用法比照 WordEmbeddingFeature 與 devPackage 底下各個
    XXX(seqDict, featureDict).getOutputDf() 的介面)。

    跟 WordEmbeddingFeature 最大的不同：ESM/T5 是「預訓練好、參數凍結」的模型，不需要(也不能)用
    使用者自己的資料集重新訓練，所以沒有 trainAndSaveModel() 這種「先在 DS_Train 上訓練一次」的概念，
    也不會有資料外洩(data leakage)的疑慮 —— 同一條序列不論放進 DS_Train/DS_Indp/DS_Val 哪一個資料集，
    算出來的 embedding 永遠一樣。

    真正需要快取的是「算好的 embedding 結果」，因為每算一次都要真的跑一次 GPU 推論，成本很高。
    因此這裡改成用 csv 快取每條「序列內容」對應的 embedding 向量(以序列字串本身當 key，而非序列名稱)，
    這樣 DS_Train/DS_Indp/DS_Val 之間、甚至不同 dataName 之間，只要出現過同一條序列(即使改了名字或
    出現在不同資料集)，都能命中快取不用重算；快取檔案會持續累積、越用越快。

    llmEmbeddingDict 的格式：
        "Usage"        : 整個 llmEmbeddingFeature type 的總開關；False 時不論底下各方法開關為何都不會執行
        "ESM"          : [開關, esmModel名稱]，例如
                          "esm2_t33_650M_UR50D"(輸出1280維) 或 "esm2_t36_3B_UR50D"(輸出2560維)
        "T5"           : [開關, t5Model名稱]，例如
                          "Rostlab/ProstT5"(輸出1024維) 或 "Rostlab/prot_t5_xl_uniref50"(輸出1024維)
        "modelDirPath" : embedding 快取 csv 存放路徑前綴(字串)，組出
                          f'{modelDirPath}_{methodName小寫}_cache.csv'，須在 Main 程式依 dataName 動態設定；
                          若為 None，則每次都重新用模型計算、不存檔快取(不建議，序列一多會很慢)。
        "blockSize"    : (可選) int，ESM/T5 推論時 padding/truncation 用的固定序列長度上限。
                          建議由 Main 程式先算出「全部資料集(Train+Indp+Val)中最長序列的長度」統一設定，
                          確保三個資料集用同一套 padding 標準去推論；若省略，會退化成「只用當次傳入的
                          seqDict 自己算最大長度」，多個資料集之間可能長度不一致，不建議在正式流程中依賴這個退化行為。
    "Usage"、"modelDirPath"、"blockSize" 都不是方法名稱，不會被當成 LLM embedding 方法處理。
    """

    _ESM_MODEL_DIM = {
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
    }
    _T5_MODEL_DIM_DEFAULT = 1024
    _NON_METHOD_KEY_TUPLE = ("modelDirPath", "Usage", "blockSize")

    def __init__(self, seqDict, llmEmbeddingDict):
        self.seqsNameLi = list(seqDict.keys())
        self.modelDirPath = llmEmbeddingDict.get("modelDirPath")
        self.blockSize = llmEmbeddingDict.get("blockSize")
        self.methodItemLi = [(name, paramLi) for name, paramLi in llmEmbeddingDict.items()
                             if name not in self._NON_METHOD_KEY_TUPLE]
        # 總開關 Usage 為 False 時，不論底下各方法開關為何都不執行
        self.b_start = llmEmbeddingDict.get("Usage", False) is True and any(
            paramLi[0] for _, paramLi in self.methodItemLi)

        if self.b_start is True:
            # blockSize 沒有外部統一指定時，退化成只用當次 seqDict 自己算最大長度(見上方 docstring 提醒)
            effectiveBlockSize = self.blockSize if self.blockSize else max(
                (len(seq) for seq in seqDict.values()), default=1)

            featureDfLi = []
            for methodName, paramLi in self.methodItemLi:
                if paramLi[0] is not True:
                    continue
                methodCachePath = self._getMethodCachePath(methodName)
                featureDfLi.append(
                    self._runMethod(methodName, paramLi, seqDict, effectiveBlockSize, methodCachePath))
            self.featureDf = pd.concat(featureDfLi, axis=1)
        else:
            pass

    def getOutputDf(self):
        if self.b_start is True:
            self.featureDf.index = self.seqsNameLi
            return self.featureDf
        else:
            pass

    def _getMethodCachePath(self, methodName):
        if self.modelDirPath is None:
            return None
        return f'{self.modelDirPath}_{methodName.lower()}_cache.csv'

    # ------------------------------------------------------------------
    # 快取讀寫：以「序列字串」為 index，而非序列名稱，讓不同資料集/不同名稱的相同序列都能命中快取
    # ------------------------------------------------------------------
    @classmethod
    def _loadCache(cls, cachePath):
        if cachePath is not None and os.path.exists(cachePath):
            cacheDf = pd.read_csv(cachePath, index_col=0)
            cacheDf.columns = [int(c) for c in cacheDf.columns]  # 讀回時欄名會變字串，轉回 int 方便後續處理
            return cacheDf
        return None

    @classmethod
    def _saveCache(cls, cacheDf, cachePath):
        if cachePath is not None:
            cacheDir = os.path.dirname(cachePath)
            if cacheDir:
                os.makedirs(cacheDir, exist_ok=True)
            cacheDf.to_csv(cachePath)

    @classmethod
    def _runMethod(cls, methodName, paramLi, seqDict, blockSize, cachePath):
        cacheDf = cls._loadCache(cachePath)

        # 同一條序列不論出現幾次(甚至改名重複出現)，只需要算一次
        uniqueSeqLi = list(dict.fromkeys(seqDict.values()))
        missingSeqLi = [seq for seq in uniqueSeqLi if cacheDf is None or seq not in cacheDf.index]

        if len(missingSeqLi) > 0:
            newDf = cls._computeEmbedding(methodName, paramLi, missingSeqLi, blockSize)
            cacheDf = newDf if cacheDf is None else pd.concat([cacheDf, newDf], axis=0)
            cacheDf = cacheDf[~cacheDf.index.duplicated(keep='last')]
            cls._saveCache(cacheDf, cachePath)

        # 依 seqDict 目前的順序，從快取表裡把每條序列的向量取出來組成輸出 DataFrame
        columnPrefix = methodName.lower()
        vectorSize = cacheDf.shape[1]
        rowLi = [cacheDf.loc[seq].values for seq in seqDict.values()]
        columnNameLi = [f'{columnPrefix}_{i}' for i in range(vectorSize)]
        return pd.DataFrame(rowLi, columns=columnNameLi)

    # ------------------------------------------------------------------
    # 實際呼叫 ESM / T5 模型算 embedding：一次載入模型，跑完所有缺快取的序列後立刻釋放 GPU 顯存
    # ------------------------------------------------------------------
    @classmethod
    def _computeEmbedding(cls, methodName, paramLi, seqLi, blockSize):
        # 延遲載入，避免沒開啟 LLM embedding 時，環境也被要求安裝 torch/transformers/esm 這些重量級套件
        from main_T5_ESM import EmbeddingsFeature

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embFeatObj = EmbeddingsFeature(blockSize)

        if methodName == "ESM":
            _, esmModelName = paramLi
            dim = cls._ESM_MODEL_DIM.get(esmModelName, 1280)
            embFeatObj.model_esm, embFeatObj.alphabet = torch.hub.load("facebookresearch/esm:main", esmModelName)
            embFeatObj.model_esm.to(device)
            rowLi = [embFeatObj.esmInfer(seq) for seq in seqLi]
            del embFeatObj.model_esm

        elif methodName == "T5":
            from transformers import T5Tokenizer, T5EncoderModel
            _, t5ModelName = paramLi
            dim = cls._T5_MODEL_DIM_DEFAULT
            embFeatObj.model_t5 = T5EncoderModel.from_pretrained(t5ModelName)
            embFeatObj.tokenizer = T5Tokenizer.from_pretrained(t5ModelName, do_lower_case=False)
            embFeatObj.model_t5.to(device)
            rowLi = [embFeatObj.t5Infer(seq) for seq in seqLi]
            del embFeatObj.model_t5

        else:
            raise ValueError(f"未知的 LLM embedding 方法 '{methodName}'")

        gc.collect()
        torch.cuda.empty_cache()

        return pd.DataFrame(rowLi, index=seqLi, columns=list(range(dim)))