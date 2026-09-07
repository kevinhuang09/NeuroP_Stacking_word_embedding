import os
import re
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec, FastText

GENSIM_MAJOR_VERSION = int(gensim.__version__.split('.')[0])


class WordEmbeddingFeature:
    """
    把胺基酸序列切成固定長度的 k-mer 當作 word embedding 的「word」，依 wordEmbeddingDict 裡
    "method" 指定的單一方法(Word2Vec、FastText...)訓練(或載入)對應模型，再把每條序列所有 k-mer word
    的向量取平均，得到一組固定長度(vector_size)的 embedding 特徵，可當成一種新的 feature type
    併入既有的 featureDict 系統(用法比照 devPackage 底下各個 XXX(seqDict, featureDict).getOutputDf()
    的介面)。

    wordEmbeddingDict 的格式：
        "Usage"       : 是否啟用 word embedding 特徵，整組共用同一個開關
        "method"      : 啟用時實際使用哪一種方法，"Word2Vec"/"FastText"/"NNLM"/"LSA"/"PPMI_SVD" 其中之一
        "Word2Vec"    : [mode("cbow"/"skipgram"), kmer_size, vector_size, window]
        "FastText"    : [mode, kmer_size, vector_size, window, min_n, max_n]
        "NNLM"        : [kmer_size, vector_size, hidden_dim, epochs]（尚未實作，選用會直接報錯）
        "LSA"         : [kmer_size, n_components]（尚未實作，選用會直接報錯）
        "PPMI_SVD"    : [kmer_size, n_components]（尚未實作，選用會直接報錯）
        "modelDirPath": 模型存檔路徑前綴（字串或 None），實際存檔路徑為
                        f'{modelDirPath}_{method小寫}.model'，需在 Main 程式依 dataName 動態設定；
                        若為 None 則每次都重新訓練、不存檔。

    modelDirPath 對應的模型檔案若已存在，會直接載入該模型做 transform，不會重新訓練；
    若不存在，才會用目前傳入的 seqDict 訓練一份新模型並存檔。
    這是為了讓 DS_Train / DS_Indp / DS_Val 共用「同一份」用訓練集算出來的向量空間，
    避免各自訓練造成向量不可比較、或是資料外洩。
    正式流程請先呼叫 WordEmbeddingFeature.trainAndSaveModel()，
    傳入完整的訓練集序列(正負樣本合併)先訓練好模型，
    之後不論是 encode DS_Train、DS_Indp 或 DS_Val 都只會載入這份模型來 transform。
    """

    _MIN_COUNT = 1
    _EPOCHS = 10
    _NOT_IMPLEMENTED_METHOD_LIST = ("NNLM", "LSA", "PPMI_SVD")

    def __init__(self, seqDict, wordEmbeddingDict):
        self.seqsNameLi = list(seqDict.keys())
        self.b_start = wordEmbeddingDict.get("Usage", False)

        if self.b_start is True:
            method = wordEmbeddingDict["method"]
            paramLi = wordEmbeddingDict[method]
            modelPath = self._getModelPath(wordEmbeddingDict.get("modelDirPath"), method)
            self.featureDf = self._runMethod(method, paramLi, seqDict, modelPath)
        else:
            pass

    def getOutputDf(self):
        if self.b_start is True:
            self.featureDf.index = self.seqsNameLi
            return self.featureDf
        else:
            pass

    @staticmethod
    def _getModelPath(modelDirPath, method):
        if modelDirPath is None:
            return None
        return f'{modelDirPath}_{method.lower()}.model'

    @staticmethod
    def _sequenceToKmers(sequence, kmerSize):
        sequence = re.sub('-', '', sequence)
        if len(sequence) < kmerSize:
            return [sequence] if sequence else []
        return [sequence[i:i + kmerSize] for i in range(len(sequence) - kmerSize + 1)]

    @classmethod
    def _runMethod(cls, method, paramLi, seqDict, modelPath):
        if method == "Word2Vec":
            mode, kmerSize, vectorSize, window = paramLi
            sg = 1 if mode == "skipgram" else 0
            if modelPath is not None and os.path.exists(modelPath):
                model = Word2Vec.load(modelPath)
            else:
                model = cls._trainWord2Vec(seqDict, kmerSize, vectorSize, window, sg)
                if modelPath is not None:
                    cls._saveModel(model, modelPath)
            return cls._transform(seqDict, model, kmerSize, vectorSize, 'word2Vec', allowOov=False)

        elif method == "FastText":
            mode, kmerSize, vectorSize, window, minN, maxN = paramLi
            sg = 1 if mode == "skipgram" else 0
            if modelPath is not None and os.path.exists(modelPath):
                model = FastText.load(modelPath)
            else:
                model = cls._trainFastText(seqDict, kmerSize, vectorSize, window, sg, minN, maxN)
                if modelPath is not None:
                    cls._saveModel(model, modelPath)
            # FastText 可用子詞組出未知 word 的向量，不需先篩選 kmer 是否存在於 model.wv 中
            return cls._transform(seqDict, model, kmerSize, vectorSize, 'fastText', allowOov=True)

        elif method in cls._NOT_IMPLEMENTED_METHOD_LIST:
            raise NotImplementedError(f"word embedding 方法 '{method}' 尚未實作，請先改用其他 method")
        else:
            raise ValueError(f"未知的 word embedding 方法 '{method}'")

    @classmethod
    def _trainWord2Vec(cls, seqDict, kmerSize, vectorSize, window, sg):
        kmerSentenceLi = [cls._sequenceToKmers(seq, kmerSize) for seq in seqDict.values()]
        commonKwargs = dict(sentences=kmerSentenceLi, window=window, min_count=cls._MIN_COUNT,
                            sg=sg, workers=1, seed=42)
        # gensim 4.0 把 size/iter 改名成 vector_size/epochs，兩個版本的參數名稱不相容，
        # 依安裝的 gensim 主版本號分別組出對應的關鍵字參數，讓程式碼同時相容 gensim 3.x 與 4.x
        if GENSIM_MAJOR_VERSION >= 4:
            return Word2Vec(vector_size=vectorSize, epochs=cls._EPOCHS, **commonKwargs)
        return Word2Vec(size=vectorSize, iter=cls._EPOCHS, **commonKwargs)

    @classmethod
    def _trainFastText(cls, seqDict, kmerSize, vectorSize, window, sg, minN, maxN):
        kmerSentenceLi = [cls._sequenceToKmers(seq, kmerSize) for seq in seqDict.values()]
        commonKwargs = dict(sentences=kmerSentenceLi, window=window, min_count=cls._MIN_COUNT,
                            sg=sg, min_n=minN, max_n=maxN, workers=1, seed=42)
        if GENSIM_MAJOR_VERSION >= 4:
            return FastText(vector_size=vectorSize, epochs=cls._EPOCHS, **commonKwargs)
        return FastText(size=vectorSize, iter=cls._EPOCHS, **commonKwargs)

    @staticmethod
    def _saveModel(model, modelPath):
        modelDir = os.path.dirname(modelPath)
        if modelDir:
            os.makedirs(modelDir, exist_ok=True)
        model.save(modelPath)

    @classmethod
    def _transform(cls, seqDict, model, kmerSize, vectorSize, columnPrefix, allowOov):
        rowLi = []
        for sequence in seqDict.values():
            kmerLi = cls._sequenceToKmers(sequence, kmerSize)
            if allowOov:
                vectorLi = [model.wv[kmer] for kmer in kmerLi]
            else:
                vectorLi = [model.wv[kmer] for kmer in kmerLi if kmer in model.wv]
            rowVector = np.zeros(vectorSize) if len(vectorLi) == 0 else np.mean(vectorLi, axis=0)
            rowLi.append(rowVector)
        columnNameLi = [f'{columnPrefix}_{i}' for i in range(vectorSize)]
        return pd.DataFrame(rowLi, columns=columnNameLi)

    @classmethod
    def trainAndSaveModel(cls, seqDict, wordEmbeddingDict):
        """
        用完整的序列集合(例如 DS_Train 正負樣本合併)，訓練 wordEmbeddingDict["method"] 指定的方法一次，
        並存到 f'{wordEmbeddingDict["modelDirPath"]}_{method小寫}.model'。
        請在對 DS_Train/DS_Indp/DS_Val 執行 dataEncodeOutPut() 之前呼叫這個函式一次，
        確保三者都是載入同一份用訓練集訓練好的模型來做 transform。
        """
        method = wordEmbeddingDict["method"]
        paramLi = wordEmbeddingDict[method]

        if method == "Word2Vec":
            mode, kmerSize, vectorSize, window = paramLi
            sg = 1 if mode == "skipgram" else 0
            model = cls._trainWord2Vec(seqDict, kmerSize, vectorSize, window, sg)
        elif method == "FastText":
            mode, kmerSize, vectorSize, window, minN, maxN = paramLi
            sg = 1 if mode == "skipgram" else 0
            model = cls._trainFastText(seqDict, kmerSize, vectorSize, window, sg, minN, maxN)
        elif method in cls._NOT_IMPLEMENTED_METHOD_LIST:
            raise NotImplementedError(f"word embedding 方法 '{method}' 尚未實作，請先改用其他 method")
        else:
            raise ValueError(f"未知的 word embedding 方法 '{method}'")

        modelDirPath = wordEmbeddingDict.get("modelDirPath")
        if modelDirPath is not None:
            cls._saveModel(model, cls._getModelPath(modelDirPath, method))
        return model
