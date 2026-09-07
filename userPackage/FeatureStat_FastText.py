import os
import re
import numpy as np
import pandas as pd
from gensim.models import FastText


class FastTextFeature:
    """
    把胺基酸序列切成固定長度的 k-mer 當作 FastText 的「word」，訓練(或載入)一份 FastText 模型，
    再把每條序列所有 k-mer word 的向量做平均或加總，得到一組固定長度(vectorSize)的 embedding 特徵，
    可當成一種新的 feature type 併入既有的 featureDict 系統(用法比照 FeatureStat_word_em.py 的
    WordEmbeddingFeature，介面跟 devPackage 底下各個 XXX(seqDict, featureDict).getOutputDf() 一致)。

    跟 Word2Vec 版本(FeatureStat_word_em.py)最大的差異是 FastText 會額外利用字元 n-gram(minN~maxN)
    學習「子詞」資訊，因此對訓練時沒看過的 k-mer(未知 word) 也能組出向量，不會像 Word2Vec 一樣直接查無該詞。

    featureDict 需要的 key：
        Usage        : 是否啟用此 feature type
        kmerSize     : 切 k-mer 的長度
        vectorSize   : FastText 向量維度，也是最終輸出的 feature 數
        window       : FastText 訓練時的上下文視窗大小
        minCount     : k-mer word 出現次數低於此值會被忽略、不建立向量
        epochs       : FastText 訓練的迭代次數
        sg           : 1 = skip-gram, 0 = CBOW
        minN         : 字元 n-gram(子詞)的最小長度
        maxN         : 字元 n-gram(子詞)的最大長度
        aggregation  : "mean" 或 "sum"，決定一條序列內所有 k-mer 向量的聚合方式
        modelPath    : FastText 模型存檔/讀取路徑

    modelPath 若已存在模型檔案，會直接載入該模型做 transform，不會重新訓練；
    若不存在，才會用目前傳入的 seqDict 訓練一份新模型並存檔。
    這是為了讓 DS_Train / DS_Indp / DS_Val 共用「同一份」用訓練集算出來的向量空間，
    避免各自訓練造成向量不可比較、或是資料外洩。
    正式流程請先呼叫 FastTextFeature.trainAndSaveModel()，
    傳入完整的訓練集序列(正負樣本合併)先訓練好模型，
    之後不論是 encode DS_Train、DS_Indp 或 DS_Val 都只會載入這份模型來 transform。
    """

    def __init__(self, seqDict, featureDict):
        self.b_start = featureDict.get("Usage", False)
        if self.b_start is True:
            self.seqsNameLi = list(seqDict.keys())
            self.kmerSize = featureDict.get("kmerSize", 3)
            self.vectorSize = featureDict.get("vectorSize", 100)
            self.aggregation = featureDict.get("aggregation", "mean")
            self.modelPath = featureDict.get("modelPath")

            if self.modelPath is not None and os.path.exists(self.modelPath):
                model = FastText.load(self.modelPath)
            else:
                model = self._trainModel(seqDict, featureDict)
                if self.modelPath is not None:
                    self._saveModel(model, self.modelPath)

            self.featureDf = self._transform(seqDict, model, self.kmerSize, self.vectorSize, self.aggregation)
        else:
            pass

    def getOutputDf(self):
        if self.b_start is True:
            self.featureDf.index = self.seqsNameLi
            return self.featureDf
        else:
            pass

    @staticmethod
    def _sequenceToKmers(sequence, kmerSize):
        sequence = re.sub('-', '', sequence)
        if len(sequence) < kmerSize:
            return [sequence] if sequence else []
        return [sequence[i:i + kmerSize] for i in range(len(sequence) - kmerSize + 1)]

    @staticmethod
    def _trainModel(seqDict, featureDict):
        kmerSize = featureDict.get("kmerSize", 3)
        kmerSentenceLi = [FastTextFeature._sequenceToKmers(seq, kmerSize) for seq in seqDict.values()]
        model = FastText(sentences=kmerSentenceLi,
                         vector_size=featureDict.get("vectorSize", 100),
                         window=featureDict.get("window", 5),
                         min_count=featureDict.get("minCount", 1),
                         sg=featureDict.get("sg", 1),
                         epochs=featureDict.get("epochs", 10),
                         min_n=featureDict.get("minN", 2),
                         max_n=featureDict.get("maxN", 4),
                         workers=1,
                         seed=42)
        return model

    @staticmethod
    def _saveModel(model, modelPath):
        modelDir = os.path.dirname(modelPath)
        if modelDir:
            os.makedirs(modelDir, exist_ok=True)
        model.save(modelPath)

    @staticmethod
    def _transform(seqDict, model, kmerSize, vectorSize, aggregation):
        rowLi = []
        for sequence in seqDict.values():
            kmerLi = FastTextFeature._sequenceToKmers(sequence, kmerSize)
            vectorLi = [model.wv[kmer] for kmer in kmerLi]  # FastText 可用子詞組出未知 word 的向量，不需先篩選是否存在
            if len(vectorLi) == 0:
                rowVector = np.zeros(vectorSize)
            elif aggregation == "sum":
                rowVector = np.sum(vectorLi, axis=0)
            else:
                rowVector = np.mean(vectorLi, axis=0)
            rowLi.append(rowVector)
        columnNameLi = [f'fastText_{i}' for i in range(vectorSize)]
        return pd.DataFrame(rowLi, columns=columnNameLi)

    @classmethod
    def trainAndSaveModel(cls, seqDict, featureDict):
        """
        用完整的序列集合(例如 DS_Train 正負樣本合併)訓練 FastText 模型並存到 featureDict['modelPath']。
        請在對 DS_Train/DS_Indp/DS_Val 執行 dataEncodeOutPut() 之前呼叫這個函式一次，
        確保三者都是載入同一份用訓練集訓練好的模型來做 transform。
        """
        model = cls._trainModel(seqDict, featureDict)
        cls._saveModel(model, featureDict["modelPath"])
        return model
