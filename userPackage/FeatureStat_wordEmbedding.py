import os
import re
import pickle
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec, FastText
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

GENSIM_MAJOR_VERSION = int(gensim.__version__.split('.')[0])


class _KmerAnalyzer:
    """
    CountVectorizer 的 analyzer 參數如果用 lambda/closure，訓練完的 vectorizer 沒辦法 pickle 存檔
    (LSA 的模型需要連同 vectorizer 一起存檔以便之後載入 transform)，所以改用這個定義在模組層級、
    只帶一個 kmerSize 屬性的 callable class，這樣才能正常被 pickle。
    """

    def __init__(self, kmerSize):
        self.kmerSize = kmerSize

    def __call__(self, sequence):
        return WordEmbeddingFeature._sequenceToKmers(sequence, self.kmerSize)


class WordEmbeddingFeature:
    """
    把胺基酸序列切成固定長度的 k-mer 當作 word embedding 的「word」，依 wordEmbeddingDict 裡
    各個方法(Word2Vec、FastText、NNLM、LSA、PPMI_SVD)各自的開關，訓練(或載入)對應模型，
    再把每條序列轉成一組固定長度的 embedding 特徵；多個方法同時開啟時，各自的欄位會水平合併成
    一份輸出 DataFrame，可當成一種新的 feature type 併入既有的 featureDict 系統
    (用法比照 devPackage 底下各個 XXX(seqDict, featureDict).getOutputDf() 的介面)。

    wordEmbeddingFeature 是一個 type，底下 Word2Vec/FastText/NNLM/LSA/PPMI_SVD 各自是獨立的
    feature type(跟 iFeature 底下 AAC/CTDC/... 各自獨立一樣)，可以同時開啟多個。

    wordEmbeddingDict 的格式：
        "Usage"     : 整個 wordEmbeddingFeature type 的總開關；False 時不論底下各方法開關為何都不會執行
        "Word2Vec"  : [開關, mode("cbow"/"skipgram"), kmer_size, vector_size, window]
        "FastText"  : [開關, mode, kmer_size, vector_size, window, min_n, max_n]
        "NNLM"      : [開關, kmer_size, vector_size, hidden_dim, epochs]
                      單層 feedforward 神經網路語言模型(Bengio-style)：用前一個 kmer 當 context
                      預測下一個 kmer，訓練完後取輸入層 embedding matrix 當每個 kmer 的向量。
        "LSA"       : [開關, kmer_size, n_components]
                      對「序列 x kmer 次數矩陣」做 TruncatedSVD，直接得到每條序列的 n_components 維向量
                      (經典 Latent Semantic Analysis 作法，在文件層級而非 kmer 層級運作)。
        "PPMI_SVD"  : [開關, kmer_size, n_components]
                      先用固定 window(=2) 統計 kmer 兩兩共現次數，計算 Positive PMI 矩陣，
                      再對 PPMI 矩陣做 TruncatedSVD 得到每個 kmer 的 n_components 維向量，
                      最後把一條序列內所有 kmer 向量取平均。
    另外可選的 "modelDirPath" key（字串或 None）用來組出各方法各自的模型存檔路徑
    (f'{modelDirPath}_{methodName小寫}.model')，需在 Main 程式依 dataName 動態設定；
    若為 None 則每次都重新訓練、不存檔。
    "Usage"、"modelDirPath" 都不是方法名稱，不會被當成 word embedding 方法處理。

    modelDirPath 若已存在對應模型檔案，會直接載入該模型做 transform，不會重新訓練；
    若不存在，才會用目前傳入的 seqDict 訓練一份新模型並存檔。
    這是為了讓 DS_Train / DS_Indp / DS_Val 共用「同一份」用訓練集算出來的向量空間，
    避免各自訓練造成向量不可比較、或是資料外洩。
    正式流程請先呼叫 WordEmbeddingFeature.trainAndSaveModel()，
    傳入完整的訓練集序列(正負樣本合併)先訓練好模型，
    之後不論是 encode DS_Train、DS_Indp 或 DS_Val 都只會載入這份模型來 transform。
    """

    _MIN_COUNT = 1
    _EPOCHS = 10
    _PPMI_WINDOW = 2  # PPMI_SVD 共現統計用的固定 window，wordEmbeddingDict 沒有另外開放這個參數

    def __init__(self, seqDict, wordEmbeddingDict):
        self.seqsNameLi = list(seqDict.keys())
        self.modelDirPath = wordEmbeddingDict.get("modelDirPath")
        self.methodItemLi = [(name, paramLi) for name, paramLi in wordEmbeddingDict.items()
                             if name not in ("modelDirPath", "Usage")]
        # 總開關 Usage 為 False 時，不論底下各方法開關為何都不執行
        self.b_start = wordEmbeddingDict.get("Usage", False) is True and any(
            paramLi[0] for _, paramLi in self.methodItemLi)

        if self.b_start is True:
            featureDfLi = []
            for methodName, paramLi in self.methodItemLi:
                if paramLi[0] is not True:
                    continue
                methodModelPath = self._getMethodModelPath(methodName)
                featureDfLi.append(self._runMethod(methodName, paramLi, seqDict, methodModelPath))
            self.featureDf = pd.concat(featureDfLi, axis=1)
        else:
            pass

    def getOutputDf(self):
        if self.b_start is True:
            self.featureDf.index = self.seqsNameLi
            return self.featureDf
        else:
            pass

    def _getMethodModelPath(self, methodName):
        if self.modelDirPath is None:
            return None
        return f'{self.modelDirPath}_{methodName.lower()}.model'

    @staticmethod
    def _sequenceToKmers(sequence, kmerSize):
        sequence = re.sub('-', '', sequence)
        if len(sequence) < kmerSize:
            return [sequence] if sequence else []
        return [sequence[i:i + kmerSize] for i in range(len(sequence) - kmerSize + 1)]

    # ------------------------------------------------------------------
    # 對外用的三個統一入口：訓練/載入一個方法的 artifact、存檔、轉成特徵 DataFrame
    # ------------------------------------------------------------------
    @classmethod
    def _runMethod(cls, methodName, paramLi, seqDict, methodModelPath):
        if methodModelPath is not None and os.path.exists(methodModelPath):
            artifact = cls._loadArtifact(methodName, methodModelPath)
        else:
            artifact = cls._trainArtifact(methodName, paramLi, seqDict)
            if methodModelPath is not None:
                cls._saveArtifact(methodName, artifact, methodModelPath)
        return cls._transformArtifact(methodName, paramLi, seqDict, artifact)

    @classmethod
    def _trainArtifact(cls, methodName, paramLi, seqDict):
        if methodName == "Word2Vec":
            _, mode, kmerSize, vectorSize, window = paramLi
            sg = 1 if mode == "skipgram" else 0
            return cls._trainWord2Vec(seqDict, kmerSize, vectorSize, window, sg)
        elif methodName == "FastText":
            _, mode, kmerSize, vectorSize, window, minN, maxN = paramLi
            sg = 1 if mode == "skipgram" else 0
            return cls._trainFastText(seqDict, kmerSize, vectorSize, window, sg, minN, maxN)
        elif methodName == "NNLM":
            _, kmerSize, vectorSize, hiddenDim, epochs = paramLi
            return cls._trainNNLM(seqDict, kmerSize, vectorSize, hiddenDim, epochs)
        elif methodName == "LSA":
            _, kmerSize, nComponents = paramLi
            return cls._trainLSA(seqDict, kmerSize, nComponents)
        elif methodName == "PPMI_SVD":
            _, kmerSize, nComponents = paramLi
            return cls._trainPPMISVD(seqDict, kmerSize, nComponents)
        else:
            raise ValueError(f"未知的 word embedding 方法 '{methodName}'")

    @classmethod
    def _transformArtifact(cls, methodName, paramLi, seqDict, artifact):
        if methodName == "Word2Vec":
            _, _, kmerSize, vectorSize, _ = paramLi
            return cls._transform(seqDict, artifact, kmerSize, vectorSize, 'word2Vec', allowOov=False)
        elif methodName == "FastText":
            _, _, kmerSize, vectorSize, _, _, _ = paramLi
            # FastText 可用子詞組出未知 word 的向量，不需先篩選 kmer 是否存在於 model.wv 中
            return cls._transform(seqDict, artifact, kmerSize, vectorSize, 'fastText', allowOov=True)
        elif methodName == "NNLM":
            _, kmerSize, _, _, _ = paramLi
            vocabIndexDict, embeddingMatrix = artifact
            return cls._transformFromTable(seqDict, vocabIndexDict, embeddingMatrix, kmerSize, 'nnlm')
        elif methodName == "LSA":
            return cls._transformLSA(seqDict, artifact, 'lsa')
        elif methodName == "PPMI_SVD":
            _, kmerSize, _ = paramLi
            vocabIndexDict, embeddingMatrix = artifact
            return cls._transformFromTable(seqDict, vocabIndexDict, embeddingMatrix, kmerSize, 'ppmiSvd')
        else:
            raise ValueError(f"未知的 word embedding 方法 '{methodName}'")

    @classmethod
    def _saveArtifact(cls, methodName, artifact, modelPath):
        modelDir = os.path.dirname(modelPath)
        if modelDir:
            os.makedirs(modelDir, exist_ok=True)
        if methodName in ("Word2Vec", "FastText"):
            artifact.save(modelPath)  # gensim 原生存檔格式
        else:
            # NNLM/PPMI_SVD 存 (vocabIndexDict, embeddingMatrix)，LSA 存 (vectorizer, svd, nComponents)
            with open(modelPath, 'wb') as f:
                pickle.dump(artifact, f)

    @classmethod
    def _loadArtifact(cls, methodName, modelPath):
        if methodName == "Word2Vec":
            return Word2Vec.load(modelPath)
        elif methodName == "FastText":
            return FastText.load(modelPath)
        else:
            with open(modelPath, 'rb') as f:
                return pickle.load(f)

    # ------------------------------------------------------------------
    # Word2Vec / FastText：沿用 gensim，kmer 向量取平均當序列特徵
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 共用工具：kmer -> 向量 對照表(NNLM/PPMI_SVD 共用)、欄位數固定補零
    # ------------------------------------------------------------------
    @classmethod
    def _transformFromTable(cls, seqDict, vocabIndexDict, embeddingMatrix, kmerSize, columnPrefix):
        vectorSize = embeddingMatrix.shape[1]
        rowLi = []
        for sequence in seqDict.values():
            kmerLi = cls._sequenceToKmers(sequence, kmerSize)
            vectorLi = [embeddingMatrix[vocabIndexDict[kmer]] for kmer in kmerLi if kmer in vocabIndexDict]
            rowVector = np.zeros(vectorSize) if len(vectorLi) == 0 else np.mean(vectorLi, axis=0)
            rowLi.append(rowVector)
        columnNameLi = [f'{columnPrefix}_{i}' for i in range(vectorSize)]
        return pd.DataFrame(rowLi, columns=columnNameLi)

    @staticmethod
    def _padColumnsToWidth(matrix, targetWidth):
        """
        訓練資料的 kmer 詞彙量太小時，TruncatedSVD 實際能算出的維度會小於要求的 n_components，
        這裡固定補 0 補到 targetWidth，確保 LSA/PPMI_SVD 輸出的欄位數永遠等於設定裡的 n_components，
        不會因為 column-discovery 用的樣本數(5 筆)跟正式訓練集資料量不同而長出不一樣的欄位數。
        """
        currentWidth = matrix.shape[1]
        if currentWidth >= targetWidth:
            return matrix[:, :targetWidth]
        pad = np.zeros((matrix.shape[0], targetWidth - currentWidth))
        return np.hstack([matrix, pad])

    # ------------------------------------------------------------------
    # NNLM：Bengio-style 單層 feedforward 神經網路語言模型
    # 用前一個 kmer(context) 預測下一個 kmer(target)，訓練完取輸入層 embedding matrix 當 kmer 向量
    # ------------------------------------------------------------------
    @classmethod
    def _trainNNLM(cls, seqDict, kmerSize, vectorSize, hiddenDim, epochs):
        kmerSentenceLi = [cls._sequenceToKmers(seq, kmerSize) for seq in seqDict.values()]
        vocabLi = sorted({kmer for sentence in kmerSentenceLi for kmer in sentence})
        vocabIndexDict = {kmer: i for i, kmer in enumerate(vocabLi)}
        vocabSize = len(vocabLi)

        contextIdxLi, targetIdxLi = [], []
        for sentence in kmerSentenceLi:
            for i in range(1, len(sentence)):
                contextIdxLi.append(vocabIndexDict[sentence[i - 1]])
                targetIdxLi.append(vocabIndexDict[sentence[i]])

        # 詞彙量太小或每條序列都只切得出一個 kmer 時，湊不出 (context, target) pair，直接回傳零向量表
        if vocabSize == 0 or len(contextIdxLi) == 0:
            return vocabIndexDict, np.zeros((max(vocabSize, 1), vectorSize))

        contextIdx = np.array(contextIdxLi)
        targetIdx = np.array(targetIdxLi)
        n = len(contextIdx)

        rng = np.random.default_rng(42)
        E = rng.normal(0, 0.1, size=(vocabSize, vectorSize))       # 輸入層 embedding，訓練完就是輸出結果
        W1 = rng.normal(0, 0.1, size=(vectorSize, hiddenDim))
        b1 = np.zeros(hiddenDim)
        W2 = rng.normal(0, 0.1, size=(hiddenDim, vocabSize))
        b2 = np.zeros(vocabSize)

        learningRate = 0.05
        for _ in range(epochs):
            x = E[contextIdx]                                     # (n, vectorSize)
            h = np.tanh(x @ W1 + b1)                                # (n, hiddenDim)
            logits = h @ W2 + b2                                    # (n, vocabSize)
            logits -= logits.max(axis=1, keepdims=True)             # softmax 前先減最大值避免溢位
            expLogits = np.exp(logits)
            probs = expLogits / expLogits.sum(axis=1, keepdims=True)

            dLogits = probs
            dLogits[np.arange(n), targetIdx] -= 1
            dLogits /= n

            dW2 = h.T @ dLogits
            db2 = dLogits.sum(axis=0)
            dh = dLogits @ W2.T
            dhRaw = dh * (1 - h ** 2)                               # tanh 導數
            dW1 = x.T @ dhRaw
            db1 = dhRaw.sum(axis=0)
            dx = dhRaw @ W1.T

            dE = np.zeros_like(E)
            np.add.at(dE, contextIdx, dx)                           # 同一個 context word 出現多次時梯度要累加

            E -= learningRate * dE
            W1 -= learningRate * dW1
            b1 -= learningRate * db1
            W2 -= learningRate * dW2
            b2 -= learningRate * db2

        return vocabIndexDict, E

    # ------------------------------------------------------------------
    # LSA：對「序列 x kmer 次數矩陣」做 TruncatedSVD，直接得到文件層級的向量
    # ------------------------------------------------------------------
    @classmethod
    def _trainLSA(cls, seqDict, kmerSize, nComponents):
        vectorizer = CountVectorizer(analyzer=_KmerAnalyzer(kmerSize))
        countMatrix = vectorizer.fit_transform(list(seqDict.values()))

        # TruncatedSVD 要求 n_components < min(n_samples, n_features)，訓練資料(尤其 column-discovery
        # 用的 5 筆樣本)太小時沒辦法用到要求的 n_components，退化成用能算的最大維度，輸出時再補零到 nComponents
        maxValidComponents = min(countMatrix.shape[0], countMatrix.shape[1]) - 1
        if maxValidComponents < 1:
            return (vectorizer, None, nComponents)

        svd = TruncatedSVD(n_components=min(nComponents, maxValidComponents), random_state=42)
        svd.fit(countMatrix)
        return (vectorizer, svd, nComponents)

    @classmethod
    def _transformLSA(cls, seqDict, artifact, columnPrefix):
        vectorizer, svd, nComponents = artifact
        countMatrix = vectorizer.transform(list(seqDict.values()))
        if svd is None:
            reducedMatrix = np.zeros((countMatrix.shape[0], nComponents))
        else:
            reducedMatrix = cls._padColumnsToWidth(svd.transform(countMatrix), nComponents)
        columnNameLi = [f'{columnPrefix}_{i}' for i in range(nComponents)]
        return pd.DataFrame(reducedMatrix, columns=columnNameLi)

    # ------------------------------------------------------------------
    # PPMI_SVD：kmer 兩兩共現 -> Positive PMI 矩陣 -> TruncatedSVD 得到 kmer 向量 -> 序列內取平均
    # ------------------------------------------------------------------
    @classmethod
    def _trainPPMISVD(cls, seqDict, kmerSize, nComponents):
        kmerSentenceLi = [cls._sequenceToKmers(seq, kmerSize) for seq in seqDict.values()]
        vocabLi = sorted({kmer for sentence in kmerSentenceLi for kmer in sentence})
        vocabIndexDict = {kmer: i for i, kmer in enumerate(vocabLi)}
        vocabSize = len(vocabLi)

        # 詞彙量太小時 TruncatedSVD 無法運作(n_components 必須 < 詞彙量)，直接回傳零向量表
        if vocabSize <= 1:
            return vocabIndexDict, np.zeros((vocabSize, nComponents))

        window = cls._PPMI_WINDOW
        rowIdxLi, colIdxLi = [], []
        for sentence in kmerSentenceLi:
            idxLi = [vocabIndexDict[kmer] for kmer in sentence]
            for i, centerIdx in enumerate(idxLi):
                for j in range(max(0, i - window), min(len(idxLi), i + window + 1)):
                    if i == j:
                        continue
                    rowIdxLi.append(centerIdx)
                    colIdxLi.append(idxLi[j])

        if len(rowIdxLi) == 0:
            return vocabIndexDict, np.zeros((vocabSize, nComponents))

        coMatrix = sparse.coo_matrix((np.ones(len(rowIdxLi)), (rowIdxLi, colIdxLi)),
                                     shape=(vocabSize, vocabSize)).tocsr()

        totalCount = coMatrix.sum()
        rowSum = np.asarray(coMatrix.sum(axis=1)).flatten()
        colSum = np.asarray(coMatrix.sum(axis=0)).flatten()

        coo = coMatrix.tocoo()
        ppmiData = np.log((coo.data * totalCount) / (rowSum[coo.row] * colSum[coo.col]))
        ppmiData = np.maximum(ppmiData, 0)  # 只保留正值，這就是 Positive PMI
        ppmiMatrix = sparse.coo_matrix((ppmiData, (coo.row, coo.col)), shape=(vocabSize, vocabSize)).tocsr()

        effectiveComponents = min(nComponents, vocabSize - 1)
        svd = TruncatedSVD(n_components=effectiveComponents, random_state=42)
        embeddingMatrix = cls._padColumnsToWidth(svd.fit_transform(ppmiMatrix), nComponents)
        return vocabIndexDict, embeddingMatrix

    @classmethod
    def trainAndSaveModel(cls, seqDict, wordEmbeddingDict):
        """
        用完整的序列集合(例如 DS_Train 正負樣本合併)，把 wordEmbeddingDict 中每個有開啟的方法
        各自訓練一次模型並存到 f'{wordEmbeddingDict["modelDirPath"]}_{methodName小寫}.model'。
        請在對 DS_Train/DS_Indp/DS_Val 執行 dataEncodeOutPut() 之前呼叫這個函式一次，
        確保三者都是載入同一份用訓練集訓練好的模型來做 transform。
        """
        if wordEmbeddingDict.get("Usage", False) is not True:
            return

        modelDirPath = wordEmbeddingDict.get("modelDirPath")
        for methodName, paramLi in wordEmbeddingDict.items():
            if methodName in ("modelDirPath", "Usage") or paramLi[0] is not True:
                continue
            artifact = cls._trainArtifact(methodName, paramLi, seqDict)
            if modelDirPath is not None:
                cls._saveArtifact(methodName, artifact, f'{modelDirPath}_{methodName.lower()}.model')
