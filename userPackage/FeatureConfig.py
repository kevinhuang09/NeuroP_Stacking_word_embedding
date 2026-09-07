ifeatureDict = {"AAC": True,
                "AAINDEX": False,  # 注意!!需等長
                "CKSAAGP": [True, 5],
                "CTDC": True,
                "CTDD": True,
                "CTDT": True,
                "CTriad": True,  # 會產生feature 343個
                "DDE": True,
                "DPC": True,
                "GAAC": True,
                "GDPC": True,
                "GTPC": True,
                "KSCTriad": [True, 0],  # 會產生feature 343個
                "QSOrder": [False, 30, 0.1],
                "TPC": False,  # 會有8000個feature 不建議開啟
                "SOCN": [False, 3],
                "APAAC": [True, 3, 0.05],
                "Geary": [True, 3],
                "Moran": [True, 3],
                "NMBroto": [True, 3],
                "CKSAAP": [True, 3],  # 會產生feature 1600個
                "BINARY": False,  # 注意!!需等長
                "PAAC": [True, 3, 0.05]
                }

PfeatureDict = {"DDOR": True,
                "RRI": True,
                "SER": True,
                "SEP": True,
                "SE": True,
                "QSO": [True, 3, 0.1]
                }  # gap=3 w=0.1

AMPfeatureDict = {"length": True,
                  "calculate_mw": [True, True],  # [0] = on(True)/off [1] = amide
                  "calculate_charge": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "charge_density": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "isoelectric_point": [True, True],  # [0] = on(True)/off [1] = amide
                  "instability_index": True,
                  "aromaticity": True,
                  "aliphatic_index": True,
                  "hydrophobic": True,
                  "aasi": True,
                  "abhprk": [True, 5],  # [0] = on(True)/off [1] = window
                  "argos": True,
                  "bulkiness": True,
                  "charge_phys": True,
                  "charge_acid": True,
                  "cougar": [True, 5],  # [0] = on(True)/off [1] = window
                  "ez": [True, 5],  # [0] = on(True)/off [1] = window
                  "flexibility": True,
                  "gravy": True,
                  "levitt_alpha": True,
                  "mss": True,
                  "msw": [True, 5],  # [0] = on(True)/off [1] = window
                  "pepArc": True,
                  "polarity": True,
                  "refractivity": True,
                  "tm_tend": True,
                  "z3": [True, 5],  # [0] = on(True)/off [1] = window
                  "z5": [True, 5],  # [0] = on(True)/off [1] = window
                  "formula": True,  # C,H,N,O,S atom composition
                  "boman_index": True,
                  "eisenberg": True,
                  "hopp_woods": True,
                  "janin": True,
                  "kytedoolittle": True
                  }

OVPfeatureDict = {"OVPC": True,
                  "OVP": [True, 4, 4]  # 兩個數為 N, C 端胺基酸數目  N= C=
                  }

MotifBitVecfeatureDict = {"Usage": False,
                          "motifList": ['FKK', 'LKL', 'KKLL', 'KWK', 'VLK',
                                        'CY'
                                        ''
                                        'CR', 'CRR', 'RFC', 'RRR', 'LKKL']
                          }

centerGDPDict = {"Usage": False, "UseGap": False, "gap_size": -1}  # 若是預測每個 amino acid 的 label 在使用

word2VecFeatureDict = {"Usage": True,  # 是否啟用 word embedding 特徵
                     "kmerSize": 3,  # 切 k-mer 的長度(構成 word2vec 中的一個"word")
                     "vectorSize": 100,  # word2vec 向量維度，也是最終輸出的 feature 數
                     "window": 5,  # word2vec 訓練時的上下文視窗大小
                     "minCount": 1,  # k-mer word 出現次數低於此值會被忽略、不建立向量
                     "epochs": 10,  # word2vec 訓練的迭代次數
                     "sg": 1,  # 1 = skip-gram, 0 = CBOW
                     "aggregation": "mean",  # 序列內所有 k-mer 向量的聚合方式: "mean" 或 "sum"
                     "modelPath": None}  # word2vec 模型存檔/讀取路徑，需在 Main 程式依 dataName 動態設定

fastTextFeatureDict = {"Usage": False,  # 是否啟用 FastText 特徵
                       "kmerSize": 3,  # 切 k-mer 的長度(構成 FastText 中的一個"word")
                       "vectorSize": 100,  # FastText 向量維度，也是最終輸出的 feature 數
                       "window": 5,  # FastText 訓練時的上下文視窗大小
                       "minCount": 1,  # k-mer word 出現次數低於此值會被忽略、不建立向量
                       "epochs": 10,  # FastText 訓練的迭代次數
                       "sg": 1,  # 1 = skip-gram, 0 = CBOW
                       "minN": 2,  # 字元 n-gram(子詞)的最小長度
                       "maxN": 4,  # 字元 n-gram(子詞)的最大長度
                       "aggregation": "mean",  # 序列內所有 k-mer 向量的聚合方式: "mean" 或 "sum"
                       "modelPath": None}  # FastText 模型存檔/讀取路徑，需在 Main 程式依 dataName 動態設定

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'centerGDPFeature': centerGDPDict,
               'word2VecFeature': word2VecFeatureDict,
               'fastTextFeature': fastTextFeatureDict}
