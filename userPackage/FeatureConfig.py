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

# ======================================================================================================================
# LLM / 蛋白質語言模型 embedding 特徵 (ESM / T5)
# 結構規則：
#   - "Usage": 整個 LLM embedding 類別的總開關。關掉時，底下每個方法即使開著也不會真的產生欄位。
#   - "modelDirPath": embedding 快取 csv 存放路徑前綴（對應 v3 的 tranEsmNmlzCsvPath / trainT5NmlzCsvPath），
#        會被 Main script 動態覆寫成含 dataName 的路徑，此處的值只是保底用。
#   - "ESM": [開關, 模型名稱]。模型名稱對應 v3 的 esmModel，例如
#        "esm2_t33_650M_UR50D"（1280維） 或 "esm2_t36_3B_UR50D"（2560維）
#   - "T5":  [開關, 模型名稱]。模型名稱對應 v3 的 t5Model，例如
#        "Rostlab/ProstT5" 或 "Rostlab/prot_t5_xl_uniref50"（皆為1024維）
#   - "readFromCSV": 是否讀取已存好的 embedding csv 快取，而非重新用模型計算
#        （對應 v3 的 b_readEsmEmbFromCSV / b_readT5EmbFromCSV，這裡合併成單一開關統一控制）
llmEmbeddingFeatureDict = {"Usage": True,        # 整個 LLM embedding 類別總開關
                          "modelDirPath": None,   # 快取 csv 路徑前綴，Main script 會依 dataName 動態設定
                          "blockSize": None,      # padding 長度，Main script 會依全部資料集算好後動態設定
                          "ESM": [True, "esm2_t33_650M_UR50D"],
                          "T5": [True, "Rostlab/ProstT5"],
                          }

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'centerGDPFeature': centerGDPDict,
               'llmEmbeddingFeature': llmEmbeddingFeatureDict}
