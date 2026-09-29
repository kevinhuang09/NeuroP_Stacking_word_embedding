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
#   - 每一個 "ESM_xxx" / "T5_xxx" key 各自代表一個獨立模型，格式為 [開關, 模型名稱]，
#        可以同時開啟多個 ESM 和/或多個 T5 模型，各自會被當成獨立的 feature type
#        （欄位、embedding 快取 csv 都各自分開，互不覆蓋）。
#        key 的開頭必須是 "ESM"、"T5"、"Ankh"、"Bert" 或 "ProteinGLM"（FeatureStat_llmEmbedding.py 依此判斷要呼叫哪個模型架構去推論），
#        後面接的字串可自訂，只是用來識別／區分同族的不同模型。
#        目前可用的模型名稱（Usage/blockSize 之外，新加入的都先設 False，要用再手動開）：
#        ESM (torch.hub facebookresearch/esm)：
#          esm2_t6_8M_UR50D（320維）、esm2_t12_35M_UR50D（480維）、esm2_t30_150M_UR50D（640維）、
#          esm2_t33_650M_UR50D（1280維）、esm2_t36_3B_UR50D（2560維）、esm2_t48_15B_UR50D（5120維，模型很大）、
#          esm1b_t33_650M_UR50S（1280維）、esm1v_t33_650M_UR90S_1~5（1280維，共5個獨立訓練的checkpoint）
#        T5 (Rostlab, 皆用 T5EncoderModel + T5Tokenizer)：
#          Rostlab/ProstT5、Rostlab/prot_t5_xl_uniref50（皆1024維）、
#          Rostlab/prot_t5_xl_bfd（1024維）、Rostlab/prot_t5_xxl_uniref50（1024維，模型很大~40GB+）、
#          Rostlab/prot_t5_xxl_bfd（1024維，模型很大~40GB+）、Rostlab/prot_t5_base_mt_uniref50（768維）
#        Ankh (ElnaggarLab, 底層架構同為 T5EncoderModel + T5Tokenizer，因此沿用 T5 的推論流程)：
#          ElnaggarLab/ankh-base（768維）、ElnaggarLab/ankh-large（1536維）、
#          ElnaggarLab/ankh2-large（1536維）、ElnaggarLab/ankh2-ext1、ElnaggarLab/ankh2-ext2、
#          ElnaggarLab/ankh3-large、ElnaggarLab/ankh3-xl
#          （ankh2-ext1/ext2/ankh3 系列輸出維度尚未在 FeatureStat_llmEmbedding.py 的 _ANKH_MODEL_DIM 裡確認過，
#           要開啟前請先核對官方文件的實際輸出維度，避免跟保底值 _ANKH_MODEL_DIM_DEFAULT 對不上）
#        Bert (Rostlab ProtTrans 系列，底層架構為 BertModel + BertTokenizer，跟 T5/Ankh 不同族)：
#          Rostlab/prot_bert（1024維）、Rostlab/prot_bert_bfd（1024維）
#        ProteinGLM (BioMap，GLM 架構，需 trust_remote_code=True 才能載入，int4 量化版)：
#          biomap-research/proteinglm-100b-int4（10240維，100B 參數量化後仍需大量 VRAM，模型很大）
#   - "readFromCSV": 是否讀取已存好的 embedding csv 快取，而非重新用模型計算
#        （對應 v3 的 b_readEsmEmbFromCSV / b_readT5EmbFromCSV，這裡合併成單一開關統一控制）

llmEmbeddingFeatureDict = {"Usage": True,        # 整個 LLM embedding 類別總開關
                          "modelDirPath": None,   # 快取 csv 路徑前綴，Main script 會依 dataName 動態設定
                          "blockSize": None,      # padding 長度，Main script 會依全部資料集算好後動態設定
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
                          "ProteinGLM_100B_int4": [False, "biomap-research/proteinglm-100b-int4"],  # 100B int4量化，VRAM需求極高
                          "ProteinGLM_10B_mlm": [True, "biomap-research/proteinglm-10b-mlm"],  # 4352維，proteinglm/ 這個鏡像 org 缺 model.safetensors.index.json，須用 biomap-research
                          "ProteinGLM_7B_clm": [True, "proteinglm/proteinglm-7b-clm"],  # 4096維，此模型 proteinglm/ org 底下檔案齊全，可直接用
                          }

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'centerGDPFeature': centerGDPDict,
               'llmEmbeddingFeature': llmEmbeddingFeatureDict}
