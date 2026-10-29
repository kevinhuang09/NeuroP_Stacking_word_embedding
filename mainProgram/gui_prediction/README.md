# gui_prediction

輸入胺基酸序列（`predict.csv`），用專案裡分數最高的 Lv2 stacking **Random Forest** 模型
(`data/finalModel_emb/Lv2_best/standard/rf_final.pkl`，session 42，對 DS_Indp 的表現
accuracy=0.9471、auc=0.9868、mcc=0.8943) 判定是否為神經肽(neuropeptide)，輸出預測機率與
Positive/Negative 判定（門檻 0.5）。

## 這個模型實際在做什麼

這不是一個「序列進、模型出」的單一分類器，而是論文裡完整的兩層 stacking：

1. 序列先算出一堆 classical 特徵(iFeature/pFeature/ampFeature/ovpFeature子特徵)跟
   7 種蛋白質語言模型 embedding(ESM-650M、ESM-3B、T5-ProstT5、T5-XL-UniRef50、
   ProtBert、ProtBertBfd、ProteinGLM-10B)。
2. 每種特徵各自丟進對應已訓練好的 Lv1 base learner(17 種 ML model 裡各自表現最好的
   那幾個)，產生 70 個機率值(meta-feature)。
3. 這 70 個 meta-feature 丟進 Lv2 的 RandomForest，得到最終機率。

因為牽涉到大型語言模型(其中 ProteinGLM-10B、ESM-3B 規模達數十億參數)，所以**要分兩個
conda 環境各跑一支程式**，跟專案既有的 `precompute_llmEmbedding.py` /
`Main_FeatureStk_emb.py` 兩階段流程是同一個理由：pycaret 環境(`neuropredictor-tool`)
鎖在 Python 3.8 + 舊版 sklearn，裝不了新版 torch；裝得動新版 torch 的環境又沒有 pycaret。

## 用法

1. 編輯 `predict.csv`，欄位需含 `sequence`（`id` 可省略，省略會自動編號）。資料夾裡已附
   兩個範例序列(一正一負)可以直接試跑。

2. 在**有 torch/transformers 的環境**算好 embedding 快取(序列的 embedding 會存進
   `data/param/NeuroP_1_llmEmbedding_*_cache.csv`，以序列內容當 key，算過一次之後
   不管哪個 predict.csv 用到同一條序列都能直接命中快取)：

   ```bash
   conda run -n base python precompute_embedding.py predict.csv
   ```

   第一次對沒出現過的序列執行，會自動下載 ESM-3B / ProteinGLM-10B 等模型權重
   (數十 GB)並用 GPU 推論，請確保網路、硬碟空間足夠且 GPU 沒有被佔滿。範例序列已經在
   訓練資料的快取裡，可以直接跳過這一步、秒級完成。

3. 切到 **pycaret 環境**(`neuropredictor-tool`)做完整預測：

   ```bash
   conda run -n neuropredictor-tool python predict.py predict.csv predict_result.csv
   ```

   這支程式完全不會 import torch，只讀快取。輸出的 `predict_result.csv` 欄位為
   `id, sequence, probability, prediction`。

## 檔案說明

- `config.py`：共用路徑設定、序列/表頭讀取工具、把 FeatureConfig 縮減成 70 個
  meta-feature實際用得到的子特徵開關。
- `precompute_embedding.py`：在 torch 環境算/確認 LLM embedding 快取。
- `predict.py`：在 pycaret 環境做完整的 Lv1 → Lv2 stacking 預測。

## 已知限制

- 目前接的是 `standard` normalization 版本的 Lv2_best RandomForest(存檔裡分數最高的一次
  RandomForest 結果)。若之後想換成分數更高的其他 Lv2 model(例如 qda/mlp/et，數字上比
  RandomForest 更高)，只需要在 `config.py` 把 `LV2_MODEL_NAME` 換掉即可，其餘流程不用動。
- `data/tuneModel_test_emb/`、`data/finalModel_emb/Lv2_best/` 等資料夾若之後被覆蓋/
  重新訓練，須重新確認 70 個 meta-feature 用到的 (feature type, model) 組合沒有改變。
