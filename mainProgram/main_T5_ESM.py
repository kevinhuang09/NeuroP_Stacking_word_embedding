import re
import torch
import gc
import numpy as np
from transformers import T5Tokenizer,T5EncoderModel

class EmbeddingsFeature:
    def __init__(self, block_size):
        self.block_size = block_size

    def t5Infer(self, seq):
        '''
        T5為每條序列生成的1x1024的embeddings特徵 回傳為dataframe
        :param seq: peptide序列
        :param block_size:block_size
        :return: dataframe內涵embeddings
        '''
        # 将序列转换为空格分隔的形式
        sequences_Example = [" ".join(list(seq))]

        # 将序列中的 [UZOB] 替换为 X
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

        # 新版 transformers 移除了 batch_encode_plus 這個公開方法，改用 tokenizer 的 __call__
        # 不再 padding/truncate 到 self.block_size，讓每條序列依照自己實際長度做 tokenize，
        # 只保留一個很寬鬆的安全上限，避免異常長序列造成記憶體爆炸
        ids = self.tokenizer(
            sequences_Example,
            add_special_tokens=True,
            truncation=True,
            max_length=1024
        )

        # 转移到 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 将模型转移到 GPU
        self.model_t5.to(device)

        with torch.no_grad():
            # 使用模型进行推理
            embedding = self.model_t5(input_ids=input_ids, attention_mask=attention_mask)

        # 获取编码层的嵌入
        encoder_embedding = embedding.last_hidden_state[0, :-1].detach().cpu()
        encoder_embedding = np.array(encoder_embedding.tolist())
        # 返回嵌入向量的总和
        return encoder_embedding.sum(axis=0)


    def esmInfer(self, seq):
        '''
        T5 為每條序列生成的 1x1280 或 1x2560 的 embeddings 特徵，回傳為 dataframe
        :param seq: peptide 序列
        :return: dataframe 內含 embeddings
        '''
        batch_converter = self.alphabet.get_batch_converter()
        self.model_esm.eval()  # 禁用 dropout 以獲得確定性結果

        # 準備數據
        data = [("tmp", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # 選擇設備（GPU 優先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ⚠️【保持 batch_tokens 為 torch.long，不能轉換為 FP16】⚠️
        batch_tokens = batch_tokens.to(device)  # 這裡保持 int64，不轉換 dtype

        # 將模型轉換為 FP16
        self.model_esm.to(device).to(torch.float16)

        # 不再 padding/truncate 到 self.block_size，讓每條序列依照自己實際長度推論；
        # 只保留一個很寬鬆的安全上限，避免異常長序列造成記憶體爆炸
        if batch_tokens.size(1) > 1024:
            batch_tokens = batch_tokens[:, :1024]

        with torch.no_grad():
            # 🔥 在 GPU 上執行 FP16 推理，但 batch_tokens 保持 int64（long）
            results = self.model_esm(batch_tokens, repr_layers=[33], return_contacts=True)

        # 提取第 33 層的 token 表示
        token_representations = results["representations"][33]

        # 確保數據回到 CPU 並轉回 FP32，以便 NumPy 處理
        tensor_data = token_representations.detach().cpu().to(torch.float32)
        token_representations = np.array(tensor_data.tolist())  # 轉回 NumPy 格式

        # 去掉開頭和結尾的特殊 token (CLS 和 EOS)
        token_representations = token_representations[0][1:-1, :]

        # 返回嵌入的總和
        return token_representations.sum(axis=0)

    """def esmInfer(self, seq):
        '''
        T5為每條序列生成的1x1280 or 1x2560的embeddings特徵 回傳為dataframe
        :param seq: peptide序列
        :param block_size:block_size
        :return: dataframe內涵embeddings
        '''
        batch_converter = self.alphabet.get_batch_converter()
        self.model_esm.eval()  # 禁用 dropout 以获得确定性结果

        # 准备数据
        data = [("tmp", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        # 选择设备（GPU 优先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("esm")
        # 将 batch_tokens 转移到 GPU
        batch_tokens = batch_tokens.to(device)
        self.model_esm.to(device)

        # 确保 batch_tokens 不会超过指定的 block size
        if batch_tokens.size(1) > self.block_size:
            batch_tokens = batch_tokens[:, :self.block_size]  # 截断过长的序列
        else:
            # 填充较短的序列，使其长度达到 block_size
            padding = torch.full((batch_tokens.size(0), self.block_size - batch_tokens.size(1)), self.alphabet.padding_idx).to(device)
            batch_tokens = torch.cat((batch_tokens, padding), dim=1)

        # 确保没有超过 block_size 的 token
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        with torch.no_grad():
            # 在 GPU 上运行模型推理
            results = self.model_esm(batch_tokens, repr_layers=[33], return_contacts=True)

        # 提取第 33 层的 token 表示
        token_representations = results["representations"][33]
        tensor_data = token_representations.detach().cpu()
        token_representations = np.array(tensor_data.tolist())  # 转回 CPU 以便后续处理

        # 去掉开头和结尾的特殊 token (CLS 和 EOS)
        token_representations = token_representations[0][1:-1, :]

        # 返回嵌入的总和
        return token_representations.sum(axis=0)"""

    def addRow(self, allPepEmbeddingsDf, name, singlePepEmbeddingsDf, y):
        '''
        把單一序列生成的embeddings dataframe(singlePepEmbeddingsDf)集合在同一dataframe
        '''
        new_row = list(singlePepEmbeddingsDf) + [y]
        allPepEmbeddingsDf.loc[name] = new_row
        return allPepEmbeddingsDf

    def T5Esm(self, dataDict, esmAllPepEmbeddingsDf, t5AllPepEmbeddingsDf, esmModel, t5Model, t5OrEsm):
        '''
        呼叫T5 ESM生成後並用addRow合併輸出 trainDF & indpDF
        :param dataDict: 序列Df
        :param esmAllPepEmbeddingsDf: 空的esmDf
        :param t5AllPepEmbeddingsDf: 空的t5Df
        :param esmModel: esmModel名稱
        :param t5Model: t5Model名稱
        :return:
        '''
        if t5OrEsm == "t5":
            self.model_t5 = T5EncoderModel.from_pretrained(t5Model)  # prot_t5_xl_uniref50
            self.tokenizer = T5Tokenizer.from_pretrained(t5Model, do_lower_case=False)

            self.model_t5.to("cuda")
            #生成Neg peptide word embeddings
            for name, seq in dataDict[0].items():
                t5AllPepEmbeddingsDf = self.addRow(t5AllPepEmbeddingsDf, name, self.t5Infer(seq), 0)
            # 生成Pos peptide word embeddings
            for name, seq in dataDict[1].items():
                t5AllPepEmbeddingsDf = self.addRow(t5AllPepEmbeddingsDf, name, self.t5Infer(seq), 1)

            #清除ＧＰＵ顯存
            del self.model_t5
            gc.collect()
            torch.cuda.empty_cache()

            return t5AllPepEmbeddingsDf

        if t5OrEsm == "esm":
            self.model_esm, self.alphabet = torch.hub.load("facebookresearch/esm:main", esmModel)
            self.model_esm.to("cuda")
            # 生成Neg peptide word embeddings
            for name, seq in dataDict[0].items():
                esmAllPepEmbeddingsDf = self.addRow(esmAllPepEmbeddingsDf, name, self.esmInfer(seq), 0)
            # 生成Pos peptide word embeddings
            for name, seq in dataDict[1].items():
                esmAllPepEmbeddingsDf = self.addRow(esmAllPepEmbeddingsDf, name, self.esmInfer(seq), 1)

            # 清除ＧＰＵ顯存
            del self.model_esm
            gc.collect()
            torch.cuda.empty_cache()

            return esmAllPepEmbeddingsDf





