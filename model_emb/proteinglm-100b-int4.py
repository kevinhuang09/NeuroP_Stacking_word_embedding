from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "biomap-research/proteinglm-100b-int4", trust_remote_code=True
)
print(getattr(config, "hidden_size", "請檢查 config 內容"))