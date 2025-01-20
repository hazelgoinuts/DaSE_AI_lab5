from transformers import BertTokenizer, BertModel

# 下载tokenizer和模型（会自动下载到 ~/.cache/huggingface/）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 保存到本地指定目录
save_path = './models/bert-base-uncased'
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)