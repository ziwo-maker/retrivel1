import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [(entry['Question'], entry['history']) for entry in df]
        self.labels = [entry['label'] for entry in df]
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text_pair = self.texts[idx]
        label = self.labels[idx]
        setence=text_pair[0]+'[SEP]'+text_pair[1][0]+' '+text_pair[1][1]
        encoding = self.tokenizer(
            setence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_labels(self):
        return self.labels

# 示例用法
# 假设你有一个DataFrame df 包含 'query', 'sentence', 'label' 列
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = CustomDataset(df, tokenizer)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# 之后你就可以用dataloader来训练你的BERT模型
