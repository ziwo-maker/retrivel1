import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class Dataset(Dataset):
    def __init__(self, df, max_length=10):
 
        
        self.texts = [(entry['Question'], entry['history']) for entry in df]
        self.labels = [entry['label'] for entry in df]
        self.max_length = max_length
        self.count=0;
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text_pair = self.texts[idx]
        label = self.labels[idx]
        setence=text_pair[0]+'[SEP]'+text_pair[1][0]+' '+text_pair[1][1]
        self.max_length=max(self.max_length,len(setence))
        if(len(setence)>=512):
            self.count+=1;
        return {
   
            'labels': torch.tensor(label, dtype=torch.long),
            'sentence':setence
        }

    def get_labels(self):
        return self.labels

# 示例用法
# 假设你有一个DataFrame df 包含 'query', 'sentence', 'label' 列
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = CustomDataset(df, tokenizer)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# 之后你就可以用dataloader来训练你的BERT模型
