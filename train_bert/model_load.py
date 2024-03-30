import torch
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel


from torch import nn
class Model(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2,dropout=0.5):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_name).to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout).to('cuda')
        self.linear = nn.Linear(768, 2).to('cuda')
        self.relu = nn.ReLU().to('cuda')
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, 
                                      attention_mask=mask,
                                      return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer