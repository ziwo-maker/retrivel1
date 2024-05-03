import json

import torch
#使用不同模型，选择对当前对话有用的历史。代码对应实验部分第一个实验
from sentence_transformers import SentenceTransformer
from utils import get_best_score,compute_mrr,compute_precision_recall
import os
from model_loade import Model
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


chat_path='/home/server/GX/Meta-Llama-3-8B-Instruct/'
#chat_path='/home/server/GX/gemma-7b/'


count=0
args={'temperature':0.1,'top_p':0.7,'max_length':1024}
write_path='data/Qrecc/test/Qrecc_test_all-Mini.jsonl'


def chat_LLM():
    data_all=[]
    with open(write_path,'r') as f:
        for _ in f:
            data_all.append(json.loads(_))

    chat_model=Model(chat_path,type='llama3')

    write_path2='data/Qrecc/test/Qrecc_test_all-Mini_llama3.jsonl'
    for data_ in data_all:
        answer=chat_model.Chat(data_['question'],data_['history'],**args)
        data_['answer']=answer

        with open(write_path2,'a',) as f:
            f.write(json.dumps(data_))
            f.write('\n')
 
    
chat_LLM()


