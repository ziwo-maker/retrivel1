from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import BertConfig, BertForSequenceClassification
import json
import random

from utils import get_meteor_score
from judge_score import get_bleu,get_ppl
model_name='./save1'
# 1. 加载分词器和模型
# tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= 0.3)
# def train(model, train_data, val_data, learning_rate, epochs):
model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')
# 2. 准备输入数据
data_path='./data/TopiOCQA/'

val_data=[]
val_data_nge=[]

with open('TopiOCQA_train_no_answer_rouge.jsonl','r') as f:
    for _ in f:
        val_data.append(json.loads(_))
count=0;
for data_ in val_data:
    select_history=data_['select_history'];
    for sel in select_history:
        if(len(sel['history'])>2):
            count+=1;
print('count',count)
print('val_data',len(val_data)) 


hypothesis= "The cat is on the mat."

# 生成的句子
references= [["look at! one cat sat on the mat"]]

score=get_ppl(hypothesis)
print(score)
# with open('data/TopiOCQA/dev/topiocqa_dev.json','r') as f:
#     data_all=json.load(f)


# count=0;
# for data_ in data_all:
#     topic=data_['Topic']
#     history=data_['Context']
#     best_history=[]
#     score_list=[]
#     if(len(history)<4):
#         continue;
#     for i in range(0,len(history),2):
#         score=get_best_score([topic],history[i]+' '+history[i+1])
#         score_list.append({'pre_history':history[i:i+2],'score':score})
#     sorted_data = sorted(score_list, key=lambda x: x['score'][0]['rouge-1']['r'], reverse=True)
#     pos_index=0;
#     for index in range(0,len(sorted_data)):
#         if(sorted_data[index]['score'][0]['rouge-1']['r']>0.8):
#             with open('positive_dev.jsonl','a') as f:
#                 f.write(json.dumps({'Question':data_['Question'],
#                                     'history':sorted_data[index]['pre_history'],'label':1,
#                                     'score':sorted_data[index]['score']}))
#                 f.write('\n')
#             pos_index+=1;
#             count+=1;
#     for index in range(pos_index,min(len(sorted_data),pos_index*2)):
#         if(sorted_data[index]['score'][0]['rouge-1']['r']<0.5):
#             with open('negivate_dev.jsonl','a') as f:
#                 f.write(json.dumps({'Question':data_['Question'],
#                                     'history':sorted_data[index]['pre_history'],'label':0,
#                                     'score':sorted_data[index]['score']}))
#                 f.write('\n')
           
# print(count)

