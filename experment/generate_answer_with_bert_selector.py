import json


import json
import os
# 参考摘要

import torch

from model_loade import Model
from judge_score import Judge_score
import copy
from utils import get_best_score






def main():
   
    chat_path='/home/server/GX/chatglm3/'
    file_path='topiocqa_dev_score_chatglm3.jsonl'
    chat_model=Model(mode_path=chat_path,type='chatglm')

    count=0
    args={'temperature':0.1,'top_p':0.7,'max_length':1024}
    data_all=[];
    with open('./data/TopiOCQA/dev/topiocqa_dev_score.jsonl','r') as f:
        for _ in f:
            data_all.append(json.loads(_))

    count=0
    current_len=0;
    if os.path.exists(file_path):
        with open(file_path,'r') as fw:
            for index,_ in enumerate(fw):
                current_len=index
                count= index
    data_all=data_all[current_len:]
    for index,data_ in enumerate(data_all):
        history=data_['Context']
        chat_len=len(history)
        question=data_['Question']
        question+=' Please answer in English'
        select_history=[]
        if(chat_len>=2):
            
            q_raw=chat_model.Chat(question,history=[],**args)

            q_all=chat_model.Chat(question,history=history,**args)

            q_party=chat_model.Chat(question,history=data_['bert_history'],**args)
            select_history.append({'history':[],'answer':q_raw})
            select_history.append({'history':history,'answer':q_all})
            select_history.append({'history':data_['bert_history'],'answer':q_party})
            with open(file_path,'a') as fw:
                fw.write(json.dumps({'question':question,'stand_answer':data_['Answer'],'select_history':select_history}))  
                fw.write('\n')  
                    
              
main()