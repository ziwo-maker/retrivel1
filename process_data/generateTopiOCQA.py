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
    path='/home/server/GX/allmini'
    chat_path='/home/server/GX/Mistral-7B-Instruct-v0.2/'
    file_path='./data/TopiOCQA/dev/TopiOCQA_dev_three_answer.jsonl'
    chat_model='Model(chat_path)'
    judeg=Judge_score(mode_path=path)
    count=0
    args={'do_sample':False,'temperature':0.1,'top_p':0.7,'max_length':'100'}
    with open('./topiocqa_dev.json','r') as f:
        data_all=json.load(f);
    count=0
    current_len=0;
    if os.path.exists(file_path):
        with open(file_path,'r') as fw:
            for index,_ in enumerate(fw):
                current_len=index
                count= index
    data_all=data_all[current_len+1:]
    for index,data_ in enumerate(data_all):
        history=data_['Context']
        chat_len=len(history)
        question=data_['Question']
        #组装所有的答案，并成list
        answer_list=[]
        #answer_list=[ans['Answer'] for ans in data_['Additional_answers'] if ans['Answer']!='UNANSWERABLE']
        answer_list.append(data_['Answer'])

        select_his=[];
        topic=data_['Topic']

        # path='/home/user/chatglm/ZhipuAI/chatglm3-6b/'
        if(chat_len>=2 and answer_list[-1]!='UNANSWERABLE'):
            #仅quer，all_history,part_history
            answer_query=chat_model.Chat(question,history=[],**args)
            select_his.append({'query':question,'history':[],'answer_his':answer_query})
            answer_all=chat_model.Chat(question,history=history,**args)
            select_his.append({'query':question,'history':history,'answer_his':answer_all})
            
            # print('score_query',score_query)
            for i in range(0,chat_len,2):
                pre_history=history[i:i+2]
                            
                answer_his=chat_model.Chat(question,history=pre_history,**args)

                # score_answer=get_best_score(answer_list,answer_his)
                # query_cos_similary=judeg.cos_similarity(answer_list[-1],[answer_his])
               
                select_his.append({'query':question,'history':pre_history,'answer_his':answer_his})
            
                #if(query_cos_similary>cos_similary):
                    # 
                # print(cos_similary)
        data_['select_history']=select_his;
        print('index',index,'count',count)
        
        with open(file_path,'a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
              
main()