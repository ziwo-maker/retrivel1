import json


import json
import os
# 参考摘要

import torch

from model_loade import Model
from judge_score import Judge_score
import copy
from utils import get_best_score
import os




#生成三种答案 raw，all， party

def main():
    path='/home/server/GX/allmini'
    chat_path='/home/server/GX/Qwen1.5-14B-Chat-GPTQ-Int8/'
    #chat_path='/home/server/GX/gemma-7b/'
    file_path='./data/TopiOCQA/dev/TopiOCQA_dev_three_answer_qwen14B.jsonl'

    chat_model=Model(mode_path=chat_path,type='qwen')
    
    count=0
    args={'temperature':0.1,'top_p':0.7,'max_length':1024}
    with open('./topiocqa_dev.json','r') as f:
        data_all=json.load(f);
    count=0



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
            stand_score=[]
            # print('score_query',score_query)
            for i in range(0,len(history),2):
                score=get_best_score(topic,history[i]+' '+history[i+1])
                if(score[0]['rouge-1']['r']>=0.3):
                    stand_score.append(history[i])
                    stand_score.append(history[i+1])

            answer_party=chat_model.Chat(question,history=stand_score,**args)
            select_his.append({'query':question,'history':stand_score,'answer_his':answer_party})
        data_['select_history']=select_his
        with open(file_path,'a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
              
main()