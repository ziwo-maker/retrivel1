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
    file_path='./data/Qrecc/trainqrecc_train_no_answer_rouge.jsonl'
    chat_model=Model(chat_path)
    judeg=Judge_score(mode_path=path)
    count=0
    args={'temperature':0.1,'top_p':0.7,'max_length':1024}
    with open('/home/server/GX/retrivel1/data/Qrecc/train/qrecc_train.json','r') as f:
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
        
        # question=color_question(question=question,history=data_['Rationale'])  不用ration的效果会好一些
        #用rouge中的recall效果更佳，现在尝试使用余弦相似度  
     
        if(chat_len>=2 and answer_list[-1]!='UNANSWERABLE'):
            best_history=[]
            ans_score_list=[]
            if(chat_len%2==1):
                chat_len-=1;
            for i in range(0,chat_len,2):
                answer_query=chat_model.Chat(question,history=history[i:i+2],**args)
                score_query=get_best_score(answer_list,answer_query)
                ans_score_list.append({'answer':answer_query,'score':score_query,'index':i})

            
            
            sorted_ans_score_list = sorted(ans_score_list, key=lambda x: x['score'][0]['rouge-1']['r'], reverse=True)
            best_ans=sorted_ans_score_list[0]
            for best_ans in sorted_ans_score_list:
                select_his.append({'history':history[best_ans['index']:best_ans['index']+2],'score_answer':'score_answer',\
                                            'score_query':best_ans['score'],'answer_his':best_ans['answer']})


        data_['select_history']=select_his;
        print('index',index,'count',count)
        
        with open(file_path,'a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
              
main()