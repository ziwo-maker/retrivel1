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






def main():

    chat_path='/home/server/GX/Mistral-7B-Instruct-v0.2/'
    #chat_path='/home/server/GX/gemma-7b/'
    file_path='./data/TopiOCQA/dev/TopiOCQA_dev_three_answer_every_turn.jsonl'

    chat_model=Model(chat_path)
    
    count=0
    args={'temperature':0.1,'top_p':0.7,'max_length':1024}
    with open('/home/server/GX/retrivel1/data/TopiOCQA/dev/topiocqa_dev.json','r') as f:
        data_all=json.load(f);
    count=0
    current_len=0;
    # if os.path.exists(file_path):
    #     with open(file_path,'r') as fw:
    #         for index,_ in enumerate(fw):
    #             current_len=index
    #             count= index
    
    # data_all=data_all[current_len:]

    for index,data_ in enumerate(data_all):
        history=data_['Context']
        chat_len=len(history)
        question=data_['Question']
        #组装所有的答案，并成list
        answer_list=[]


        answer_list.append(data_['Answer'])

        select_his=[];


        if(chat_len%2==1):
            continue
        # path='/home/user/chatglm/ZhipuAI/chatglm3-6b/'
        if(chat_len>=2 and answer_list[-1]!='UNANSWERABLE'):

            for i in range(0,len(history),2):
                answer_query=chat_model.Chat(question,history=history[i:i+2],**args)
                select_his.append({'query':question,'history':history[i:i+2],'answer_his':answer_query})
        data_['select_history']=select_his
        with open(file_path,'a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
#计算最佳分数的         
def main2():
    read_file='./data/Qrecc/test/Qrecc_test_three_answer_mistral_every_turn.jsonl'
    write_file='./data/Qrecc/test/Qrecc_test_three_answer_mistral.jsonl'
    data_all=[]
    with open(read_file,'r') as f :
        for _ in f:
            data_all.append(json.loads(_))
    count=0;
    sum_=0            
    for data_ in data_all:
        select_history=data_['select_history']
        stand_ans=data_['Answer']
        score_list=[]
        if(len(select_history)==0):
            continue
        for one_turn in select_history:
            one_turn['score']=get_best_score(stand_ans,one_turn['answer_his'])[0]['rouge-1']['r']
            score_list.append(one_turn)
        sorted_data = sorted(score_list, key=lambda x: x['score'], reverse=True)
        selected=[]
        if len(sorted_data)>1:
            best_score=sorted_data[0]['score']
            selected.append(sorted_data[0])
            for i in range(1,len(sorted_data)):
                if(best_score-sorted_data[i]['score']<=0.15):
                    selected.append(sorted_data[i])
        data_['select_history']=selected
        sum_+=len(selected)/len(select_history)
        count+=1;
        with open(write_file,'a') as f:
            f.write(json.dumps(data_))
            f.write('\n')
    print(sum_/count)
main()