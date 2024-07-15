import json

import torch
#调用初始数据集即可
from sentence_transformers import SentenceTransformer
from utils import get_best_score,compute_mrr,compute_precision_recall

#计算不同轮次和总轮次占比的分数变化趋势。
def cal_cos():
    data_all=[]
    
    with open('data/TopiOCQA/dev/TopiOCQA_dev_three_answer_qwen.jsonl') as f:
      for _ in f:
        try:
          data_all.append(json.loads(_))
        except Exception as e:
          pass;
    count1=[0]*12
    count2=[0]*12
    count3=[0]*12
    sum_mrr=sum_pre=sum_reacall=0           
    for data_ in data_all:
        history=data_['Context']
        # question=data_['Question']
        topic=data_['Topic']
        chat_len=len(history)
        stand_score=[]
        stand_answer=data_['Answer'];
        max_turn=0
        if(chat_len>=2 and len(data_['select_history'])>=2):
            select_history=data_['select_history']
            pre_history_list=[]
            pre_history_list.append(data_["Answer"])
            for i in range(0,len(history),2): 
                score=get_best_score([topic],history[i]+' '+history[i+1])


            tmp=len(history)/20.0
            max_turn=max(max_turn,len(history))
            score=get_best_score(data_['Answer'],data_['select_history'][2]['answer_his'])[0]['rouge-1']['r']
            score1=get_best_score(data_['Answer'],data_['select_history'][1]['answer_his'])[0]['rouge-1']['r']
            for i in range(0,12,1):
               
                  count1[i]+=score
                  count2[i]+=1
                  count3[i]+=score1
    for  i in range(len(count1)):
        if(count2[i]!=0):
            print(count1[i]/count2[i]-count3[i]/count2[i],)
    print(max_turn)    
    # print([x-y for x,y in zip(count1,count3)])
    #print(count3)

cal_cos()
