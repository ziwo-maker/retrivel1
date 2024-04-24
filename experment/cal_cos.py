import json

import torch
#调用初始数据集即可
from sentence_transformers import SentenceTransformer
from utils import get_best_score,compute_mrr,compute_precision_recall
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def main():
    
    with open('/content/topiocqa_dev.json','r') as f:
        data_all=json.load(f)
    count1=0
    sum_mrr=sum_pre=sum_reacall=0
    for data_ in data_all:

        history=data_['Context']
        question=data_['Question']
        topic=data_['Topic']
        chat_len=len(history)
        
        if(chat_len>=2):
            pre_history_list=[]
            pre_history_list.append(question)
            data_dict=[]
            stand_score=[]
            count=0;
            for i in range(0,chat_len,2):
                pre_history_list.append(history[i]+' '+history[i+1])
                score=get_best_score([topic],history[i]+' '+history[i+1])
                if(score[0]['rouge-1']['r']>0.7):
                    stand_score.append(1)
                    count+=1;
                else:
                    stand_score.append(0)
            embeddings = model.encode(pre_history_list)  
            question_embedding=torch.tensor(embeddings[0])
            embeddings=torch.tensor(embeddings[1:])
            similarity_scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)

            for setn, score,index in zip(pre_history_list[1:],similarity_scores,stand_score):
                data_dict.append({'history':setn,'score':score,'label':index})
            data_dict_sort=sorted(data_dict,key=lambda x: x["score"], reverse=True)
            
            if(count==0):
                continue;
            precision, recall = compute_precision_recall(data_dict_sort)
            if precision==0 or recall ==0:
                continue
            sum_mrr+=compute_mrr(data_dict_sort)
            
            count1+=1;
            sum_pre+=precision;
            sum_reacall+=recall
            # print(precision,recall)
    print('main')     
    print(sum_mrr/count1,sum_pre/count1,sum_reacall/count1,)
def cal_cos():
    data_all=[]
    
    with open('./data/TopiOCQA/dev/TopiOCQA_dev_three_answer.jsonl') as f:
      for _ in f:
        try:
          data_all.append(json.loads(_))
        except Exception as e:
          pass;
    count1=0
    sum_mrr=sum_pre=sum_reacall=0           
    for data_ in data_all:

        history=data_['Context']
        # question=data_['Question']
        topic=data_['Topic']
        chat_len=len(history)
        stand_score=[]
        
        if(chat_len>=2 and len(data_['select_history'])>=2):
            select_history=data_['select_history']
            pre_history_list=[]
            pre_history_list.append(data_["Answer"])
            count=0;
            for i in range(0,len(history),2): 
                score=get_best_score([topic],history[i]+' '+history[i+1])
                if(score[0]['rouge-1']['r']>0.7):
                    stand_score.append(history[i])
                    stand_score.append(history[i+1])
                    count+=1;
            Cotext_histort=[]
            for i in range(2,len(select_history)):
                pre_history_list.append(select_history[i]['answer_his'])
                Cotext_histort.append(select_history[i]['history'])
            embeddings = model.encode(pre_history_list)  
            question_embedding=torch.tensor(embeddings[0])
            embeddings=torch.tensor(embeddings[1:])
            similarity_scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)
            data_dict=[]
            for his,score in zip(Cotext_histort, similarity_scores):
                if(his[0] in stand_score):
                    label=1;
                else :
                    label=0;
                data_dict.append({'history':his,'score':score,'label':label})
            data_dict_sort=sorted(data_dict,key=lambda x: x["score"], reverse=True)
            
            if len(stand_score)==0:
                continue
            precision, recall = compute_precision_recall(data_dict_sort)
            
            sum_mrr+=compute_mrr(data_dict_sort)
            count1+=1;
            
            sum_pre+=precision
            sum_reacall+=recall
    print('cal_cos')
    print(sum_mrr/count1,sum_pre/count1,sum_reacall/count1)

def cal_recall():
    data_all=[]
    
    with open('') as f:
      for _ in f:
        try:
          data_all.append(json.loads(_))
        except Exception as e:
          pass;
    count1=0
    sum_mrr=sum_pre=sum_reacall=0           
    for data_ in data_all:

        history=data_['Context']
        # question=data_['Question']
        topic=data_['Topic']
        chat_len=len(history)
        stand_score=[]
        
        if(chat_len>=2 and len(data_['select_history'])>=2):
            select_history=data_['select_history']
            stand_answer=data_['Answer']
            pre_history_list=[]
            pre_history_list.append(data_["Answer"])
            count=0;
            for i in range(0,len(history),2):
                score=get_best_score([topic],history[i]+' '+history[i+1])
                if(score[0]['rouge-1']['r']>0.3):
                    stand_score.append(history[i])
                    stand_score.append(history[i+1])
                    count+=1;
            data_dict=[]

            for i in range(2,len(select_history)):
                
                score=get_best_score([stand_answer],select_history[i]['answer_his'])
                if(select_history[i]['history'][0] in stand_score):
                    label=1;
                else :
                    label=0;
                data_dict.append({'history':select_history[i]['history'],'score':score[0]['rouge-1']['r'],'label':label})
            
            data_dict_sort=sorted(data_dict,key=lambda x: x["score"], reverse=True)
            
           
            precision, recall = compute_precision_recall(data_dict_sort)
            if len(stand_score)==0:
                continue
            # if precision==0 and recall==0:
            #    print(stand_answer)
            #    print(data_dict_sort)
            #sum_mrr+=compute_mrr(data_dict_sort)
            
            count1+=1;
            
            sum_pre+=precision
            sum_reacall+=recall
    print('cal_recall')
    print(sum_mrr/count1,sum_pre/count1,sum_reacall/count1)






cal_recall()