import json


import json
import os
# 参考摘要

import torch

from model_loade import Model
from judge_score import Judge_score
from color import Color_
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# def Judeg(per_data,now_data):

#     prompt='''This is the current question and answer pair, query:{},answer:{}. Is the current answer based on the previous question and answer \
#         pair query:{},answer:{}. Please answer yes or no'''.format(now_data['Question'],now_data['Answer'],per_data['Question'],per_data['Answer'])
#     output=chat_model.Chat(prompt)
#     output=output.replace(prompt,'')
#     if('no' in output):
#         return False;
#     elif('yes' in output):
#         return True;
#     else:
#         raise ValueError('逻辑判断不正确')


def main():
    path='/home/server/GX/allmini'
    chat_path='/home/server/GX/chatglm3'
    file_path='TopiOCQA_train_with_siglehis_rouge.jsonl'
    chat_model=Model(chat_path)
    judeg=Judge_score(mode_path=path)
    count=0
    args={'do_sample':False,'temperature':0.1,'top_p':0.7}
    with open('./topiocqa_train.json','r') as f:
        data_all=json.load(f);
    count=0
    current_len=0;
    if os.path.exists(file_path):
        with open(file_path,'r') as fw:
            for index,_ in enumerate(fw):
                current_len=index
    data_all=data_all[current_len+1:]
    for index,data_ in enumerate(data_all):
        history=['','']+data_['Context']
        chat_len=len(history)
        question=data_['Question']
        answer=data_['Answer'] 
        select_his=[];
        # question=color_question(question=question,history=data_['Rationale'])  不用ration的效果会好一些
        #用rouge中的recall效果更佳，现在尝试使用余弦相似度  
        # path='/home/user/chatglm/ZhipuAI/chatglm3-6b/'
        if(chat_len>=2 and answer!='UNANSWERABLE'):

            # args['max_new_tokens']=len(answer)
            answer_query=chat_model.Chat(question,history=Color_.color_history(history),**args)
            cos_similary=judeg.cos_similarity(answer,[answer_query])
            score_query=judeg.get_rougescore(answer,answer_query)
            if score_query[0]['rouge-1']['r']>0.1:
            
                for i in range(chat_len-2,-2,-2):
                    # print(chat_len)
                    if(i!=chat_len):
                        pre_history=history[i:i+2]
                    else:
                        pre_history=[]
                    # for  j in range(0,i,2):
                    #     pre_history=history[j:j+2]+pre_history1
                    #     if(pre_history[0]==''):
                    #         pre_history=pre_history[2:]
                    answer_his=chat_model.Chat(question,history=Color_.color_history(pre_history),**args)

                    score_answer=judeg.get_rougescore(answer,answer_his)
                    query_cos_similary=judeg.cos_similarity(answer,[answer_his])
                    if(score_answer[0]['rouge-1']['r']>score_query[0]['rouge-1']['r']):
                        select_his.append({'history':pre_history,'score_answer':score_answer,'score_query':score_query,'answer_his':answer_his,'answer_all':answer_query,'query_cos_similary':float(query_cos_similary),'cos_similary':float(cos_similary)})
                        count+=1;
                #if(query_cos_similary>cos_similary):
                    # 
                # print(cos_similary)
        data_['select_history']=select_his;
        print('index',index,'count',count)
        
        with open(file_path,'a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
              
main()