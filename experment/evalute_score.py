#一次性返回recall，ppl，F1
#读取文件所有的测试样例，并全部返回。
import json

import json
from judge_score import get_bert_score,get_ppl,get_bleurt
from utils import get_best_score,get_bleu_score,get_meteor_score

def main():
    read_file='./data/TopiOCQA/dev/topiocqa_dev_all-MiniLM_mistral.jsonl'
    data_all=[];
    with open(read_file,'r') as f:
        for index,_ in enumerate(f):
            data_all.append(json.loads(_))
            if index==1000:
                break;
    recall_score=0
    answer_list=[];
    stand_answer_list=[]
    for data_ in data_all:
        answer=data_['answer'];
        stand_answer=data_['stand_answer']
        recall_score+=get_best_score(stand_answer,answer)[0]['rouge-1']['r']
        answer_list.append(answer)
        stand_answer_list.append(stand_answer)
    ppl_score=get_ppl(answer_list)['mean_perplexity']
    meteor_score=get_meteor_score(stand_answer_list,answer_list)['meteor']
    print('recall',recall_score/len(data_all))
    print('ppl',ppl_score)
    print('meteor_score',meteor_score)



#*three_answer*.jsonl
def cal_threetype():
    read_file='./data/TopiOCQA/dev/TopiOCQA_dev_three_answer_llama3.jsonl'
    data_all=[];
    with open(read_file,'r') as f:
        for index,_ in enumerate(f):
            data_all.append(json.loads(_))
            if index==1000:
                break;
    
    
    for index in range(0,3):
        recall_score=0
        answer_list=[];
        stand_answer_list=[]
        for data_ in data_all:
           
            if(len(data_['select_history'])<3):
                continue
            answer=data_['select_history'][index]['answer_his'];
            stand_answer=data_['Answer']
            
            tmp=get_best_score(stand_answer,answer)[0]['rouge-1']['r']
            recall_score+=tmp
            # if(tmp<0.1):
            #     continue
            answer_list.append(answer)
            stand_answer_list.append(stand_answer)
        ppl_score=get_ppl(answer_list)['mean_perplexity']
        meteor_score=get_meteor_score(stand_answer_list,answer_list)['meteor']
        print(index)
        print('recall',recall_score/len(stand_answer_list))
        print('ppl',ppl_score)
        print('meteor_score',meteor_score)

cal_threetype()