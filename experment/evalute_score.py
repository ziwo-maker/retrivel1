#一次性返回recall，ppl，F1
#读取文件所有的测试样例，并全部返回。
import json

import json
from judge_score import get_bert_score,get_ppl,get_bleurt
from utils import get_best_score,get_bleu_score,get_meteor_score


def main():
    read_file='/content/drive/MyDrive/retrivel1-main/data/TopiOCQA/dev/topiocqa_dev_all-MiniLM_qwen.jsonl'
    data_all=[];
    with open(read_file,'r') as f:
        for index,_ in enumerate(f):
            data_all.append(json.loads(_))

    recall_score=0
    count=0
    answer_list=[];
    stand_answer_list=[]
    for data_ in data_all:
        answer=data_['answer'];
        stand_answer=data_['stand_answer']
        tmp=get_best_score(stand_answer,answer)[0]['rouge-1']['r']
        
        recall_score+=tmp
        
        if not isinstance(answer, str):
            print("answer 是一个字符串")
        answer_list.append(answer)
        stand_answer_list.append(stand_answer)
    print('recall',recall_score/len(answer_list))
    print(count)
    ppl_score=get_ppl(answer_list)['mean_perplexity']
    
    meteor_score=get_meteor_score(stand_answer_list,answer_list)['meteor']
    
    
    print('meteor_score',meteor_score)
    print('ppl',ppl_score)


#*three_answer*.jsonl
def cal_threetype():
    read_file='data/TopiOCQA/dev/TopiOCQA_dev_three_answer_mistral.jsonl'
    data_all=[];
    with open(read_file,'r') as f:
        for index,_ in enumerate(f):
            data_all.append(json.loads(_))
            if index==1000:
                break;
    
    #输出三组，分为是raw all party

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
            tmp1=get_best_score(stand_answer,data_['select_history'][1]['answer_his'])[0]['rouge-1']['r']
            tmp2=get_best_score(stand_answer,data_['select_history'][2]['answer_his'])[0]['rouge-1']['r']
            if(tmp2<0.15):
                continue
            recall_score+=tmp


            answer_list.append(answer)
            stand_answer_list.append(stand_answer)
        
        ppl_score=get_ppl(answer_list)['mean_perplexity']
        meteor_score=get_meteor_score(stand_answer_list,answer_list)['meteor']
        print(index)
        print('meteor_score',meteor_score)
        print('recall',recall_score/len(stand_answer_list))
        print('ppl',ppl_score)
        
main()
