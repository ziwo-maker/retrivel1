import json
from judge_score import get_bert_score,get_ppl,get_bleurt
from utils import get_best_score,get_bleu_score,get_meteor_score

#用于计算测评分数。当前计算 ppl bert_score,recall.bleurt



def main():

    data_all=[]
    score1=0.0
    score2=0.0
    score3=0.0
    score4=0.0
    count=0
    count1=0
    count2=0
    with open('data/TopiOCQA/dev/TopiOCQA_dev_three_answer_mistral.jsonl') as f:
        for _ in f:
          try:
            data_all.append(json.loads(_))
          except Exception as e:
            pass;
    for data_ in data_all:
        stand_ans=data_['Answer'];
       
        select_history=data_['select_history'];
        if(len(select_history)<2 or len(select_history[-1]['history'])==0):
          continue                                                                        
        raw_query=select_history[0]['answer_his']
        all_query=select_history[1]['answer_his']
        score_rouge_raw=get_best_score(stand_ans,raw_query)[0]['rouge-1']['r']
        score_rouge_all=get_best_score(stand_ans,all_query)[0]['rouge-1']['r']
        score_rouge_party=get_best_score(stand_ans,select_history[-1]['answer_his'])[0]['rouge-1']['r']
        max_score=0.0
        for i in range(2,len(select_history)-1):
          score_tmp=get_best_score(stand_ans,select_history[i]['answer_his'])
          score_tmp=score_tmp[0]['rouge-1']['r']
          max_score=max(score_tmp,max_score)

        if(score_rouge_party<0.2):
            continue;
        if(score_rouge_all>score_rouge_party):
           count1+=1
          #  if(len(select_history[-1]['history'])==0):
          #     print(len(select_history[-1]['history']))
        count+=1;
        score1+=score_rouge_raw
        score2+=score_rouge_all
        score3+=max_score
        score4+=score_rouge_party
    print(score1/count)
    print(score2/count)
    print(score3/count)
    print(score4/count)
    print(count1)








def main2():

    data_all=[]
    score1=0.0
    score2=0.0
    score3=0.0
    score4=0.0
    count=0
    count1=0
    with open('/content/drive/MyDrive/retrivel1-main/data/TopiOCQA/dev/TopiOCQA_dev_three_answer_mistral_NEW.jsonl') as f:
        for _ in f:
          try:
            data_all.append(json.loads(_))
          except Exception as e:
            pass;
    stand_ans_list=[]
    raw_query_list=[]
    all_query_list=[]
    party_list=[]
    for data_ in data_all:
        select_history=data_['select_history'];
        if(len(select_history)<2 or len(select_history[-1]['history'])==0):
          continue
        stand_ans_list.append(data_['Answer'])
        
        
        raw_query_list.append(select_history[0]['answer_his'])
        all_query_list.append(select_history[1]['answer_his'])
        party_list.append(select_history[-1]['answer_his'])
    # score1=get_ppl(raw_query_list)
    # score2=get_ppl(all_query_list)
    # score3=get_ppl(party_list)

    score1=get_meteor_score(stand_ans_list,raw_query_list)
    score2=get_meteor_score(stand_ans_list,all_query_list)
    score3=get_meteor_score(stand_ans_list,party_list)
    print(score1)
    print(score2)
    print(score3)
    # print(sum(score1['meteor'])/len(raw_query_list))
    # print(sum(score2['meteor'])/len(all_query_list))
    # print(sum(score3['meteor'])/len(party_list))


    # print(sum(score1['f1'])/len(raw_query_list))
    # print(sum(score2['f1'])/len(all_query_list))
    # print(sum(score3['f1'])/len(party_list))


main()