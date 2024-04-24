import json
from judge_score import get_bert_score,get_ppl
from utils import get_best_score,get_bleu_score,get_meteor_score

#用于计算测评分数。当前计算 ppl bert_score,recall.bleurt

def main():

    data_all=[]
    score1=0.0
    score2=0.0
    
    count=0
    with open('/content/drive/MyDrive/retrivel1-main/data/TopiOCQA/dev/topiocqa_dev_score.jsonl') as f:
        for _ in f:
            data_all.append(json.loads(_))
    for data_ in data_all:
        stand_ans=data_['Answer'];
       
        select_history=data_['select_history'];
        raw_query=select_history[0]['answer_his']
        all_query=select_history[1]['answer_his']

        score_rouge_raw=get_best_score(stand_ans,raw_query)
        score_rouge_all=get_best_score(stand_ans,all_query)
        max_score=0.0
        for i in range(2,len(select_history)):
          score_tmp=get_best_score(stand_ans,select_history[i]['answer_his'])
          score_tmp=score_tmp[0]['rouge-1']['r']
          max_score=max(score_tmp,max_score)
        
        # score_meteor_query=get_ppl(ans_query)
        # score_meteor_all=get_ppl(ans_all)
        # score_meteor_query=get_meteor_score(stand_ans,ans_query)
        # score_meteor_all=get_meteor_score(stand_ans,ans_all)
        
        # with open('ppl_score_topiocqa.jsonl','a') as f:
        #   f.write(json.dumps({'answer_query':score_meteor_query,'answer_all':score_meteor_all}))
        #   f.write('\n')

        # if(score_rouge_query[0]['rouge-1']['r']>score_rouge_all[0]['rouge-1']['r']):
        #   score1+=score_meteor_query['meteor']
        #   score2+=score_meteor_all['meteor']
        #   count+=1;
        print(score1/count)
        print(score2/count)

        
main()