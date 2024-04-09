import json
from judge_score import get_bert_score,get_ppl
from utils import get_best_score,get_bleu_score,get_meteor_score


def main():
    read_file=''
    write_file=''
    data_all=[]
    score1=0.0
    score2=0.0
    
    count=0
    with open('/content/retrivel1-main/data/TopiOCQA/dev/topiocqa_dev_score.jsonl') as f:
        for _ in f:
            data_all.append(json.loads(_))
    for data_ in data_all:
        stand_ans=data_['Answer'];
        ans_query=data_['ans_query']
        ans_all=data_['ans_all']

        score_rouge_query=get_best_score(stand_ans,ans_query)
        score_rouge_all=get_best_score(stand_ans,ans_all)

        # score_meteor_query=get_ppl(ans_query)
        # score_meteor_all=get_ppl(ans_all)
        score_meteor_query=get_bert_score(stand_ans,ans_query)
        score_meteor_all=get_bert_score(stand_ans,ans_all)
        with open('ppl_score_topiocqa.jsonl','a') as f:
          f.write(json.dumps({'answer_query':score_meteor_query,'answer_all':score_meteor_all}))
          f.write('\n')

        if(score_rouge_query[0]['rouge-1']['r']>score_rouge_all[0]['rouge-1']['r']):
          score1+=score_meteor_query['perplexities'][0]
          score2+=score_meteor_all['perplexities'][0]
          count+=1;
    print(score1/count)
    print(score2/count)

        
main()