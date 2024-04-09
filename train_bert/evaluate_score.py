import json
from judge_score import get_bert_score,get_bleu
from utils import get_best_score,get_bleu_score,get_meteor_score


def main():
    read_file=''
    write_file=''
    data_all=[]
    with open('data/TopiOCQA/dev/topiocqa_dev_score.jsonl') as f:
        for _ in f:
            data_all.append(json.loads(_))
    for data_ in data_all:
        stand_ans=data_['Answer'];
        ans_query=data_['ans_query']
        ans_all=data_['ans_all']
        score_rouge_query=get_best_score(stand_ans,ans_query)
        score_rouge_all=get_best_score(stand_ans,ans_all)
        
        print(score_rouge_all)
main()