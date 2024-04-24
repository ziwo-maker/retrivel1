import json

import torch
#使用不同模型，选择对当前对话有用的历史。
from sentence_transformers import SentenceTransformer
from utils import get_best_score,compute_mrr,compute_precision_recall
import os
from model_loade import Model



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
write_path='./data/TopiOCQA/dev/topiocqa_dev_bert.jsonl'
chat_path='/home/server/GX/Mistral-7B-Instruct-v0.2/'
#chat_path='/home/server/GX/gemma-7b/'
file_path='./data/Qrecc/test/Qrecc_test_three_answer_mistral_every_turn.jsonl'



count=0
args={'temperature':0.1,'top_p':0.7,'max_length':1024}
def main():
    write_path='./data/TopiOCQA/dev/topiocqa_dev_bert.jsonl'
    
    with open('./data/TopiOCQA/dev/topiocqa_dev.json','r') as f:
        data_all=json.load(f)

    count1=0

    for data_ in data_all:

        history=data_['Context']
        question=data_['Question']

        chat_len=len(history)
        
        if(chat_len>=2):
            pre_history_list=[]
            pre_history_list.append(question)
            data_dict=[]

            count=0;
            for i in range(0,chat_len,2):
                pre_history_list.append(history[i]+' '+history[i+1])
            embeddings = model.encode(pre_history_list)  
            question_embedding=torch.tensor(embeddings[0])
            embeddings=torch.tensor(embeddings[1:])
            similarity_scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)

            for setn, score in zip(pre_history_list[1:],similarity_scores):
                data_dict.append({'history':setn,'score':score})
            for i in range(0,len(history),2):
                data_dict[i/2]['his']=history[i,i+2]
            data_dict_sort=sorted(data_dict,key=lambda x: x["score"], reverse=True)
            his_turn=min(2,len(history)/2)
            his_turn=max(1,his_turn)
            print(his_turn)
            chat_hist=[]
            for _ in data_dict_sort[:his_turn]:
                chat_hist+=_['his']
            with open(write_path,'a') as f:
                f.write(json.dumps({'question':question,'history':chat_hist['history'],'stand_answer':data_['Answer']}))
                f.write('\n')


def chat_LLM():
    data_all=[]
    with open(write_path,'a') as f:
        for _ in f:
            data_all.append(json.loads(_))
    chat_model=Model(chat_path)
    answer_stand=[];
    answer_query=[]
    write_path2='./data/TopiOCQA/dev/topiocqa_dev_bert_chatmodel.jsonl'
    for data_ in data_all:
        answer=chat_model.Chat(data_['question'],data_['history'],**args)
        data_['answer']=answer
        with open(write_path2,'a',) as f:
            f.write(data_)
            f.write('\n')


    
main()


