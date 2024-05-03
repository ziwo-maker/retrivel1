import json

import torch
#使用不同模型，选择对当前对话有用的历史。代码对应实验部分第一个实验
from sentence_transformers import SentenceTransformer
from utils import get_best_score,compute_mrr,compute_precision_recall
import os
from model_loade import Model
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


chat_path='/home/server/GX/Meta-Llama-3-8B-Instruct/'
#chat_path='/home/server/GX/gemma-7b/'


count=0
args={'temperature':0.1,'top_p':0.7,'max_length':1024}
write_path='data/Qrecc/test/qrecc_test_seletor_noweight.jsonl'
def main():
    
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    data_all=[];
    with open('data/Qrecc/test/Qrecc_test_mistral.jsonl','r') as f:
        for _ in f:
            data_all.append(json.loads(_))
            if(len(data_all)==1000):
                break;

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
                data_dict[int(i/2)]['his']=history[i:i+2]
            data_dict_sort=sorted(data_dict,key=lambda x: x["score"], reverse=True)
            his_turn=min(2,len(history)/2)
            his_turn=max(1,his_turn)

            chat_hist=[]
            for _ in data_dict_sort[:his_turn]:
                chat_hist+=_['his']
            with open(write_path,'a') as f:
                f.write(json.dumps({'question':question,'history':chat_hist,'stand_answer':data_['Answer']}))
                f.write('\n')


def select_selector():
    model_name='./save_noweight'
    tokenizer = BertTokenizer.from_pretrained('/home/server/GX/bert-base-uncased/')
    config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= 0.0)
    # def train(model, train_data, val_data, learning_rate, epochs):
    model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')
    data_all=[];
    # with open('data/TopiOCQA/dev/topiocqa_dev_seletor.jsonl','r') as f:
    #     for _ in f:
    #         data_all.append(json.loads(_))
            
    with open('data/Qrecc/test/qrecc_test.json','r') as f:
        
        data_all=json.load(f)
    count1=0
    # for ij,data_ in enumerate(data_all):
        
    #     history=data_['all_history']
    #     question=data_['question']
    #     pre_history=[];
    #     for i in range(0,len(history),2):
    #         if(len(history)<=2):
    #             continue;
    #         sentence=question+'[SEP]'+history[i]+' '+history[i+1]
    #         inputs=tokenizer(sentence,return_tensors="pt").to('cuda')
    #         output=model_bert(**inputs);
    #         y_pred = output[0].argmax(dim=1)
    #         if(y_pred.item()==1):
    #             pre_history+=history[i:i+2]
    #     if(pre_history==data_['history']):
    #         count1+=1;
    for ij,data_ in enumerate(data_all):
        
        history=data_['Context']
        if(len(history)%2==1):
            continue
        question=data_['Question']
        pre_history=[];
        for i in range(0,len(history),2):
            if(len(history)<=2):
                continue;
            sentence=question+'[SEP]'+history[i]+' '+history[i+1]
            inputs=tokenizer(sentence,return_tensors="pt").to('cuda')
            output=model_bert(**inputs);
            y_pred = output[0].argmax(dim=1)
            if(y_pred.item()==1):
                pre_history+=history[i:i+2]
       
    
        with open(write_path,'a') as f:
            f.write(json.dumps({'question':question,'history':pre_history,'stand_answer':data_['Answer'],'all_history':history}))
            f.write('\n')

def chat_LLM():
    data_all=[]
    with open(write_path,'r') as f:
        for _ in f:
            data_all.append(json.loads(_))

    chat_model=Model(chat_path,type='llama3')

    write_path2='data/Qrecc/test/Qrecc_test_selecotr_llama3.jsonl'
    for data_ in data_all:
        answer=chat_model.Chat(data_['question'],data_['history'],**args)
        data_['answer']=answer

        with open(write_path2,'a',) as f:
            f.write(json.dumps(data_))
            f.write('\n')
 
    
chat_LLM()


