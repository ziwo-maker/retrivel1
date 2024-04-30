import json
import random
from model_loade import Model
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer
import os

from judge_score import Judge_score
from color import colo_histoty
model_name='/home/server/GX/save1/'
tokenizer = BertTokenizer.from_pretrained('/home/server/GX/bert-base-uncased/')
config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= 0.0)
# def train(model, train_data, val_data, learning_rate, epochs):
model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')

def main():
    chat_path='/home/server/GX/Mistral-7B-Instruct-v0.2/'
    write_path='data/TopiOCQA/dev/topiocqa_dev_seletor_mistral.jsonl'
    data_path='./data/TopiOCQA/'
    judge=Judge_score()

    count=0;    
      
    with open('./data/TopiOCQA/dev/topiocqa_dev.json','r') as f:
        val_data=json.load(f)
    args={'temperature':0.1,'top_p':0.7,'max_length':1024}
    val_data=val_data[1000:]
    chat_model=Model(chat_path)
    for index, data_ in enumerate(val_data):
        history=data_['Context']
        question=data_['Question']
        pre_history=[];
        for i in range(0,len(history),2):
            if(len(history)<=2):
                continue;
            sentence=question+'[SEP]'+history[i]+' '+history[i+1]
            inputs=tokenizer(sentence,return_tensors="pt").to('cuda')
            output=model_bert(**inputs,);
            y_pred = output[0].argmax(dim=1)
            if(y_pred.item()==1):
                pre_history+=history[i:i+2]
        if(len(pre_history)==0):
            continue;
  
        ans_query=chat_model.Chat(question,history=pre_history,**args)

        with open(write_path,'a') as f:
                f.write(json.dumps({'question':question,'history':pre_history,'stand_answer':data_['Answer'],'answer':ans_query}))
                f.write('\n')

main()

        
