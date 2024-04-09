import json
import random
from model_loade import Model
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer
import os

from judge_score import Judge_score
from color import colo_histoty
model_name='./save1'
tokenizer = BertTokenizer.from_pretrained('/home/server/GX/bert-base-uncased/')
config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= 0.0)
# def train(model, train_data, val_data, learning_rate, epochs):
model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def main():
    chat_path='/home/server/GX/Mistral-7B-Instruct-v0.2/'
    write_path='./data/TopiOCQA/dev/topiocqa_dev_score.jsonl'
    data_path='./data/TopiOCQA/'
    judge=Judge_score()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    count=0;    
    with open('data/Qrecc/test/qrecc_test.json','r') as f:
        val_data=json.load(f)
    args={'do_sample':False,'temperature':0.1,'top_p':0.7}

    #chat_model=Model(chat_path)
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
  
        # ans_query=chat_model.Chat(question,history=pre_history,**args)
        # ans_all=chat_model.Chat(question,history=history,**args)
        # data_['ans_all']=ans_all
        # data_['ans_query']=ans_query
        # data_['bert_history']=pre_history
        # with open(write_path,'a') as fw:
        #     fw.write(json.dumps(data_))
        #     fw.write('\n')
main()

        
