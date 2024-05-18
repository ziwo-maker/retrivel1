import json
import random
from model_loade import Model
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer
import os
from utils import draw_plt,calculate_metrics
from judge_score import Judge_score
from color import colo_histoty
model_name='/home/server/GX/save_noweight/'
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

from dataloader import Dataset
import torch
from torch.utils.data import  DataLoader

from dataloader import Dataset


def ave_f1():
    val_data=[]
    with open(f'positive_dev.jsonl','r') as f:
        for _ in f:
            val_data.append(json.loads(_))

    val_data_nge=[]
    positive_len=len(val_data)
    with open(f'negivate_dev.jsonl','r') as f:
        for index,_ in enumerate(f):
            val_data_nge.append(json.loads(_))
    random.shuffle(val_data_nge)
    #val_data+=val_data_nge[:positive_len]
    val_data+=val_data_nge
    
    batch_size=32
    
    val_data = Dataset(val_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,shuffle=True)
    total_acc_val = 0
    total_loss_val = 0
    # # 将模型置为评估模式
    model_bert.eval()
    # 不需要计算梯度
    rr=pp=ff=acc_1=0
    countt=0
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for val_batch in val_dataloader:
            countt+=1
            encoding=tokenizer(
                val_batch['sentence'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to('cuda')
            val_labels = val_batch['labels'].to('cuda')
            # 通过模型得到输出
            val_outputs = model_bert(**encoding,labels=val_labels)
            # 计算损失
            y_pred_label = val_outputs[1].argmax(dim=1)
            r,p,f,acc=calculate_metrics(val_labels.view(-1),y_pred_label)
            
            rr+=r
            pp+=p;
            ff+=f
            acc_1+=acc

            # 计算准确率

            total_acc_val += acc*batch_size
            
    # 输出验证集的损失和准确率
    print(rr/countt,pp/countt,ff/countt,acc_1/countt)        
ave_f1()