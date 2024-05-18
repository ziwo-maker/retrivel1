import torch
from torch.utils.data import  DataLoader

from dataloader import Dataset


import torch.nn as nn 
import json
from torch.optim import Adam


import random

from transformers import BertConfig, BertForSequenceClassification,BertTokenizer

from transformers import AdamW
from utils import draw_plt,calculate_metrics
import logging
import os
from tqdm import tqdm
logging.basicConfig(level=logging.INFO,filename='train_log.log', # 设置日志级别为 DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
logger = logging.getLogger(__name__)
#参数配置

model_name='/home/server/GX/bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate=1e-6
epochs=1;
batch_size=32
weight_decay = 1e-5
drop_out=0.5
train_size=1000
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def train():
    # 处理数据
    # 通过Dataset类获取训练和验证集
    config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= drop_out)
# def train(model, train_data, val_data, learning_rate, epochs):
    model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')
    train_data=[];
    val_data=[];
    data_path='./data/TopiOCQA/'
    #从train和dev加载正例和负例
    with open(f'positive.jsonl','r') as f:
        for _ in f:
            train_data.append(json.loads(_))
    #train_data=train_data[:train_size]
    positive_len=int(len(train_data)*1)
    train_data_nge=[]
    
    with open(f'negivate.jsonl','r') as f:
        for index,_ in enumerate(f):
            train_data_nge.append(json.loads(_))
    random.shuffle(train_data_nge)
    #train_data+=train_data_nge[:positive_len]
    train_data+=train_data_nge
    random.shuffle(train_data)
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
    
  
    train_data = Dataset(train_data)
    val_data = Dataset(val_data)

# 将训练集划分为训练集和验证集
    
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,shuffle=True)


    tokenizer = BertTokenizer.from_pretrained(model_name)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model_bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model_bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
                                                     
    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger.info(f'''learng_rate:{learning_rate}.epoch:{epochs},batch_size:{batch_size},weight_decay:{weight_decay}.train_size:{train_size}''')
    # 开始进入训练循环
    model_bert.train()
    train_loss_history=[]
    dev_history=[]
    for epoch_num in range(epochs):
        # # 定义两个变量，用于存储训练集的准确率和损失
        # total_acc_train = 0
        # total_loss_train = 0
        # # 进度条函数tqdm
        # epoch_acc=0;
        # for train_batch in tqdm(train_dataloader):
            
        #     encoding=tokenizer(
        #         train_batch['sentence'],
        #         padding='max_length',
        #         truncation=True,
        #         max_length=512,
        #         return_tensors="pt",
        #         return_attention_mask=True
        #     ).to('cuda')
        #     #train_attention_mask = train_batch['attention_mask'].to(device)
        #     train_labels = train_batch['labels'].to(device)

        #     # 将模型置为训练模式
        #     #
        #     optimizer.zero_grad()
        #     # 通过模型得到输出
        #     outputs =model_bert(**encoding,labels=train_labels)
        #     # 计算损失
        #     y_pred_prob = outputs[1]    
        #     y_pred_label  = y_pred_prob.argmax(dim=1)
        #     batch_loss  = criterion(y_pred_prob.view(-1, 2), train_labels.view(-1))
        #     total_loss_train += batch_loss.item()
        #     train_loss_history.append(batch_loss.item())
        #     # 反向传播和参数更新
        #     r,p,f,acc=calculate_metrics(train_labels.view(-1),y_pred_label )
        #     epoch_acc+=acc*batch_size
        #     batch_loss.backward()
        #     optimizer.step()
        #     #print(batch_loss)
        #     # 计算准确率
        # print(f"r:{r},p:{p},f:{f},acc:{acc}")
        # # 输出训练集的损失和准确率
        # print(f"Epoch {epoch_num + 1}/{epochs}, Train Loss: {total_loss_train/len(train_dataloader)}, Train Accuracy: {epoch_acc/len(train_data)},Acc:{acc}")
        # logger.info(f"Epoch {epoch_num + 1}/{epochs}, Train Loss: {total_loss_train/len(train_dataloader)}, Train Accuracy: {epoch_acc/len(train_data)}")
        # logger.info(f"r:{r},p:{p},f:{f}")
        # # ------ 验证模型 -----------
        # # 定义两个变量，用于存储验证集的准确率和损失
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
                val_labels = val_batch['labels'].to(device)
                # 通过模型得到输出
                val_outputs = model_bert(**encoding,labels=val_labels)
                # 计算损失
                y_pred_label = val_outputs[1].argmax(dim=1)
                r,p,f,acc=calculate_metrics(val_labels.view(-1),y_pred_label)
                val_loss = criterion(val_outputs.logits, val_labels)
                rr+=r
                pp+=p;
                ff+=f
                acc_1+=acc
                total_loss_val += val_loss.item()
                # 计算准确率
                dev_history.append(val_loss.item())
                total_acc_val += acc*batch_size
                clea
        # 输出验证集的损失和准确率
        print(rr/countt,pp/countt,ff/countt,acc_1/countt)
        print(f"Epoch {epoch_num + 1}/{epochs}, Validation Loss: {total_loss_val/len(val_dataloader)}, Validation Accuracy: {total_acc_val/(len(val_dataloader)*batch_size)}")
        logger.info(f"Epoch {epoch_num + 1}/{epochs}, Validation Loss: {total_loss_val/len(val_dataloader)}, Validation Accuracy: {total_acc_val/(len(val_dataloader)*batch_size)}")
    # draw_plt(train_loss_history,dev_history)
    # model_bert.save_pretrained('./save_noweight/')
train()