import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from dataloader import Dataset
from model_load import Model
from transformers import BertTokenizer, BertModel
import torch.nn as nn 
import json
from torch.optim import Adam
from sklearn.model_selection import train_test_split
# 创建数据加载器
import random
import os
from transformers import BertConfig, BertForSequenceClassification
import matplotlib.pyplot as plt
from transformers import AdamW
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name='/home/server/GX/bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(model_name)

from tqdm import tqdm
learning_rate=1e-5
epochs=2;
batch_size=32
weight_decay = 1e-2

def draw_plt(train_loss=[],dev_loss=[]):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, dev_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
def train():
    # 处理数据
    # 通过Dataset类获取训练和验证集
    config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob= 0.3)
# def train(model, train_data, val_data, learning_rate, epochs):
    model_bert=BertForSequenceClassification.from_pretrained(model_name, config=config).to('cuda')
    train_data=[];
    val_data=[];
    data_path='./data/TopiOCQA/'
    #从train和dev加载正例和负例
    with open(f'{data_path}train/TopiOCQA_train_with_siglehis.jsonl','r') as f:
        for _ in f:
            train_data.append(json.loads(_))
    train_data=train_data[:6000]
    positive_len=len(train_data)
    with open(f'{data_path}train/TopiOCQA_train_with_siglehis_negetive.jsonl','r') as f:
        for index,_ in enumerate(f):
            train_data.append(json.loads(_))
            if(index==positive_len):
                break;
    with open(f'{data_path}dev/TopiOCQA_dev_with_siglehis.jsonl','r') as f:
        for _ in f:
            val_data.append(json.loads(_))


    positive_len=len(val_data)
    with open(f'{data_path}dev/TopiOCQA_dev_with_siglehis_negetive.jsonl','r') as f:
        for index,_ in enumerate(f):
            val_data.append(json.loads(_))
            if(index==positive_len):
                break;
    
    random.shuffle(train_data)
    train_data = Dataset(train_data, model.tokenizer)
    
    random.shuffle(val_data)
    val_data = Dataset(val_data, model.tokenizer)
    print()
# 将训练集划分为训练集和验证集
    
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)




    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model_bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model_bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
                                                     
    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 开始进入训练循环
    train_loss_history=[]
    dev_history=[]
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        epoch_acc=0;
        for train_batch in tqdm(train_dataloader):
            #encoding = train_batch['encoding'].to(device)
            encoding=tokenizer(
                train_batch['sentence'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to('cuda')
            #train_attention_mask = train_batch['attention_mask'].to(device)
            train_labels = train_batch['labels'].to(device)

            # # 将模型置为训练模式
            # model.bert.train()
            optimizer.zero_grad()
            # 通过模型得到输出
            outputs =model_bert(**encoding)
            # 计算损失
            y_pred_prob = outputs[0]    
            y_pred_label  = y_pred_prob.argmax(dim=1)
            batch_loss  = criterion(y_pred_prob.view(-1, 2), train_labels.view(-1))
            total_loss_train += batch_loss.item()
            train_loss_history.append(batch_loss.item())
            # 反向传播和参数更新
            acc = ((y_pred_label == train_labels.view(-1)).sum()).item()
            epoch_acc+=acc
            batch_loss.backward()
            optimizer.step()
            # 计算准确率

        # 输出训练集的损失和准确率
        print(f"Epoch {epoch_num + 1}/{epochs}, Train Loss: {total_loss_train/len(train_dataloader)}, Train Accuracy: {epoch_acc/len(train_data)}")

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 将模型置为评估模式
        model.bert.eval()
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_batch in val_dataloader:

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
                acc = ((y_pred_label == val_labels.view(-1)).sum()).item()
                val_loss = criterion(val_outputs.logits, val_labels)
                total_loss_val += val_loss.item()
                # 计算准确率
                dev_history.append(val_loss.item())
                total_acc_val += acc

        # 输出验证集的损失和准确率
        draw_plt(train_loss_history,dev_history)
        print(f"Epoch {epoch_num + 1}/{epochs}, Validation Loss: {total_loss_val/len(val_dataloader)}, Validation Accuracy: {total_acc_val/len(val_dataloader)}")
train()