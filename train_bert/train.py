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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name='/home/server/GX/bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(model_name)

from tqdm import tqdm
learning_rate=1e-5
epochs=4;
batch_size=24
# def train(model, train_data, val_data, learning_rate, epochs):
def train():
    # 处理数据
    # 通过Dataset类获取训练和验证集
    train_data=[];
    #加载正例和负例
    with open('TopiOCQA_dev_with_siglehis.jsonl','r') as f:
        for _ in f:
            train_data.append(json.loads(_))
    positive_len=len(train_data)
    with open('TopiOCQA_dev_with_siglehis_negetive.jsonl','r') as f:
        for index,_ in enumerate(f):
            train_data.append(json.loads(_))
            if(index==positive_len):
                break;
    random.shuffle(train_data)
    train_dataset = Dataset(train_data, model.tokenizer)
    

# 将训练集划分为训练集和验证集
    train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)





    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_batch in tqdm(train_dataloader):
            input_id = train_batch['input_ids'].squeeze(1).to(device)
            train_attention_mask = train_batch['attention_mask'].to(device)
            train_labels = train_batch['labels'].to(device)

            # 将模型置为训练模式
            model.train()
            # 通过模型得到输出
            outputs = model(input_id,train_attention_mask)
            # 计算损失
            batch_loss  = criterion(outputs, train_labels)
            total_loss_train += batch_loss.item()
            # 反向传播和参数更新
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # 计算准确率

        # 输出训练集的损失和准确率
        print(f"Epoch {epoch_num + 1}/{epochs}, Train Loss: {total_loss_train/len(train_dataloader)}, Train Accuracy: {total_acc_train/len(train_dataset)}")

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
            # total_acc_val = 0
            # total_loss_val = 0
            # # 将模型置为评估模式
            # model.eval()
            # # 不需要计算梯度
            # with torch.no_grad():
            #     # 循环获取数据集，并用训练好的模型进行验证
            #     for val_batch in val_dataloader:
            #         val_input_ids = val_batch['input_ids'].to(device)
            #         val_attention_mask = val_batch['attention_mask'].to(device)
            #         val_labels = val_batch['labels'].to(device)

            #         # 通过模型得到输出
            #         val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
            #         # 计算损失
            #         val_loss = criterion(val_outputs.logits, val_labels)
            #         total_loss_val += val_loss.item()
            #         # 计算准确率
            #         total_acc_val += (val_outputs.logits.argmax(dim=1) == val_labels).sum().item()

            # # 输出验证集的损失和准确率
            # print(f"Epoch {epoch_num + 1}/{epochs}, Validation Loss: {total_loss_val/len(val_dataloader)}, Validation Accuracy: {total_acc_val/len(val_dataloader)}")
train()