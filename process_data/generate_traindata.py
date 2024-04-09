import json
import os

# query history,label
def main():
    #generate positive
    read_file_path='./TopiOCQA_train_with_siglehis_rouge.jsonl'
    write_file_path='./TopiOCQA_train_with_siglehis.jsonl'
    if os.path.exists(write_file_path):
        os.remove(write_file_path)
        print(f"File '{write_file_path}' deleted.")
    data_all=[];
    with open(read_file_path,'r') as f:
        for _ in f:
            data_all.append(json.loads(_))
    for data_ in data_all:
        if(data_['select_history']==[]):
            continue;
        query=data_['Question']
        for turn  in data_['select_history']:
            history=turn['history']
            if(history[0]!=''):
   
                with open(write_file_path,'a') as fw:
                    fw.write(json.dumps({'Question':query,'history':history,'label':1}))
                    fw.write('\n')
def get_postive(query,data_with_postive):
    history=[]
    for _ in data_with_postive:
        if(query==_['Question']):
            history+=_['history']
    return history
def generate_negtaive():
    read_file_path='./TopiOCQA_train_with_siglehis_rouge.jsonl'
    write_file_path='./TopiOCQA_train_with_siglehis.jsonl'
    write_file_path2='./TopiOCQA_train_with_siglehis_negetive.jsonl'
    data_all=[];
    data_with_postive=[];
    if os.path.exists(write_file_path2):
        os.remove(write_file_path2)
        print(f"File '{write_file_path2}' deleted.")
    with open(read_file_path,'r') as f:
        for _ in f:
            data_all.append(json.loads(_))
    with open(write_file_path,'r') as f:
        for _ in f:
            data_with_postive.append(json.loads(_))
    count=0
    for data_ in data_all:
        query=data_['Question']
        postive_his=get_postive(query,data_with_postive)
        all_history=data_['Context']
        print(count)
        if(count>=30000):
            break
        for i in range(0,len(all_history),2):
            if(all_history[i] not in postive_his and all_history[i]!=''):
                count+=1;
                
                with open(write_file_path2,'a') as fw:
                    fw.write(json.dumps({'Question':query,'history':all_history[i:i+2],'Conversation_no':data_['Conversation_no'],'Turn_no':data_['Turn_no'],'label':0}))
                    fw.write("\n")    

    
generate_negtaive()