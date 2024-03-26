import json
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge
import json
import os
# 参考摘要
from nltk.translate.bleu_score import sentence_bleu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Model():
    def __init__(self) -> None:
        path='/public/home/thirring/chatglm3'
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
    def Chat(self,prompt,history=[]):

        response,_=self.model.chat(self.tokenizer,prompt,history)
        return response

def Judeg(per_data,now_data):

    prompt='''This is the current question and answer pair, query:{},answer:{}. Is the current answer based on the previous question and answer \
        pair query:{},answer:{}. Please answer yes or no'''.format(now_data['Question'],now_data['Answer'],per_data['Question'],per_data['Answer'])
    output=chat_model.Chat(prompt)
    output=output.replace(prompt,'')
    if('no' in output):
        return False;
    elif('yes' in output):
        return True;
    else:
        raise ValueError('逻辑判断不正确')
def get_rougescore(ref,sys):
    rouge = Rouge()
    scores = rouge.get_scores(sys, ref)
    return scores;
def color_history(history):
    format_history=[]
    for i in range(0,len(history),2):
        format_history.append({'role': 'user', 'content': history[i]})
        format_history.append({'role': 'assistant', 'metadata': '', 'content': history[i+1]})    
    return format_history



def color_question(question,history):
    #使用提示词将question和history组合起来
    prompt='''Please Answer the question based on doc, doc:{}, question:{}'''.format(history,question)
    return prompt;
chat_model=Model();
def main():
    count=0
    with open('./topiocqa_dev.json','r') as f:
        data_all=json.load(f);
    count=0
    for index,data_ in enumerate(data_all):
        history=data_['Context']

        chat_len=len(history)
        question=data_['Question']
        answer=data_['Answer']  
        select_his=[];
        question=color_question(question=question,history=data_['Rationale'])
        print('question',question)
        #用rouge中的recall效果更佳
        if(chat_len>=6 and answer!='UNANSWERABLE'):
            answer_query=chat_model.Chat(question,history=color_history(history))
            score_query=get_rougescore(answer,answer_query)
            for i in range(chat_len-2,-2,-2):
                pre_history=history[i:]
                answer_his=chat_model.Chat(question,history=color_history(pre_history))
                score_answer=get_rougescore(answer,answer_his)
                if(score_answer[0]['rouge-1']['r']>score_query[0]['rouge-1']['r'] and i !=0):
                    select_his.append({'history':pre_history,'score_answer':score_answer,'answer_his':answer_his,'answer_all':answer_query})
                    count+=1;
        data_['select_history']=select_his;
        print('index',index,'count',count)
        
        with open('./TopiOCQA_dev_withhist_ration.jsonl','a') as fw:
            fw.write(json.dumps(data_))  
            fw.write('\n')  
                    
    print(count)                
main()
