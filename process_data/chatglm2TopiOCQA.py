import json
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge
import json
# 参考摘要
from nltk.translate.bleu_score import sentence_bleu

class Model():
    def __init__(self) -> None:
        path='/public/home/thirring/GLM/chatglm3-6b-base'
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
    def Chat(self,prompt,history=[]):

        response,_=self.model.chat(self.tokenizer,prompt,history)
        return response
chat_model=Model();
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

def main():
    count=0
    with open('./topiocqa_train.json','r') as f:
        data_all=json.load(f);

    for data_ in data_all:
        hisotry=data_['Context']
        flag=0;
        chat_len=len(hisotry)
        question=data_['Question']
        answer=data_['Answer']  
        if(chat_len!=0 and answer!='UNANSWERABLE'):
            for i in range(chat_len-2,-2,-2):
                pre_history=hisotry[i:]
                answer_query=chat_model.Chat(question)
                answer_his=chat_model.Chat(question,history=pre_history)
                score_query=get_rougescore(answer,answer_query)
                score_answer=get_rougescore(answer,answer_his)
main()