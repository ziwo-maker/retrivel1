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
        import pdb
        pdb.set_trace()
        inputs=self.tokenizer([prompt],return_tensors='pt').to('cuda')
        outputs=self.model.generate(**inputs,max_new_tokens=len(prompt))
        return self.tokenizer.decode(outputs[0].tolist());
chat_model=Model();
def Judeg(per_data,now_data):

    prompt='''This is the current question and answer pair, query:{},answer:{}. Is the current answer based on the previous question and answer \
        pair query:{},answer:{}. Please answer yes or no'''.format(now_data['Question'],now_data['Answer'],per_data['Question'],per_data['Answer'])
    output=chat_model.Chat(prompt)
    output=output.replace(prompt,'').lower()
    if('no' in output):
        return False;
    elif('yes' in output):
        return True;
    else:
        raise ValueError('逻辑判断不正确')


def get_bleu(ref,sys):
    score = sentence_bleu(ref, sys);

def get_score(ref,sys):
    rouge = Rouge()
    scores = rouge.get_scores(sys, ref)
    return scores;
def main():
    count=0
    with open('./qrecc_train.json','r') as f:
        data_all=json.load(f);
        Conversation_no=None;
        change=None;
        for index,data_ in  enumerate(data_all):

            if(Conversation_no==None):
                Conversation_no=data_['Conversation_no']
                continue;
            if(Conversation_no!=data_['Conversation_no'] ):
                #下一轮对话。
                Conversation_no=data_['Conversation_no']
                continue
            #Topic 转变 检查上下文是否转变.
            
            change=Judeg(data_all[index-1],data_);
            if(change==None):
                change=False;
            data_['change']=change;
            with open('./qrecc_train_change.jsonl','a') as fw:
                fw.write(json.dumps(data_))
                fw.write('\n')

main()