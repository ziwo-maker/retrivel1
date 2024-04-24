from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from color import color_history_chat,colo_histoty
import re
import torch


def filter_answers(message, answers):
    for mes in message:
        answers=re.sub(mes,'',answers)
    return answers
#history的color需要在这里面执行
class Model():
    def __init__(self,mode_path,type='mistral') -> None:
        
        self.type=type;
        
        if(type=='mistral'):
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(mode_path, trust_remote_code=True).half().to('cuda')
            self.model = self.model.eval()
        
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(mode_path, trust_remote_code=True).half().cuda()
            self.model = self.model.eval()
            

    def Chat(self,prompt,history=[],**kwargs):

        if(self.type=='chatglm'):
            history=color_history_chat(history)
            response,_=self.model.chat(self.tokenizer,prompt,history,**kwargs)
        
            return response
        # import pdb;
        # pdb.set_trace()
        messages=colo_histoty(history)
        messages.append(colo_histoty([prompt]))
        
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(encodeds,do_sample=False,**kwargs)
        decoded = self.tokenizer.batch_decode(generated_ids)

        pos=decoded[0].rfind("[/INST]")
        answer=decoded[0][pos+len('[/INST]'):]

        return answer