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
        
        if(type=='mistral' ):
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(mode_path, trust_remote_code=True).half().to('cuda')
            self.model = self.model.eval()
        elif type=='llama3' or type=='qwen':
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(mode_path,  torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).half().to('cuda')
            
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(mode_path, trust_remote_code=True).half().cuda()
            self.model = self.model.eval()
            
    def Chat(self,prompt,history=[],**kwargs):

        if(self.type=='chatglm'):
            history=color_history_chat(history)
            response,_=self.model.chat(self.tokenizer,prompt,history,**kwargs)
        
            return response

        messages=colo_histoty(history,self.type)
        messages.append(colo_histoty([prompt],self.type))
        
        encodeds = self.tokenizer.apply_chat_template(messages,add_generation_prompt=True, return_tensors="pt").to('cuda')
        
        if(self.type=='llama3'):
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            generated_ids = self.model.generate(encodeds,eos_token_id=terminators,do_sample=False,**kwargs)
  
            response = generated_ids[0][encodeds.shape[-1]:]
            answer=self.tokenizer.decode(response, do_sample=False)
            return answer 
        elif self.type=='qwen':
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
                )
            model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            generated_ids = self.model.generate(encodeds,do_sample=False,**kwargs)
            decoded = self.tokenizer.batch_decode(generated_ids)

            pos=decoded[0].rfind("[/INST]")
            answer=decoded[0][pos+len('[/INST]'):]

            return answer