from transformers import AutoTokenizer, AutoModel
class Model():
    def __init__(self,mode_path) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(mode_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        
    def Chat(self,prompt,history=[],**kwargs):

        response,_=self.model.chat(self.tokenizer,prompt,history,**kwargs)
        return response