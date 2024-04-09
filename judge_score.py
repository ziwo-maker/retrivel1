from rouge import Rouge
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import evaluate
from datasets import load_metric
from nltk.translate import meteor_score
# from nltk.translate import cider
def get_rougescore( ref, sys):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(sys, ref)
    except Exception as e:
        scores=[{'rouge-1':{'r':2,'p':2}}]
    return scores

class Judge_score():
    def __init__(self,mode_path='/home/server/GX/allmini/') -> None:
        self.cos_model = SentenceTransformer(mode_path)
        model_name = "/data/dataset/gpt2/"
        self.ppltokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.pplmodel = GPT2LMHeadModel.from_pretrained(model_name)

    def cos_similarity(self, stand_ans: str, others: List[str]):
        if not isinstance(others, list):
            others=[others]
        
        embedding = self.cos_model.encode([stand_ans] + others)
        stand_ans_embedding=torch.tensor(embedding[0]);
        embedding=torch.tensor(embedding[1:])
        similarity_scores = torch.nn.functional.cosine_similarity(stand_ans_embedding, embedding)
        similarity_scores=similarity_scores.numpy()
        return similarity_scores[0]
def get_ppl(text):

    #metric = evaluate.load(path='/home/server/GX/evaluate/metrics/bleu/')
    metric = evaluate.load('perplexity')
    results = metric.compute(text)
    return results

    # def 
def get_bert_score(ref,sys):
    metric = evaluate.load('bertscore')
    results = metric.compute(predictions=sys, references=ref)
    return results
def get_bleu(ref,sys):
    metric = evaluate.load('bleu')
    results = metric.compute(predictions=sys, references=ref)
    return results
def get_meteor(ref,sys):
    met_score = meteor_score.meteor_score(ref, sys)
    return met_score
# def get_cider(ref,sys):
#     cider_scorer = cider.Cider()
#     cid_score =cider_scorer.compute_score(ref, sys)
#     return cid_score