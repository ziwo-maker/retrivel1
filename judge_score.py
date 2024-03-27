from rouge import Rouge
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import torch
class Judge_score():
    def __init__(self,mode_path) -> None:
        self.cos_model = SentenceTransformer(mode_path)

    def get_rougescore(self, ref, sys):
        rouge = Rouge()
        scores = rouge.get_scores(sys, ref)
        return scores

    def cos_similarity(self, stand_ans: str, others: List[str]):
        if not isinstance(others, list):
            others=[others]
        
        embedding = self.cos_model.encode([stand_ans] + others)
        stand_ans_embedding=torch.tensor(embedding[0]);
        embedding=torch.tensor(embedding[1:])
        similarity_scores = torch.nn.functional.cosine_similarity(stand_ans_embedding, embedding)
        similarity_scores=similarity_scores.numpy()
        return similarity_scores[0]