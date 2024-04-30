import matplotlib.pyplot as plt
import logging
# 创建数据加载器
import numpy as np
from judge_score import get_rougescore
import re
from judge_score import get_meteor,get_bleu
def draw_plt(train_loss=[],dev_loss=[]):
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b', label='Training loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_trend.png')



def separate_words_and_punctuation(text):
    # 使用正则表达式将单词和标点符号分开
    separated_text = re.findall(r"[\w']+|[.,!?;]", text)
    # 将分开的单词和标点符号连接成字符串
    separated_text = ' '.join(separated_text)
    return separated_text
def calculate_metrics(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum().item()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-8)

    recall = true_positive / (true_positive + false_negative + 1e-8)
    acc = ((y_pred ==y_true).sum()).item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return recall,precision, f1,acc/len(y_true+1e-8)
#返回多个答案对比中的recall.这里虽然对比的是recall但是还是返回全部，方便分析记录。
def get_best_score(answe_list,evalute_answer):
    if not isinstance(answe_list, list):
        answe_list=[answe_list]
    max_score=0.0;
    for index in range(len(answe_list)):
        answe_list[index]=separate_words_and_punctuation(answe_list[index])
    evalute_answer=separate_words_and_punctuation(evalute_answer)

    score_list=[]
    for ans in answe_list:
        score=get_rougescore(ans.lower(),evalute_answer.lower())
        max_score=max(max_score,score[0]['rouge-1']['r'])
        score_list.append(score)
    for score in score_list:
        if(score[0]['rouge-1']['r']==max_score):
            return score
def get_meteor_score(ref,sys):
    #先转换大小写，再分词。
    if not isinstance(ref, list):
        ref=[ref]
    for i in range(len(ref)):

        ref[i]=separate_words_and_punctuation(ref[i]).lower()
        
    
    if not isinstance(sys, list):
        sys=[sys]
    for i in range(len(sys)):

        sys[i]=separate_words_and_punctuation(sys[i]).lower()
    
    score=get_meteor(ref,sys)
    return score
def get_bleu_score(ref,sys):
    #先转换大小写，再分词。
    if not isinstance(ref, list):
        ref=[ref]
    for i in range(len(ref)):
        ref[i]=separate_words_and_punctuation(ref[i]).lower()
        ref[i]=ref[i].split()
    if not isinstance(sys, list):
        sys=[sys]

    for i in range(len(ref)):
        sys[i]=separate_words_and_punctuation(sys[i]).lower()
        sys[i]=sys[i].split()
    score=get_bleu(ref,sys)
    return score
def compute_mrr(sorted_list):
    total_mrr = 0
    count = 0
    for item in sorted_list:
        count += 1
        if item['label'] == 1:
            total_mrr += 1 / count
    return total_mrr 


def compute_precision_recall(sorted_list):
    total_retrieved = len([item for item in sorted_list if item['label'] == 1])
    evaluate_list=sorted_list[:int(len(sorted_list)/2)+1]

    half_len=min(len(evaluate_list),total_retrieved)
    total_relevant = sum(item['label'] for item in evaluate_list)
    recall = total_relevant / total_retrieved if total_retrieved > 0 else 0
    precision = total_relevant / half_len if half_len > 0 else 0


    return precision, recall

def jaccard_similarity(ref, sys):
    # 将句子分割成单词并转换为集合
    sentence1=separate_words_and_punctuation(ref).lower()
    sentence2=separate_words_and_punctuation(sys).lower()
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    
    # 计算交集和并集的大小
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # 计算Jaccard相似度
    similarity = intersection / union if union != 0 else 0
    
    return similarity