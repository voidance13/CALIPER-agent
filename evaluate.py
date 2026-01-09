import regex
import string
from collections import Counter
import json
import re
from sentence_transformers import SentenceTransformer, util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

model = SentenceTransformer('paraphrase-mpnet-base-v2')

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # if normalized_prediction in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC
    # if normalized_ground_truth in ['yes', 'no'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def eval_hit(prediction, answer):
    if normalize_answer(answer) in normalize_answer(prediction):
        return 1
    return 0

def parse_answer(response):
    try:
        return response[re.search("Answer:\n", response).end():]
    except:
        try: 
            return response[re.search("answer:\n", response).end():]
        except:
            try:
                return response[re.search("Answer: ", response).end():]
            except:
                try:
                    return response[re.search("answer: ", response).end():]
                except:
                    try:
                        return response[re.search("Answer:", response).end():]
                    except:
                        try:
                            return response[re.search("answer:", response).end():]
                        except:
                            line = response.split("\n")[-1]
                            try:
                                return line[re.search(": ", line).end():]
                            except:
                                return line

def evaluate_generation(answer, result):
    embedding1 = model.encode(answer, convert_to_tensor=True)
    embedding2 = model.encode(result, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    f1, precision, recall = f1_score(result, answer)
    hit = eval_hit(result, answer)

    return cosine_score.item(), f1, precision, recall, hit    
    
def evaluate_retrieval(supporting_facts, retrieved_docs):
    hit3 = 0
    hit5 = 0
    hit10 = 0
    for supporting_fact in supporting_facts:
        if supporting_fact in retrieved_docs[:10]:
            hit10 += 1
        if supporting_fact in retrieved_docs[:5]:
            hit5 += 1
        if supporting_fact in retrieved_docs[:3]:
            hit3 += 1
    l = len(supporting_facts)
    return hit3 / l, hit5 / l, hit10 / l
