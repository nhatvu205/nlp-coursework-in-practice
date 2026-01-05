import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import re
from tqdm import tqdm

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = set(pred_tokens) & set(truth_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def get_predictions(model, tokenizer, test_data, config, device):
    model.eval()
    predictions = {}
    
    for item in tqdm(test_data, desc='Predicting'):
        question = item['question']
        context = item['context']
        qid = item['id']
        
        encoded = tokenizer(
            question,
            context,
            truncation='only_second',
            max_length=config.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            padding='max_length'
        )
        
        offset_mapping = encoded.pop('offset_mapping')[0].cpu().numpy()
        inputs = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()
        
        start_idx = torch.argmax(start_logits, dim=1).item()
        end_idx = torch.argmax(end_logits, dim=1).item()
        
        question_tokens = tokenizer(question, add_special_tokens=True, return_offsets_mapping=True)
        question_end = question_tokens['offset_mapping'][-1][1] if len(question_tokens['offset_mapping']) > 0 else 0
        
        if start_idx > end_idx or start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
            predictions[qid] = ""
        else:
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx][1]
            
            if start_char == 0 and end_char == 0:
                predictions[qid] = ""
            elif start_char <= question_end:
                predictions[qid] = ""
            else:
                answer_start = start_char - question_end - 1
                answer_end = end_char - question_end - 1
                if answer_start >= 0 and answer_end <= len(context):
                    answer_text = context[answer_start:answer_end]
                    predictions[qid] = answer_text.strip()
                else:
                    predictions[qid] = ""
    
    return predictions

def evaluate(predictions: Dict[str, str], ground_truth: Dict[str, Dict]) -> Tuple[float, float]:
    exact_matches = []
    f1_scores = []
    
    for qid, pred_text in predictions.items():
        if qid not in ground_truth:
            continue
        
        gt = ground_truth[qid]
        
        if gt['is_impossible']:
            if pred_text == "" or pred_text is None:
                exact_matches.append(1.0)
                f1_scores.append(1.0)
            else:
                exact_matches.append(0.0)
                f1_scores.append(0.0)
        else:
            if pred_text is None or pred_text == "":
                exact_matches.append(0.0)
                f1_scores.append(0.0)
            else:
                gt_answer = gt['answer'] if gt['answer'] else ""
                em = compute_exact_match(pred_text, gt_answer)
                f1 = compute_f1(pred_text, gt_answer)
                exact_matches.append(em)
                f1_scores.append(f1)
    
    em_score = np.mean(exact_matches) if exact_matches else 0.0
    f1_score = np.mean(f1_scores) if f1_scores else 0.0
    
    return f1_score, em_score

