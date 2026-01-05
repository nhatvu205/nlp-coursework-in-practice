import json
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class ViQuADDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=384, doc_stride=128, max_answer_length=30, is_training=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        context = example['context']
        question = example['question']
        is_impossible = example.get('is_impossible', False)
        
        tokenized = self.tokenizer(
            question,
            context,
            truncation='only_second',
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        offset_mapping = tokenized['offset_mapping'].squeeze()
        
        start_positions = torch.tensor(0, dtype=torch.long)
        end_positions = torch.tensor(0, dtype=torch.long)
        
        if self.is_training:
            if not is_impossible and example.get('answer'):
                answer_text = example['answer']
                
                question_tokens = self.tokenizer(question, add_special_tokens=True, return_offsets_mapping=True)
                question_char_end = question_tokens['offset_mapping'][-1][1] if len(question_tokens['offset_mapping']) > 0 else 0
                
                answer_start_in_context = context.find(answer_text)
                
                if answer_start_in_context != -1:
                    answer_start_char = question_char_end + 1 + answer_start_in_context
                    answer_end_char = answer_start_char + len(answer_text)
                    
                    token_start_idx = 0
                    token_end_idx = 0
                    found_start = False
                    found_end = False
                    
                    for i, (start, end) in enumerate(offset_mapping):
                        if start == 0 and end == 0:
                            continue
                        
                        if not found_start and start <= answer_start_char < end:
                            token_start_idx = i
                            found_start = True
                        
                        if found_start and not found_end:
                            if start < answer_end_char <= end:
                                token_end_idx = i
                                found_end = True
                                break
                    
                    if found_start and found_end and token_start_idx <= token_end_idx:
                        start_positions = torch.tensor(token_start_idx, dtype=torch.long)
                        end_positions = torch.tensor(token_end_idx, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'is_impossible': torch.tensor(1 if is_impossible else 0, dtype=torch.long),
            'context': context,
            'question': question,
            'answer': example.get('answer', ''),
            'offset_mapping': offset_mapping
        }

def load_json_files(directory: str) -> List[Dict]:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if 'data' in content:
                    for article in content['data']:
                        for paragraph in article['paragraphs']:
                            context = paragraph['context']
                            for qa in paragraph['qas']:
                                item = {
                                    'context': context,
                                    'question': qa['question'],
                                    'id': qa['id']
                                }
                                
                                if 'is_impossible' in qa:
                                    item['is_impossible'] = qa['is_impossible']
                                else:
                                    item['is_impossible'] = False
                                
                                if not item['is_impossible'] and len(qa['answers']) > 0:
                                    item['answer'] = qa['answers'][0]['text']
                                else:
                                    item['answer'] = None
                                
                                if 'plausible_answers' in qa and len(qa['plausible_answers']) > 0:
                                    item['plausible_answer'] = qa['plausible_answers'][0]['text']
                                else:
                                    item['plausible_answer'] = None
                                
                                data.append(item)
                else:
                    data.extend(content)
    return data

def prepare_datasets(config, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(config.models[model_name])
    
    train_data = load_json_files(os.path.join(config.data_path, config.train_dir))
    dev_data = load_json_files(os.path.join(config.data_path, config.dev_dir))
    
    train_dataset = ViQuADDataset(
        train_data,
        tokenizer,
        max_length=config.max_length,
        doc_stride=config.doc_stride,
        max_answer_length=config.max_answer_length,
        is_training=True
    )
    
    dev_dataset = ViQuADDataset(
        dev_data,
        tokenizer,
        max_length=config.max_length,
        doc_stride=config.doc_stride,
        max_answer_length=config.max_answer_length,
        is_training=False
    )
    
    return train_dataset, dev_dataset, tokenizer

def load_test_data(config):
    test_data = []
    test_dir = os.path.join(config.data_path, config.test_dir)
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.json') and 'ground_truth' not in filename:
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    test_data.extend(content)
                elif 'data' in content:
                    for article in content['data']:
                        for paragraph in article['paragraphs']:
                            context = paragraph['context']
                            for qa in paragraph['qas']:
                                item = {
                                    'context': context,
                                    'question': qa['question'],
                                    'id': qa['id']
                                }
                                test_data.append(item)
    
    ground_truth_file = os.path.join(test_dir, 'ground_truth_private_test.json')
    ground_truth = {}
    if os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            if isinstance(gt_data, list):
                for item in gt_data:
                    ground_truth[item['id']] = item
            elif 'data' in gt_data:
                for article in gt_data['data']:
                    for paragraph in article['paragraphs']:
                        for qa in paragraph['qas']:
                            ground_truth[qa['id']] = {
                                'is_impossible': qa.get('is_impossible', False),
                                'answer': qa['answers'][0]['text'] if not qa.get('is_impossible', False) and len(qa.get('answers', [])) > 0 else None
                            }
    
    return test_data, ground_truth

