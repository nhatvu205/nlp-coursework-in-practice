import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from evaluator import get_predictions, evaluate

class EarlyStopping:
    def __init__(self, patience=3, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class ModelTrainer:
    def __init__(self, model, tokenizer, config, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_em': []
        }
    
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def evaluate_val_f1(self, val_loader):
        val_data = []
        ground_truth = {}
        
        for i, item in enumerate(val_loader.dataset.data):
            val_data.append({
                'id': f'val_{i}',
                'question': item['question'],
                'context': item['context']
            })
            ground_truth[f'val_{i}'] = {
                'is_impossible': item.get('is_impossible', False),
                'answer': item.get('answer', '') if item.get('answer') else ''
            }
        
        predictions = get_predictions(
            self.model,
            self.tokenizer,
            val_data,
            self.config,
            self.device
        )
        
        f1_score, em_score = evaluate(predictions, ground_truth)
        return f1_score, em_score
    
    def train(self, train_loader, val_loader):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            threshold=self.config.early_stopping_threshold
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.config.num_epochs}')
            
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc='Training')
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            train_loss = total_loss / len(train_loader)
            val_loss = self.validate(val_loader)
            val_f1, val_em = self.evaluate_val_f1(val_loader)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['val_em'].append(val_em)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Val F1: {val_f1:.4f}, Val EM: {val_em:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            
            early_stopping(val_f1)
            if early_stopping.early_stop:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break
    
    def plot_training_history(self, model_name: str):
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        if self.training_history['val_f1'] or self.training_history['val_em']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_title(f'{model_name} - Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            if self.training_history['val_f1']:
                axes[0, 1].plot(epochs[:len(self.training_history['val_f1'])], 
                               self.training_history['val_f1'], 'g-', label='Val F1')
                axes[0, 1].set_title(f'{model_name} - F1 Score')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            else:
                axes[0, 1].axis('off')
            
            if self.training_history['val_em']:
                axes[1, 0].plot(epochs[:len(self.training_history['val_em'])], 
                               self.training_history['val_em'], 'm-', label='Val EM')
                axes[1, 0].set_title(f'{model_name} - Exact Match')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Exact Match')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].axis('off')
            
            if self.training_history['val_f1'] and self.training_history['val_em']:
                axes[1, 1].plot(epochs[:len(self.training_history['val_f1'])], 
                               self.training_history['val_f1'], 'g-', label='F1')
                axes[1, 1].plot(epochs[:len(self.training_history['val_em'])], 
                               self.training_history['val_em'], 'm-', label='EM')
                axes[1, 1].set_title(f'{model_name} - F1 vs EM')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                axes[1, 1].axis('off')
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
            axes.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
            axes.set_title(f'{model_name} - Training History')
            axes.set_xlabel('Epoch')
            axes.set_ylabel('Loss')
            axes.legend()
            axes.grid(True)
        
        plt.tight_layout()
        plt.show()

