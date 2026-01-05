import torch
import numpy as np
import random
import argparse
import sys
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import matplotlib.pyplot as plt

from config import Config
from data_loader import prepare_datasets, load_test_data
from model_trainer import ModelTrainer
from evaluator import get_predictions, evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune transformer models on UIT-ViQuADv2')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['mbert', 'xlmr', 'phobert', 'all'],
                       default=None,
                       help='Models to train: mbert, xlmr, phobert, or all (default: from config)')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    set_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.models:
        if 'all' in args.models:
            models_to_train = list(config.models.keys())
        else:
            models_to_train = args.models
    else:
        models_to_train = config.models_to_train
    
    valid_models = set(config.models.keys())
    models_to_train = [m for m in models_to_train if m in valid_models]
    
    if not models_to_train:
        print('Error: No valid models selected. Available models:', list(config.models.keys()))
        sys.exit(1)
    
    print(f'Models to train: {[m.upper() for m in models_to_train]}')
    results = {}
    
    for model_name in models_to_train:
        print(f'\n{"="*60}')
        print(f'Training {model_name.upper()}')
        print(f'{"="*60}')
        
        print('Loading datasets...')
        train_dataset, dev_dataset, tokenizer = prepare_datasets(config, model_name)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f'Loading model: {config.models[model_name]}')
        model = AutoModelForQuestionAnswering.from_pretrained(config.models[model_name])
        
        trainer = ModelTrainer(model, tokenizer, config, device)
        
        print('Starting training...')
        trainer.train(train_loader, dev_loader)
        
        print('Plotting training history...')
        trainer.plot_training_history(model_name.upper())
        
        print('Evaluating on test set...')
        test_data, ground_truth = load_test_data(config)
        
        test_predictions = get_predictions(
            trainer.model,
            tokenizer,
            test_data,
            config,
            device
        )
        
        f1_score, em_score = evaluate(test_predictions, ground_truth)
        results[model_name] = {'f1': f1_score, 'em': em_score}
        
        print(f'\n{model_name.upper()} Results:')
        print(f'F1 Score: {f1_score:.4f}')
        print(f'Exact Match: {em_score:.4f}')
        
        del model, trainer, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f'\n{"="*60}')
    print('FINAL RESULTS COMPARISON')
    print(f'{"="*60}')
    
    model_names = list(results.keys())
    f1_scores = [results[m]['f1'] for m in model_names]
    em_scores = [results[m]['em'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='skyblue')
    bars2 = ax.bar(x + width/2, em_scores, width, label='Exact Match', color='lightcoral')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: F1 Score vs Exact Match')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (f1, em) in enumerate(zip(f1_scores, em_scores)):
        ax.text(i - width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, em + 0.01, f'{em:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print('\nDetailed Results:')
    for model_name in model_names:
        print(f'{model_name.upper()}: F1={results[model_name]["f1"]:.4f}, EM={results[model_name]["em"]:.4f}')

if __name__ == '__main__':
    main()

