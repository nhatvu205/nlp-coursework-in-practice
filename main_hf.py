import sys
import os
sys.path.append('pytorch_qa')

import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
    default_data_collator,
)
import evaluate
from pytorch_qa.trainer_qa import QuestionAnsweringTrainer
from pytorch_qa.utils_qa import postprocess_qa_predictions

from config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_model(model_name, model_path, config):
    logger.info(f"Training {model_name.upper()}")
    
    train_dir = os.path.join(config.data_path, config.train_dir)
    dev_dir = os.path.join(config.data_path, config.dev_dir)
    test_dir = os.path.join(config.data_path, config.test_dir)
    
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.json')] if os.path.exists(train_dir) else []
    dev_files = [os.path.join(dev_dir, f) for f in os.listdir(dev_dir) if f.endswith('.json')] if os.path.exists(dev_dir) else []
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.endswith('.json') and 'ground_truth' not in f] if os.path.exists(test_dir) else []
    
    data_files = {}
    if train_files:
        data_files["train"] = train_files[0] if len(train_files) == 1 else train_files
    if dev_files:
        data_files["validation"] = dev_files[0] if len(dev_files) == 1 else dev_files
    if test_files:
        data_files["test"] = test_files[0] if len(test_files) == 1 else test_files
    
    if not data_files:
        raise ValueError(f"No data files found in {config.data_path}")
    
    raw_datasets = load_dataset("json", data_files=data_files, field="data")
    
    config_model = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config_model)
    
    if not (hasattr(tokenizer, "_tokenizer") or getattr(tokenizer, "is_fast", False)):
        raise TypeError("This script requires a fast tokenizer")
    
    column_names = raw_datasets["train"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(config.max_length, tokenizer.model_max_length)
    
    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            
            if len(answers.get("answer_start", [])) == 0 or answers.get("is_impossible", False):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples
    
    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    train_dataset = raw_datasets["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=column_names,
    )
    
    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
    )
    
    test_examples = raw_datasets.get("test", None)
    if test_examples:
        predict_dataset = test_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
        )
    else:
        predict_dataset = None
        test_examples = None
    
    data_collator = default_data_collator
    
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, model_name),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.seed,
        report_to="none",
    )
    
    metric = evaluate.load("squad_v2")
    
    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=20,
            max_answer_length=config.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=None,
            log_level=logging.WARNING,
            prefix=stage,
        )
        
        formatted_predictions = [
            {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} 
            for k, v in predictions.items()
        ]
        
        references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    from transformers import TrainerCallback
    
    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, patience=3, threshold=0.001):
            self.patience = patience
            self.threshold = threshold
            self.counter = 0
            self.best_score = None
            self.early_stop = False
        
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get('metrics', {})
            current_f1 = metrics.get('eval_f1', 0)
            
            if self.best_score is None:
                self.best_score = current_f1
            elif current_f1 < self.best_score + self.threshold:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    control.should_training_stop = True
            else:
                self.best_score = current_f1
                self.counter = 0
    
    training_history = {'train_loss': [], 'eval_loss': [], 'eval_f1': [], 'eval_exact': []}
    early_stopping_cb = EarlyStoppingCallback(
        patience=config.early_stopping_patience,
        threshold=config.early_stopping_threshold
    )
    
    class HistoryCallback(TrainerCallback):
        def __init__(self, history):
            self.history = history
        
        def on_log(self, args, state, control, **kwargs):
            logs = kwargs.get('logs', {})
            if 'loss' in logs and 'epoch' in logs:
                if logs['epoch'] not in [h.get('epoch', -1) for h in self.history.get('train_loss', [])]:
                    self.history['train_loss'].append({'epoch': logs['epoch'], 'loss': logs['loss']})
        
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get('metrics', {})
            epoch = state.epoch if hasattr(state, 'epoch') else len(self.history['eval_loss'])
            self.history['eval_loss'].append({'epoch': epoch, 'loss': metrics.get('eval_loss', 0)})
            self.history['eval_f1'].append({'epoch': epoch, 'f1': metrics.get('eval_f1', 0)})
            self.history['eval_exact'].append({'epoch': epoch, 'em': metrics.get('eval_exact', 0)})
    
    history_cb = HistoryCallback(training_history)
    trainer.add_callback(early_stopping_cb)
    trainer.add_callback(history_cb)
    
    train_result = trainer.train()
    
    eval_metrics = trainer.evaluate()
    
    train_losses = [h['loss'] for h in training_history['train_loss']]
    eval_losses = [h['loss'] for h in training_history['eval_loss']]
    eval_f1s = [h['f1'] for h in training_history['eval_f1']]
    eval_ems = [h['em'] for h in training_history['eval_exact']]
    
    epochs = range(1, len(eval_losses) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if train_losses:
        axes[0, 0].plot(epochs[:len(train_losses)], train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, eval_losses, 'r-', label='Eval Loss')
    axes[0, 0].set_title(f'{model_name.upper()} - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, eval_f1s, 'g-', label='Eval F1')
    axes[0, 1].set_title(f'{model_name.upper()} - F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, eval_ems, 'm-', label='Eval EM')
    axes[1, 0].set_title(f'{model_name.upper()} - Exact Match')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Exact Match')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, eval_f1s, 'g-', label='F1')
    axes[1, 1].plot(epochs, eval_ems, 'm-', label='EM')
    axes[1, 1].set_title(f'{model_name.upper()} - F1 vs EM')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    test_results = {}
    if test_examples and predict_dataset:
        logger.info("Evaluating on test set...")
        results = trainer.predict(predict_dataset, test_examples)
        test_results = results.metrics
        
        logger.info(f"Test F1: {test_results.get('test_f1', 0):.4f}")
        logger.info(f"Test EM: {test_results.get('test_exact', 0):.4f}")
    
    return {
        'f1': test_results.get('test_f1', eval_metrics.get('eval_f1', 0)),
        'em': test_results.get('test_exact', eval_metrics.get('eval_exact', 0))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['mbert', 'xlmr', 'phobert', 'all'],
                       default=None)
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
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
        logger.error(f"No valid models selected. Available: {list(config.models.keys())}")
        return
    
    logger.info(f"Models to train: {[m.upper() for m in models_to_train]}")
    
    results = {}
    for model_name in models_to_train:
        try:
            result = train_model(model_name, config.models[model_name], config)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    if results:
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS COMPARISON")
        logger.info("="*60)
        
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
        
        logger.info("\nDetailed Results:")
        for model_name in model_names:
            logger.info(f"{model_name.upper()}: F1={results[model_name]['f1']:.4f}, EM={results[model_name]['em']:.4f}")


if __name__ == "__main__":
    main()

