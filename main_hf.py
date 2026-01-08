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
from llm_inference import evaluate_llm, evaluate_llm_samples, load_llm, LLM_MAP

from config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_flat_samples(split_dir):
    samples = []
    if not os.path.isdir(split_dir):
        return samples
    for filename in os.listdir(split_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(split_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
        items = []
        if isinstance(content, list):
            items = content
        elif isinstance(content, dict) and "data" in content:
            items = content["data"]
        for it in items:
            obj = {
                "id": it.get("id", f"id_{len(samples)}"),
                "context": it.get("context", ""),
                "question": it.get("question", ""),
                "is_impossible": it.get("is_impossible", False),
            }
            
            if obj["is_impossible"]:
                obj["answer"] = ""
            else:
                obj["answer"] = ""
                if "answer" in it and it["answer"]:
                    obj["answer"] = it["answer"]
                elif "answers" in it:
                    answers = it["answers"]
                    if isinstance(answers, dict):
                        if "text" in answers and isinstance(answers["text"], list) and len(answers["text"]) > 0:
                            obj["answer"] = answers["text"][0]
                        elif "text" in answers:
                            obj["answer"] = answers["text"]
                        elif "answer_start" in answers and len(answers.get("answer_start", [])) == 0:
                            obj["answer"] = ""
                        obj["is_impossible"] = answers.get("is_impossible", obj["is_impossible"])
                    elif isinstance(answers, list) and len(answers) > 0:
                        ans0 = answers[0]
                        if isinstance(ans0, dict):
                            obj["answer"] = ans0.get("text", ans0.get("answer", ""))
                        else:
                            obj["answer"] = str(ans0) if ans0 else ""
                    elif isinstance(answers, str):
                        obj["answer"] = answers
            samples.append(obj)
    return samples


def load_ground_truth(test_dir):
    ground_truth = {}
    logger.info(f"Searching for ground truth in: {test_dir}")
    if not os.path.exists(test_dir):
        logger.warning(f"Test directory not found: {test_dir}")
        return ground_truth
    
    all_files = os.listdir(test_dir)
    logger.info(f"All files in test directory: {all_files}")
    ground_truth_files = [f for f in all_files if 'ground_truth' in f.lower() and f.endswith('.json')]
    logger.info(f"Ground truth files found after filter: {ground_truth_files}")
    
    if not ground_truth_files:
        logger.warning(f"No ground truth files found matching pattern 'ground_truth*.json' in {test_dir}")
        return ground_truth
    
    for filename in ground_truth_files:
        filepath = os.path.join(test_dir, filename)
        logger.info(f"Loading ground truth from: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            logger.info(f"Ground truth file structure: {type(content)}, keys: {content.keys() if isinstance(content, dict) else 'N/A'}")
            
            items = []
            if isinstance(content, list):
                items = content
            elif isinstance(content, dict) and "data" in content:
                items = content["data"]
            elif isinstance(content, dict):
                for key in content.keys():
                    val = content[key]
                    if isinstance(val, list):
                        logger.info(f"Found list under key '{key}' with {len(val)} items")
                        items = val
                        break
            
            logger.info(f"Processing {len(items)} items from ground truth file")
            if len(items) > 0:
                logger.info(f"Sample ground truth entry: {items[0]}")
            
            for it in items:
                sample_id = it.get("id", "")
                if not sample_id:
                    continue
                
                answer = ""
                is_impossible = it.get("is_impossible", False)
                
                if not is_impossible:
                    if "answer" in it and it["answer"]:
                        answer = it["answer"]
                    elif "answers" in it:
                        answers = it["answers"]
                        if isinstance(answers, dict):
                            if "text" in answers and isinstance(answers["text"], list) and len(answers["text"]) > 0:
                                answer = answers["text"][0]
                            elif "text" in answers:
                                answer = answers["text"]
                        elif isinstance(answers, list) and len(answers) > 0:
                            ans0 = answers[0]
                            if isinstance(ans0, dict):
                                answer = ans0.get("text", ans0.get("answer", ""))
                            else:
                                answer = str(ans0) if ans0 else ""
                        elif isinstance(answers, str):
                            answer = answers
                
                ground_truth[sample_id] = {
                    "answer": answer,
                    "is_impossible": is_impossible
                }
        except Exception as e:
            logger.error(f"Error loading ground truth from {filename}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Loaded {len(ground_truth)} ground truth entries")
    if len(ground_truth) > 0:
        sample_ids = list(ground_truth.keys())[:3]
        logger.info(f"Sample ground truth IDs: {sample_ids}")
    
    return ground_truth


def merge_ground_truth(samples, ground_truth):
    if len(samples) > 0:
        logger.info(f"Sample test IDs before merge: {[s.get('id') for s in samples[:3]]}")
    
    matched = 0
    for sample in samples:
        sample_id = sample.get("id", "")
        if sample_id in ground_truth:
            sample["answer"] = ground_truth[sample_id]["answer"]
            sample["is_impossible"] = ground_truth[sample_id]["is_impossible"]
            matched += 1
    
    logger.info(f"Merged ground truth for {matched}/{len(samples)} samples")
    if matched == 0 and len(samples) > 0 and len(ground_truth) > 0:
        logger.warning(f"No IDs matched! Test IDs type: {type(samples[0].get('id'))}, GT IDs type: {type(list(ground_truth.keys())[0])}")
    
    return samples


def sample_data(samples, ratio=1.0, seed=42):
    if ratio >= 1.0:
        return samples
    if ratio <= 0.0:
        return []
    np.random.seed(seed)
    n_samples = int(len(samples) * ratio)
    indices = np.random.choice(len(samples), size=n_samples, replace=False)
    sampled = [samples[i] for i in sorted(indices)]
    return sampled


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
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        is_fast = hasattr(tokenizer, "_tokenizer") or getattr(tokenizer, "is_fast", False)
        if not is_fast:
            logger.warning(f"{model_name} does not have fast tokenizer, using slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception as e:
        logger.warning(f"Could not load fast tokenizer for {model_name}: {e}, using slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config_model)
    
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
            self.current_epoch = -1
            self.last_train_loss = None
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch = state.epoch if hasattr(state, 'epoch') else len(self.history['eval_loss'])
        
        def on_log(self, args, state, control, **kwargs):
            logs = kwargs.get('logs', {})
            if 'loss' in logs:
                self.last_train_loss = logs['loss']
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if self.last_train_loss is not None:
                self.history['train_loss'].append(self.last_train_loss)
                self.last_train_loss = None
        
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get('metrics', {})
            epoch = state.epoch if hasattr(state, 'epoch') else len(self.history['eval_loss'])
            self.history['eval_loss'].append(metrics.get('eval_loss', 0))
            self.history['eval_f1'].append(metrics.get('eval_f1', 0))
            self.history['eval_exact'].append(metrics.get('eval_exact', 0))
    
    history_cb = HistoryCallback(training_history)
    trainer.add_callback(early_stopping_cb)
    trainer.add_callback(history_cb)
    
    train_result = trainer.train()
    
    eval_metrics = trainer.evaluate()
    
    eval_losses = training_history['eval_loss']
    eval_f1s = training_history['eval_f1']
    eval_ems = training_history['eval_exact']
    train_losses = training_history['train_loss']
    
    num_epochs = len(eval_losses)
    if num_epochs == 0:
        logger.warning(f"No evaluation metrics collected for {model_name}")
        return {'f1': 0, 'em': 0}
    
    epochs = list(range(1, num_epochs + 1))
    
    if len(train_losses) != num_epochs:
        logger.warning(f"Train losses count ({len(train_losses)}) != epochs ({num_epochs}), using eval loss for train loss")
        train_losses = eval_losses.copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(epochs, train_losses[:num_epochs], 'b-', label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, eval_losses, 'r-', label='Eval Loss', marker='s')
    axes[0, 0].set_title(f'{model_name.upper()} - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, eval_f1s, 'g-', label='Eval F1', marker='o')
    axes[0, 1].set_title(f'{model_name.upper()} - F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, eval_ems, 'm-', label='Eval EM', marker='o')
    axes[1, 0].set_title(f'{model_name.upper()} - Exact Match')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Exact Match')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, eval_f1s, 'g-', label='F1', marker='o')
    axes[1, 1].plot(epochs, eval_ems, 'm-', label='EM', marker='s')
    axes[1, 1].set_title(f'{model_name.upper()} - F1 vs EM')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    os.makedirs(config.output_dir, exist_ok=True)
    plot_path = os.path.join(config.output_dir, f"{model_name}_training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training history plot to {plot_path}")
    plt.close()
    
    train_metrics = {}
    if train_result and hasattr(train_result, 'metrics'):
        train_metrics = train_result.metrics
    
    train_metrics_result = {}
    try:
        train_predictions = trainer.predict(train_dataset)
        train_metrics_result = train_predictions.metrics if hasattr(train_predictions, 'metrics') else {}
    except Exception as e:
        logger.warning(f"Could not evaluate on training set: {e}")
    
    logger.info("\n" + "="*60)
    logger.info(f"{model_name.upper()} - SUMMARY STATISTICS")
    logger.info("="*60)
    
    logger.info("\n📊 TRAINING SET:")
    logger.info(f"  Loss: {train_metrics.get('train_loss', 'N/A'):.4f}" if isinstance(train_metrics.get('train_loss'), (int, float)) else f"  Loss: {train_metrics.get('train_loss', 'N/A')}")
    if train_metrics_result:
        logger.info(f"  F1 Score: {train_metrics_result.get('train_f1', 'N/A'):.4f}" if isinstance(train_metrics_result.get('train_f1'), (int, float)) else f"  F1 Score: {train_metrics_result.get('train_f1', 'N/A')}")
        logger.info(f"  Exact Match: {train_metrics_result.get('train_exact', 'N/A'):.4f}" if isinstance(train_metrics_result.get('train_exact'), (int, float)) else f"  Exact Match: {train_metrics_result.get('train_exact', 'N/A')}")
    logger.info(f"  Samples: {train_metrics.get('train_samples', 'N/A')}")
    logger.info(f"  Runtime: {train_metrics.get('train_runtime', 'N/A'):.2f}s" if isinstance(train_metrics.get('train_runtime'), (int, float)) else f"  Runtime: {train_metrics.get('train_runtime', 'N/A')}")
    
    logger.info("\n📊 VALIDATION SET (DEV):")
    logger.info(f"  Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}" if isinstance(eval_metrics.get('eval_loss'), (int, float)) else f"  Loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info(f"  F1 Score: {eval_metrics.get('eval_f1', 'N/A'):.4f}" if isinstance(eval_metrics.get('eval_f1'), (int, float)) else f"  F1 Score: {eval_metrics.get('eval_f1', 'N/A')}")
    logger.info(f"  Exact Match: {eval_metrics.get('eval_exact', 'N/A'):.4f}" if isinstance(eval_metrics.get('eval_exact'), (int, float)) else f"  Exact Match: {eval_metrics.get('eval_exact', 'N/A')}")
    if 'eval_HasAns_f1' in eval_metrics:
        logger.info(f"  HasAns F1: {eval_metrics.get('eval_HasAns_f1', 'N/A'):.4f}")
        logger.info(f"  HasAns Exact: {eval_metrics.get('eval_HasAns_exact', 'N/A'):.4f}")
    if 'eval_NoAns_f1' in eval_metrics:
        logger.info(f"  NoAns F1: {eval_metrics.get('eval_NoAns_f1', 'N/A'):.4f}")
        logger.info(f"  NoAns Exact: {eval_metrics.get('eval_NoAns_exact', 'N/A'):.4f}")
    logger.info(f"  Samples: {eval_metrics.get('eval_samples', 'N/A')}")
    
    test_results = {}
    if test_examples and predict_dataset:
        logger.info("\n📊 TEST SET:")
        try:
            results = trainer.predict(predict_dataset, test_examples)
            test_results = results.metrics
            
            logger.info(f"  Loss: {test_results.get('test_loss', 'N/A'):.4f}" if isinstance(test_results.get('test_loss'), (int, float)) else f"  Loss: {test_results.get('test_loss', 'N/A')}")
            logger.info(f"  F1 Score: {test_results.get('test_f1', 'N/A'):.4f}" if isinstance(test_results.get('test_f1'), (int, float)) else f"  F1 Score: {test_results.get('test_f1', 'N/A')}")
            logger.info(f"  Exact Match: {test_results.get('test_exact', 'N/A'):.4f}" if isinstance(test_results.get('test_exact'), (int, float)) else f"  Exact Match: {test_results.get('test_exact', 'N/A')}")
            if 'test_HasAns_f1' in test_results:
                logger.info(f"  HasAns F1: {test_results.get('test_HasAns_f1', 'N/A'):.4f}")
                logger.info(f"  HasAns Exact: {test_results.get('test_HasAns_exact', 'N/A'):.4f}")
            if 'test_NoAns_f1' in test_results:
                logger.info(f"  NoAns F1: {test_results.get('test_NoAns_f1', 'N/A'):.4f}")
                logger.info(f"  NoAns Exact: {test_results.get('test_NoAns_exact', 'N/A'):.4f}")
            logger.info(f"  Samples: {test_results.get('test_samples', 'N/A')}")
        except Exception as e:
            logger.error(f"Error evaluating test set: {e}")
            import traceback
            logger.error(traceback.format_exc())
            test_results = {}
    else:
        logger.info("\n📊 TEST SET: Not available")
    
    logger.info("\n" + "="*60)
    
    final_f1 = test_results.get('test_f1', eval_metrics.get('eval_f1', 0)) if test_results else eval_metrics.get('eval_f1', 0)
    final_em = test_results.get('test_exact', eval_metrics.get('eval_exact', 0)) if test_results else eval_metrics.get('eval_exact', 0)
    
    return {
        'f1': final_f1,
        'em': final_em,
        'train_f1': train_metrics_result.get('train_f1', 0),
        'train_em': train_metrics_result.get('train_exact', 0),
        'eval_f1': eval_metrics.get('eval_f1', 0),
        'eval_em': eval_metrics.get('eval_exact', 0),
        'test_f1': test_results.get('test_f1', 0),
        'test_em': test_results.get('test_exact', 0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['mbert', 'xlmr', 'roberta', 'all'],
                       default=None)
    parser.add_argument('--llm_zero_shot', action='store_true', help='Run zero-shot LLM baseline')
    parser.add_argument('--llm_few_shot', action='store_true', help='Run few-shot LLM baseline')
    parser.add_argument('--llm_models', type=str, nargs='+', choices=['youtu', 'qwen', 'all'], default=None)
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                       help='Ratio of data to sample for evaluation (0.0-1.0). Default: 1.0 (use all data)')
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
    if args.models:
        if 'all' in args.models:
            models_to_train = list(config.models.keys())
        else:
            models_to_train = args.models
    else:
        # CHỈ lấy từ config nếu KHÔNG có LLM flags
        if not (args.llm_zero_shot or args.llm_few_shot):
            models_to_train = config.models_to_train
        else:
            models_to_train = []  # Không train nếu chỉ chạy LLM
    
    valid_models = set(config.models.keys())
    models_to_train = [m for m in models_to_train if m in valid_models]
    
    results = {}
    if models_to_train:
        logger.info(f"Models to train: {[m.upper() for m in models_to_train]}")
        for model_name in models_to_train:
            try:
                result = train_model(model_name, config.models[model_name], config)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
    elif not (args.llm_zero_shot or args.llm_few_shot):
        logger.error(f"No valid models selected. Available: {list(config.models.keys())}")
        return
    
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
        comparison_path = os.path.join(config.output_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {comparison_path}")
        plt.close()
        
        logger.info("\n" + "="*60)
        logger.info("DETAILED RESULTS SUMMARY")
        logger.info("="*60)
        for model_name in model_names:
            r = results[model_name]
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Train - F1: {r.get('train_f1', 0):.4f}, EM: {r.get('train_em', 0):.4f}")
            logger.info(f"  Dev   - F1: {r.get('eval_f1', 0):.4f}, EM: {r.get('eval_em', 0):.4f}")
            logger.info(f"  Test  - F1: {r.get('test_f1', 0):.4f}, EM: {r.get('test_em', 0):.4f}")

    if args.llm_zero_shot or args.llm_few_shot:
        dev_dir = os.path.join(config.data_path, config.dev_dir)
        test_dir = os.path.join(config.data_path, config.test_dir)
        
        logger.info(f"Loading dev samples from: {dev_dir}")
        logger.info(f"Loading test samples from: {test_dir}")
        
        dev_samples = load_flat_samples(dev_dir)
        test_samples = load_flat_samples(test_dir)
        
        logger.info(f"Loaded {len(dev_samples)} dev samples, {len(test_samples)} test samples")
        
        ground_truth = load_ground_truth(test_dir)
        if ground_truth:
            logger.info(f"Loaded ground truth for {len(ground_truth)} test samples")
            test_samples = merge_ground_truth(test_samples, ground_truth)
            matched = sum(1 for s in test_samples if s.get("id") in ground_truth)
            logger.info(f"Matched {matched}/{len(test_samples)} test samples with ground truth")
            
            test_samples_with_gt = [s for s in test_samples if s.get("id") in ground_truth]
            logger.info(f"Filtering test set to only samples with ground truth: {len(test_samples_with_gt)} samples")
            test_samples = test_samples_with_gt
        else:
            logger.warning("No ground truth file found for test set. Test metrics may be inaccurate.")
            def has_valid_answer(s):
                ans = s.get("answer", "")
                if isinstance(ans, str):
                    return ans.strip() != ""
                elif isinstance(ans, list):
                    return len(ans) > 0
                return False
            has_answer = sum(1 for s in test_samples if has_valid_answer(s) or s.get("is_impossible", False))
            logger.info(f"Test samples with answers from test files: {has_answer}/{len(test_samples)}")
            if has_answer == 0:
                logger.warning("WARNING: No test samples have ground truth! Test metrics will be 0.")
        
        if args.sample_ratio < 1.0:
            logger.info(f"Sampling {args.sample_ratio*100:.1f}% of data for evaluation")
            dev_samples = sample_data(dev_samples, ratio=args.sample_ratio, seed=config.seed)
            test_samples = sample_data(test_samples, ratio=args.sample_ratio, seed=config.seed)
            logger.info(f"Dev samples: {len(dev_samples)}, Test samples: {len(test_samples)}")
            
            def has_valid_answer(s):
                ans = s.get("answer", "")
                if isinstance(ans, str):
                    return ans.strip() != ""
                elif isinstance(ans, list):
                    return len(ans) > 0
                return False
            has_answer_test = sum(1 for s in test_samples if has_valid_answer(s) or s.get("is_impossible", False))
            logger.info(f"Test samples with ground truth after sampling: {has_answer_test}/{len(test_samples)}")
        
        llm_models = args.llm_models
        if not llm_models or "all" in llm_models:
            llm_models = list(LLM_MAP.keys())
        for llm_key in llm_models:
            if llm_key not in LLM_MAP:
                logger.warning(f"Skipping unknown LLM key: {llm_key}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing LLM: {llm_key.upper()}")
            logger.info(f"{'='*60}")
            
            if args.llm_zero_shot:
                logger.info(f"\n[LLM] Loading {llm_key.upper()} model for zero-shot inference...")
                model, tokenizer, device = load_llm(llm_key)
                logger.info(f"[LLM] Model loaded. Device: {device}")
                
                logger.info(f"\n[INFERENCE] Running zero-shot on dev set")
                evaluate_llm_samples(model, tokenizer, device, llm_key, dev_samples, mode="zero", 
                                   output_dir=config.output_dir, split_name="dev")
                logger.info(f"[INFERENCE] Running zero-shot on test set")
                evaluate_llm_samples(model, tokenizer, device, llm_key, test_samples, mode="zero", 
                                   output_dir=config.output_dir, split_name="test")
            
            if args.llm_few_shot:
                if not args.llm_zero_shot:
                    logger.info(f"\n[LLM] Loading {llm_key.upper()} model for few-shot inference...")
                    model, tokenizer, device = load_llm(llm_key)
                    logger.info(f"[LLM] Model loaded. Device: {device}")
                
                logger.info(f"\n[INFERENCE] Running few-shot on dev set")
                evaluate_llm_samples(model, tokenizer, device, llm_key, dev_samples, mode="few", 
                                   output_dir=config.output_dir, split_name="dev")
                logger.info(f"[INFERENCE] Running few-shot on test set")
                evaluate_llm_samples(model, tokenizer, device, llm_key, test_samples, mode="few", 
                                   output_dir=config.output_dir, split_name="test")


if __name__ == "__main__":
    main()
