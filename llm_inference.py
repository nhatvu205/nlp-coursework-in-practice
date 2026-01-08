import os
import json
import random
import torch
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

LLM_MAP = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi2": "microsoft/phi-2"
}


def normalize_answer(s):
    import re

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def build_zero_shot_prompt(context, question):
    return (
        "Bạn là hệ thống trả lời câu hỏi tiếng Việt. "
        'Trả lời ngắn gọn dựa trên ngữ cảnh. Nếu không có đáp án trong ngữ cảnh, trả lời "Không có câu trả lời.".\n'
        f"Ngữ cảnh: {context}\n"
        f"Câu hỏi: {question}\n"
        "Trả lời:"
    )


def build_few_shot_prompt(context, question, shots):
    shot_blocks = []
    for s in shots:
        ans = s.get("answer", "Không có câu trả lời.")
        if s.get("is_impossible"):
            ans = "Không có câu trả lời."
        shot_blocks.append(
            f"Ngữ cảnh: {s['context']}\nCâu hỏi: {s['question']}\nTrả lời: {ans}\n"
        )
    shot_text = "\n".join(shot_blocks)
    return (
        "Bạn là hệ thống trả lời câu hỏi tiếng Việt. "
        'Trả lời ngắn gọn dựa trên ngữ cảnh. Nếu không có đáp án trong ngữ cảnh, trả lời "Không có câu trả lời.".\n'
        + shot_text
        + f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời:"
    )


def load_llm(model_key, device=None):
    model_id = LLM_MAP[model_key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if device == "cuda":
        model = model.to(device)
    model.eval()

    return model, tokenizer, device


def choose_shots(samples, k=3):
    if len(samples) <= k:
        return samples
    positives = [s for s in samples if not s.get("is_impossible")]
    negatives = [s for s in samples if s.get("is_impossible")]
    shot_list = []
    if positives:
        shot_list.append(random.choice(positives))
    if negatives:
        shot_list.append(random.choice(negatives))
    while len(shot_list) < k and samples:
        shot_list.append(random.choice(samples))
    return shot_list[:k]


def generate_answer(model, tokenizer, device, context, question, model_key, mode="zero", shots=None, max_new_tokens=64):
    if mode == "few" and shots:
        prompt_text = build_few_shot_prompt(context, question, shots)
    else:
        prompt_text = build_zero_shot_prompt(context, question)

    use_chat_template = model_key in ["qwen", "llama2", "mistral"]
    
    if use_chat_template:
        system_msg = (
            "Bạn là hệ thống Question Answering cho ViQuAD. "
            "Nhiệm vụ: trích xuất CHÍNH XÁC một cụm từ liên tiếp từ Context. "
            "Chỉ trả về CỤM TỪ TRẢ LỜI. "
            "KHÔNG viết câu hoàn chỉnh. "
            "KHÔNG giải thích. "
            "KHÔNG thêm ký tự nào khác."
        )
        
        if model_key == "qwen":
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text}
            ]
        elif model_key in ["llama2", "mistral"]:
            messages = [
                {"role": "user", "content": f"{system_msg}\n\n{prompt_text}"}
            ]
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = prompt_text
    else:
        prompt = prompt_text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    if device == "cuda":
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    input_len = input_ids.shape[1]

    with torch.no_grad():
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        
        outputs = model.generate(**generate_kwargs)

    generated_ids = outputs[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    answer = answer.strip().split("\n")[0]
    answer = answer.strip(" .,:;\"'")

    if "Trả lời:" in answer:
        answer = answer.split("Trả lời:")[-1].strip()

    return answer


def evaluate_llm_samples(model, tokenizer, device, model_key, samples, mode, output_dir, split_name, shots_per_prompt=3):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[INFERENCE] Starting {mode}-shot inference on {split_name} set ({len(samples)} samples)...")
    
    preds = []
    em_list = []
    f1_list = []

    pbar = tqdm(enumerate(samples), total=len(samples), desc=f"{model_key.upper()} {mode}-shot {split_name}", ncols=100)
    for i, ex in pbar:
        shots = None
        if mode == "few":
            shots = choose_shots(samples, k=shots_per_prompt)
        pred = generate_answer(
            model, tokenizer, device, ex["context"], ex["question"], model_key, mode=mode, shots=shots
        )
        truth = ex.get("answer") or ""
        if ex.get("is_impossible"):
            truth = ""
        em = compute_em(pred, truth)
        f1 = compute_f1(pred, truth)
        preds.append(
            {
                "id": ex.get("id", f"sample_{i}"),
                "prediction": pred,
                "ground_truth": truth,
                "is_impossible": ex.get("is_impossible", False),
            }
        )
        em_list.append(em)
        f1_list.append(f1)
        
        if (i + 1) % 10 == 0:
            avg_f1 = sum(f1_list) / len(f1_list)
            avg_em = sum(em_list) / len(em_list)
            pbar.set_postfix({"F1": f"{avg_f1:.3f}", "EM": f"{avg_em:.3f}"})
            pbar.refresh()
    
    pbar.close()

    metrics = {
        "f1": float(sum(f1_list) / len(f1_list)) if f1_list else 0.0,
        "em": float(sum(em_list) / len(em_list)) if em_list else 0.0,
        "count": len(preds),
    }

    logger.info(f"[INFERENCE] {split_name} set completed. F1: {metrics['f1']:.4f}, EM: {metrics['em']:.4f}")

    metrics_path = os.path.join(output_dir, f"llm_{model_key}_{mode}_{split_name}_metrics.json")
    preds_path = os.path.join(output_dir, f"llm_{model_key}_{mode}_{split_name}_preds.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["F1", "EM"], [metrics["f1"], metrics["em"]], color=["skyblue", "salmon"])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model_key.upper()} - {mode} shot ({split_name})")
    for j, v in enumerate([metrics["f1"], metrics["em"]]):
        ax.text(j, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"llm_{model_key}_{mode}_{split_name}_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return metrics, preds_path, metrics_path, plot_path


def evaluate_llm(model_key, samples, mode, output_dir, shots_per_prompt=3):
    logger.info(f"[LLM] Loading {model_key.upper()} model for inference (NOT training)...")
    model, tokenizer, device = load_llm(model_key)
    logger.info(f"[LLM] Model {model_key.upper()} loaded. Device: {device}")
    
    return evaluate_llm_samples(model, tokenizer, device, model_key, samples, mode, output_dir, "eval", shots_per_prompt)

