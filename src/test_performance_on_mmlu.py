import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

import torch
import torch.nn.functional as F
import argparse
from novice_io import load_disentangled, materialize_dense_model
import torch.nn as nn
from test_equality import load_model

REQUESTED_DEVICE = None

if REQUESTED_DEVICE is not None:
    DEVICE = REQUESTED_DEVICE
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if DEVICE == "cuda":
    TORCH_DTYPE = torch.bfloat16
elif DEVICE == "mps":
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32



# ---------------------------
# Utility: format MMLU prompt
# ---------------------------
def format_mmlu_question(question, choices):
    prompt = question.strip() + "\n\n"
    labels = ["A", "B", "C", "D"]
    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


# ---------------------------
# Compute log-likelihood of each option
# ---------------------------
def score_choice(model, tokenizer, prompt, choice_text, device):
    text = prompt + " " + choice_text

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


# ---------------------------
# Predict answer
# ---------------------------
def predict(model, tokenizer, question, choices, device):
    prompt = format_mmlu_question(question, choices)

    scores = []
    for choice in choices:
        score = score_choice(model, tokenizer, prompt, choice, device)
        scores.append(score)

    return int(torch.tensor(scores).argmax().item())


# ---------------------------
# Evaluate on subset of MMLU
# ---------------------------
def evaluate_mmlu(model, tokenizer, dataset, max_samples=500, device="cuda"):
    correct = 0
    total = 0

    for i, item in enumerate(dataset):
        if i % 1 == 0:
            print("Completed examples: ", i)

        if i >= max_samples:
            break

        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]  # 0-3

        pred = predict(model, tokenizer, question, choices, device)

        if pred == answer:
            correct += 1
        total += 1

    return correct / total


# ---------------------------
# MAIN COMPARISON FUNCTION
# ---------------------------
def compare_models_on_mmlu(model_a, model_b, tokenizer, max_samples=500, device="cpu"):
    dataset = load_dataset("cais/mmlu", "all", split="test")

    print("Evaluating base model...")
    acc_a = evaluate_mmlu(model_a, tokenizer, dataset, max_samples, device)

    print("Evaluating disentangled model...")
    acc_b = evaluate_mmlu(model_b, tokenizer, dataset, max_samples, device)

    print("\n===== MMLU RESULTS =====")
    print(f"Base model accuracy: {acc_a:.4f}")
    print(f"Disentangled model accuracy: {acc_b:.4f}")
    print(f"Difference     : {acc_b - acc_a:+.4f}")

    return acc_a, acc_b

from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", default = "Qwen/Qwen3.5-0.8B-Base", help="Base model to be compared, must be available from AutoModelForCausalLM")
    parser.add_argument("-d", "--disentangled", default = "src/data/models/novice_qwen_rf0.1.pt", help="Path to disentangled model: if not specified, disentangle base model")
    parser.add_argument("-r", "--rank_fraction", default=1.0, required=False, type=float, help="Fraction of ranks to keep")

    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = load_model(args.model)
    disentangled = load_disentangled(args.disentangled, device=DEVICE)
    materialize_dense_model(disentangled, rank_fraction=args.rank_fraction)

    compare_models_on_mmlu(base_model, disentangled, tokenizer, max_samples=100, device=DEVICE)



#NOTE: seems like inference takes a lot longer for disentangled model: figure out why
if __name__ == "__main__":
    main()
