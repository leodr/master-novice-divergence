import torch
import torch.nn.functional as F
import argparse
from transformers import AutoModelForCausalLM
from novice_model import NoviceModel

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

import torch
import torch.nn.functional as F

# -------------------------
# KL DEBUG CHECKS
# -------------------------


def get_synthetic_batch(batch_size=4, seq_len=32, vocab_size=1000, seed=0):
    torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }


def extract_outputs(model, batch):
    with torch.no_grad():
        outputs = model(
            **batch,
            output_attentions=True,
            output_hidden_states=True,   # IMPORTANT for hidden states
            return_dict=True
        )

    # ✅ Decoder-only models (Qwen/GPT/LLaMA): no last_hidden_state
    hidden = None
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        hidden = outputs.hidden_states[-1]  # last layer hidden state

    logits = getattr(outputs, "logits", None)
    attentions = getattr(outputs, "attentions", None)

    return hidden, logits, attentions


def assert_models_equivalent(
    model_a,
    model_b,
    atol_hidden=1e-5,
    atol_logits=1e-4,
    cosine_threshold=0.999,
    attention_cosine_threshold=0.999,
    check_attention=False
):
    """
    Compares two models for functional equivalence.
    """

    model_a.eval()
    model_b.eval()

    batch = get_synthetic_batch()
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    h_a, l_a, attn_a = extract_outputs(model_a, batch)
    h_b, l_b, attn_b = extract_outputs(model_b, batch)

    results = {}
    passed = True

    # -------------------------
    # 1. Hidden state check
    # -------------------------
    if h_a is not None and h_b is not None:
        hidden_mse = F.mse_loss(h_a, h_b).item()
        results["hidden_mse"] = hidden_mse

        if hidden_mse > atol_hidden:
            print(f"Hidden states differ too much: {hidden_mse:.6g}, threshold: {atol_hidden}")
            passed = False

   # -------------------------
    # 2. Logit similarity (SAFE KL divergence)
    # -------------------------
    if l_a is not None and l_b is not None:

        # ensure numerical stability
        l_a = l_a.float()
        l_b = l_b.float()

        # P || Q
        log_probs_a = F.log_softmax(l_a, dim=-1)
        probs_b = F.softmax(l_b, dim=-1)

        kl_ab = F.kl_div(
            log_probs_a,
            probs_b,
            reduction="batchmean",
            log_target=False
        ).item()

        results["logit_kl"] = kl_ab

        if kl_ab > atol_logits:
            print(f"Logits KL too large: {kl_ab:.6g}, threshold: {atol_logits}")
            passed = False

        # -------------------------
        # Optional: symmetric KL (more strict)
        # -------------------------
        log_probs_b = F.log_softmax(l_b, dim=-1)
        probs_a = F.softmax(l_a, dim=-1)

        kl_ba = F.kl_div(
            log_probs_b,
            probs_a,
            reduction="batchmean",
            log_target=False
        ).item()

        results["logit_symmetric_kl"] = 0.5 * (kl_ab + kl_ba)
    # -------------------------
    # 3. Attention check
    # -------------------------
    if check_attention and attn_a is not None and attn_b is not None:
        attn_diffs = []

        for a, b in zip(attn_a, attn_b):
            attn_diffs.append((a - b).abs().mean().item())

        attn_diff = sum(attn_diffs) / len(attn_diffs)
        results["raw_attention_diff"] = attn_diff

        a = torch.cat([x.flatten() for x in attn_a])
        b = torch.cat([x.flatten() for x in attn_b])

        attn_cos = F.cosine_similarity(a, b, dim=0).item()
        results["attention_cosine"] = attn_cos

        if attn_cos < attention_cosine_threshold:
            print(f"Attention not similar enough: {attn_cos}, threshold: {attention_cosine_threshold}")
            passed = False

    # -------------------------
    # Summary
    # -------------------------
    if passed:
        print("Models are equivalent within tolerance")
    else:
        print("Models are NOT equivalent within tolerance")

    for k, v in results.items():
        print(f"{k}: {v:.6g}")

    return results


def load_model(model_name):
    load_kwargs = {
        "dtype": TORCH_DTYPE,
        "trust_remote_code": True,
    }
    if DEVICE == "cuda":
        load_kwargs["device_map"] = "cuda"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()
    model.config.use_cache = False
    return model

def test_equality(model, disentangled_model):
    results = assert_models_equivalent(model, disentangled_model)
    print(results)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="Base model to be compared, must be available from AutoModelForCausalLM")
    parser.add_argument("-d", "--disentangled", required=True, help="Path to disentangled model: if not specified, disentangle base model")
    parser.add_argument("-r", "--rank_fraction", default=1.0, required=False, type=float, help="Fraction of ranks to keep")
    parser.add_argument("-c", "--check_attention", default=False, required=False, type=bool, help="Whether to check attention during test")

    return parser.parse_args()

def main():
    args = parse_args()
    base_model = load_model(args.model)
    novice = NoviceModel.from_checkpoint(args.disentangled, device=DEVICE)
    baked = novice.bake(args.rank_fraction, in_place=True)
    test_equality(base_model, baked)



if __name__ == "__main__":
    main()
