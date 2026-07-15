import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Initialize the elite, high-precision technical cross-encoder on your RTX 5090
model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# This model utilizes full self-attention cross-encoding layers
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
)
model.eval()


def check_medical_synonyms(pairs: list) -> list:
    """
    Simultaneously pushes text pairs through cross-attention matrices.
    Applies an inline batch Min-Max scale to maximize the mathematical spread
    between correct and incorrect pairings across the current dataset.
    """
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(
        "cuda"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Pull raw un-bounded logits
        logits = outputs.logits.squeeze(-1).float()

        # --- BATCH MIN-MAX SCALING ---
        # Instead of risking flattening via Sigmoid, we calculate the absolute minimum
        # and maximum logit values across your active batch execution.
        min_logit = torch.min(logits)
        max_logit = torch.max(logits)

        # Prevent division by zero if all logits happen to be identical
        if max_logit == min_logit:
            scores = torch.ones_like(logits)
        else:
            # Stretches the highest logit cleanly to 1.0 and the lowest logit to 0.0
            scores = (logits - min_logit) / (max_logit - min_logit)

    return scores.cpu().tolist()


# =====================================================================
# Verification Array
# =====================================================================
if __name__ == "__main__":
    # Retaining your exact requested local import structure
    from cases import passing_cases

    results = check_medical_synonyms(passing_cases)

    for (concept, definition), score in zip(passing_cases, results):
        print(
            f"Concept: {concept:<14} | Target: {definition:<12} | Spread Score: {score:.4f}"
        )
