import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Initialize the elite cross-encoder setup on your RTX 5090
model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# This model utilizes a built-in Classification head (Sequence Classification)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()


def check_medical_synonyms(pairs: list) -> list:
    """
    Pushes text pairs through cross-attention matrices.
    Applies temperature scaling and empirical bias correction to translate
    BGE's non-linear negative logits into a wide, high-contrast 0.0 to 1.0 range.
    """
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(
        "cuda"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Pull raw un-bounded logits (typically ranging from -8.0 to 0.0 for short strings)
        logits = outputs.logits.squeeze(-1).float()

        # --- LOGIT CALIBRATION GRAPH ---
        # BGE-M3 uses a strict negative logit boundary. Short, weak matches score around -6.0.
        # Strong, exact matches score around -3.0 or higher.
        # We subtract an empirical baseline bias (-4.5) and apply a sharp temperature division (0.45)
        # to aggressively stretch the mathematical variance between true and false assertions.
        bias_threshold = -4.5
        temperature = 0.45

        calibrated_logits = (logits - bias_threshold) / temperature

        # Force the output to bound safely between 0.0 and 1.0
        scores = torch.sigmoid(calibrated_logits)

    return scores.cpu().tolist()


# =====================================================================
# Verification Array
# =====================================================================
if __name__ == "__main__":
    from cases import passing_cases

    results = check_medical_synonyms(passing_cases)

    for (concept, definition), score in zip(passing_cases, results):
        print(
            f"Concept: {concept:<14} | Target: {definition:<12} | Spread Score: {score:.4f}"
        )
