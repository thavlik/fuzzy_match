import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def fuzzy_dot(a: str, b: str) -> float:
    """
    Calculates the semantic similarity between two strings using cosine similarity.
    Returns a float where 1.0 means identical, and lower values mean less similar.
    """
    encoded = tokenizer(
        [a, b],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=64,
    )

    with torch.no_grad():
        # MedCPT uses the final [CLS] hidden state as the query representation.
        embeddings = model(**encoded).last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    dot_product = np.dot(
        embeddings[0].cpu().numpy(),
        embeddings[1].cpu().numpy(),
    )

    return float(dot_product)


def fuzzy_match(a: str, b: str, alpha: float) -> bool:
    return fuzzy_dot(a, b) > alpha


if __name__ == "__main__":
    from cases import passing_cases

    for a, b in passing_cases:
        print(f"dot('{a}', '{b}') = {fuzzy_dot(a, b):.4f}")
