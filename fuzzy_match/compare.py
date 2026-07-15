import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nvidia/llama-embed-nemotron-8b",
    model_kwargs={"torch_dtype": "bfloat16"},
    trust_remote_code=True,
)


def fuzzy_dot(a: str, b: str) -> float:
    """
    Calculates the semantic distance between two strings using Cosine Distance.
    Returns a float where 1.0 means identical, and lower values mean less similar.
    """
    av: np.ndarray = model.encode(a)
    bv: np.ndarray = model.encode(b)

    # Because Sentence Transformers perfectly normalizes vectors to an L2 norm of 1,
    # the dot product IS the cosine similarity.
    dot_product = np.dot(av, bv)

    return float(dot_product)


def fuzzy_match(a: str, b: str, alpha: float) -> bool:
    return fuzzy_dot(a, b) > alpha


if __name__ == "__main__":
    from cases import cases

    for a, b in cases:
        print(f"dot('{a}', '{b}') = {fuzzy_dot(a, b):.4f}")
