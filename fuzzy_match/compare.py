import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "nvidia/llama-embed-nemotron-8b",
    model_kwargs={"torch_dtype": "bfloat16"},
    trust_remote_code=True,
)


def contrastive_similarity(
    target: str, candidate: str, baseline: str = "stain"
) -> float:
    # 1. Encode target, candidate, and a shared baseline keyword
    v_target = model.encode(target)
    v_candidate = model.encode(candidate)
    v_baseline = model.encode(baseline)

    # 2. Subtract the baseline noise to isolate the true distinguishing variance
    v_target_isolated = v_target - v_baseline
    v_candidate_isolated = v_candidate - v_baseline

    # 3. Manually re-normalize to unit lengths
    v_target_isolated /= np.linalg.norm(v_target_isolated)
    v_candidate_isolated /= np.linalg.norm(v_candidate_isolated)

    # 4. Compute the corrected dot product
    return float(np.dot(v_target_isolated, v_candidate_isolated))


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
    from cases import passing_cases

    for a, b in passing_cases:
        print(f"dot('{a}', '{b}') = {fuzzy_dot(a, b):.4f}")
