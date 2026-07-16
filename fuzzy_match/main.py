import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize Llama-3.1-8B-Instruct natively on your RTX 5090
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

# Pre-fetch the exact token IDs for "yes" and "no" inside the Llama-3 vocabulary
# We check both lowercase and capitalized variants to capture all probability mass
YES_IDS = [
    tokenizer.convert_tokens_to_ids("yes"),
    tokenizer.convert_tokens_to_ids("Yes"),
]
NO_IDS = [tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("No")]


def check_medical_synonyms(pairs: list) -> list:
    """
    Evaluates concept-definition pairs using a few-shot conversational template.
    Extracts the isolated token probability mass at the first prediction transition.
    """
    scores = []

    for concept, definition in pairs:
        # A few-shot prompt teaches the model exactly how to map histological terms
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise clinical vocabulary auditor. Your job is to verify if two medical "
                    "or scientific terms are accurate synonyms or descriptions of each other.\n"
                    "Examples:\n"
                    "Concept: basophilic | Target: blue -> yes\n"
                    "Concept: eosinophilic | Target: pink -> yes\n"
                    "Respond with exactly one word: yes or no."
                ),
            },
            {"role": "user", "content": f"Concept: {concept} | Target: {definition}"},
        ]

        # Use the official chat template formatting natively
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

            # Isolate the raw un-bounded logits from the final token transition layer
            next_token_logits = outputs.logits[0, -1, :].float()

            # Apply a global softmax to convert raw logits to true probability distributions
            probs = F.softmax(next_token_logits, dim=-1)

            # Sum up the probability weights assigned to yes vs no
            weight_yes = sum(probs[y_id].item() for y_id in YES_IDS)
            weight_no = sum(probs[n_id].item() for n_id in NO_IDS)

            # Normalize the mathematical spread strictly between the two valid outcomes
            total_mass = weight_yes + weight_no
            if total_mass == 0:
                score = 0.0
            else:
                score = weight_yes / total_mass

            scores.append(score)

    return scores


def fuzzy_score(a: str, b: str) -> float:
    """
    Returns a similarity score between 0.0 and 1.0 for two strings.
    This function is exported to Prolog as the fuzzy_score/2 predicate.
    """
    return check_medical_synonyms([(a, b)])[0]


def fuzzy_match(a: str, b: str, threshold: float) -> bool:
    """
    Returns True if the two strings are considered similar based on the model's evaluation.
    This function is exported to Prolog as the fuzzy_match/3 predicate, with a threshold parameter.
    """
    return fuzzy_score(a, b) >= threshold


# =====================================================================
# Verification Array
# =====================================================================
if __name__ == "__main__":
    # Natively reads your local cases file configuration
    from tests.cases import failing_cases, passing_cases

    print("")
    results = check_medical_synonyms(passing_cases)
    minimum, maximum = 99999.0, -99999.0
    minimum = min(score for score in results)
    maximum = max(score for score in results)
    print(f"Should pass (min={minimum:.4f}, max={maximum:.4f}):")
    scored_cases = sorted(
        zip(passing_cases, results), key=lambda item: item[1], reverse=True
    )
    for (concept, definition), score in scored_cases:
        print(
            f"Concept: {concept:<14} | Target: {definition:<12} | Spread Score: {score:.4f}"
        )

    print("")
    results = check_medical_synonyms(failing_cases)
    minimum, maximum = 99999.0, -99999.0
    minimum = min(score for score in results)
    maximum = max(score for score in results)
    print(f"Should fail (min={minimum:.4f}, max={maximum:.4f}):")
    scored_cases = sorted(
        zip(failing_cases, results), key=lambda item: item[1], reverse=True
    )
    for (concept, definition), score in scored_cases:
        print(
            f"Concept: {concept:<14} | Target: {definition:<12} | Spread Score: {score:.4f}"
        )
