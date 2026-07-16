import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a true instruction model (8B fits easily into your 29GB VRAM)
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)


def get_alignment_score(concept: str, definition: str) -> float:
    """
    Asks the LLM if 'definition' is logically true for 'concept'.
    Returns the exact probability (0.0 to 1.0) of a 'Yes' token.
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"Does the medical term '{concept}' accurately describe or mean '{definition}'? "
        f"Answer with exactly one word, Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the logits for the very next predicted token
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        # Get token IDs for "Yes" and "No"
        yes_id = tokenizer.convert_tokens_to_ids("Yes")
        no_id = tokenizer.convert_tokens_to_ids("No")

        # Normalize the probabilities between just Yes and No
        yes_prob = probs[yes_id]
        no_prob = probs[no_id]

        score = yes_prob / (yes_prob + no_prob)
        return float(score)


if __name__ == "__main__":
    from cases import passing_cases

    for a, b in passing_cases:
        print(f"get_alignment_score('{a}', '{b}') = {get_alignment_score(a, b):.4f}")
