import re
from dataclasses import dataclass

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

RELATION_LABELS = {
    "equivalent": "A",
    "contradictory": "B",
    "unrelated": "C",
}
CONTRADICTION_VETO_THRESHOLD = 0.5
CONTRADICTION_CUE_PATTERN = re.compile(
    r"\b(?:no|not|without|absent|denies|negative\s+(?:for|test|result)|"
    r"reduces?|decreases?|increases?|clear|normal|hard|infrequent|only|as\s+needed)\b",
    re.IGNORECASE,
)


def get_label_token_id(label: str) -> int:
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(
            f"Relation label {label!r} is not a single token: {token_ids}"
        )
    return token_ids[0]


RELATION_TOKEN_IDS = {
    relation: get_label_token_id(label) for relation, label in RELATION_LABELS.items()
}


@dataclass(frozen=True)
class RelationEvaluation:
    relation_logits: dict[str, float]
    relation_probabilities: dict[str, float]
    contradiction_veto_enabled: bool
    label_probability_mass: float
    top_token: str
    top_token_id: int
    top_token_probability: float

    @property
    def predicted_relation(self) -> str:
        if (
            self.contradiction_veto_enabled
            and self.relation_probabilities["contradictory"]
            >= CONTRADICTION_VETO_THRESHOLD
        ):
            return "contradictory"
        return max(
            ("equivalent", "unrelated"),
            key=self.relation_probabilities.get,
        )

    @property
    def score(self) -> float:
        if self.predicted_relation != "equivalent":
            return 0.0
        non_contradictory_mass = (
            self.relation_probabilities["equivalent"]
            + self.relation_probabilities["unrelated"]
        )
        return self.relation_probabilities["equivalent"] / non_contradictory_mass


CONTRADICTION_SYSTEM_PROMPT = """You are a precise clinical vocabulary auditor. Decide only whether a medical concept and target contradict each other.

B = CONTRADICTORY: They describe opposite or mutually incompatible clinical states.
A = NOT CONTRADICTORY: They are equivalent, compatible, or unrelated.

A concept naming a condition asserts that it is present unless the concept itself expresses absence. Compare both the clinical idea and its assertion polarity. If one expression affirms an idea and the other denies or reverses that same idea, choose B even when every other medical word overlaps. Words such as no, not, without, absent, denies, negative, increased, and decreased can reverse the relationship. A negated expression can be compatible when the concept itself expresses absence, such as afebrile and without fever. Unrelated expressions are N, not B.

Respond with exactly one label: A or B."""

EQUIVALENCE_SYSTEM_PROMPT = """You are a precise clinical vocabulary auditor. Decide whether a medical or scientific concept and target name or accurately describe the same idea.

A = EQUIVALENT: The target may be a synonym, plain-language description, visual description, morphology, stain color, clinical shorthand, or functional definition of the concept. Do not require identical wording or taxonomic precision when the target accurately conveys the intended medical feature.
C = UNRELATED: The expressions do not describe the same clinical or scientific idea. Mere medical co-occurrence, treatment relationships, causes, or anatomical proximity are not equivalence.

Assume contradiction has already been checked. Respond with exactly one label: A or C."""

CONTRADICTION_DEMONSTRATIONS = [
    ("peripheral edema", "no limb swelling", "B"),
    ("afebrile", "fever present", "B"),
    ("diuretic", "reduces urine production", "B"),
    ("bacteremia", "bloodstream without bacteria", "B"),
    ("peripheral edema", "limb swelling", "A"),
    ("afebrile", "without a fever", "A"),
    ("epistaxis", "nosebleed", "A"),
    ("epistaxis", "joint pain", "A"),
    ("eosinophilic", "pinkish red", "A"),
    ("basophilic", "blue-purple", "A"),
    ("hypertonic", "tight muscle", "A"),
    ("multicolored bruises", "bruises in different healing stages", "A"),
    ("ovoid cocci in chains", "chains of elliptical spheres", "A"),
]

EQUIVALENCE_DEMONSTRATIONS = [
    ("comma-shaped", "curved rod", "A"),
    ("eosinophilic", "red", "A"),
    ("eosinophilic", "pinkish red", "A"),
    ("basophilic", "blue", "A"),
    ("basophilic", "blue-purple", "A"),
    ("hypertonic", "tight muscle", "A"),
    ("multicolored bruises", "bruises in different healing stages", "A"),
    ("ovoid cocci in chains", "chains of elliptical spheres", "A"),
    ("gram-negative bacilli", "pink-staining rod-shaped bacteria", "A"),
    ("epistaxis", "joint pain", "C"),
    ("magnetic resonance imaging", "antibiotic medication", "C"),
    ("bradycardia", "skin rash", "C"),
    ("peripheral edema", "kidney stone", "C"),
    ("diuretic", "magnetic resonance imaging", "C"),
    ("bacteremia", "joint dislocation", "C"),
    ("fracture", "nasal congestion", "C"),
]


def build_messages(
    system_prompt: str,
    demonstrations: list[tuple[str, str, str]],
    concept: str,
    definition: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for example_concept, example_definition, label in demonstrations:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Concept: {example_concept} | Target: {example_definition}"
                    ),
                },
                {"role": "assistant", "content": label},
            ]
        )
    messages.append(
        {"role": "user", "content": f"Concept: {concept} | Target: {definition}"}
    )
    return messages


def evaluate_stage(
    concept: str,
    definition: str,
    system_prompt: str,
    demonstrations: list[tuple[str, str, str]],
    labels: dict[str, int],
) -> tuple[dict[str, float], dict[str, float], float]:
    prompt = tokenizer.apply_chat_template(
        build_messages(system_prompt, demonstrations, concept, definition),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :].float()
        vocabulary_probabilities = F.softmax(next_token_logits, dim=-1)
        label_logits_tensor = torch.stack(
            [next_token_logits[token_id] for token_id in labels.values()]
        )
        label_probabilities_tensor = F.softmax(label_logits_tensor, dim=0)

    label_logits = {
        label: label_logits_tensor[index].item() for index, label in enumerate(labels)
    }
    label_probabilities = {
        label: label_probabilities_tensor[index].item()
        for index, label in enumerate(labels)
    }
    label_probability_mass = sum(
        vocabulary_probabilities[token_id].item() for token_id in labels.values()
    )
    return label_logits, label_probabilities, label_probability_mass


def evaluate_medical_relations(
    pairs: list[tuple[str, str]],
) -> list[RelationEvaluation]:
    """
    Classifies concept-definition pairs and retains diagnostics from the first
    prediction transition.
    """
    evaluations = []

    for concept, definition in pairs:
        contradiction_veto_enabled = bool(
            CONTRADICTION_CUE_PATTERN.search(f"{concept} {definition}")
        )
        if contradiction_veto_enabled:
            contradiction_logits, contradiction_probabilities, contradiction_mass = (
                evaluate_stage(
                    concept,
                    definition,
                    CONTRADICTION_SYSTEM_PROMPT,
                    CONTRADICTION_DEMONSTRATIONS,
                    {
                        "contradictory": RELATION_TOKEN_IDS["contradictory"],
                        "not_contradictory": RELATION_TOKEN_IDS["equivalent"],
                    },
                )
            )
        else:
            contradiction_logits = {
                "contradictory": float("-inf"),
                "not_contradictory": 0.0,
            }
            contradiction_probabilities = {
                "contradictory": 0.0,
                "not_contradictory": 1.0,
            }
            contradiction_mass = 1.0
        equivalence_logits, equivalence_probabilities, equivalence_mass = (
            evaluate_stage(
                concept,
                definition,
                EQUIVALENCE_SYSTEM_PROMPT,
                EQUIVALENCE_DEMONSTRATIONS,
                {
                    "equivalent": RELATION_TOKEN_IDS["equivalent"],
                    "unrelated": RELATION_TOKEN_IDS["unrelated"],
                },
            )
        )

        contradictory_probability = contradiction_probabilities["contradictory"]
        not_contradictory_probability = 1.0 - contradictory_probability
        relation_probabilities = {
            "equivalent": not_contradictory_probability
            * equivalence_probabilities["equivalent"],
            "contradictory": contradictory_probability,
            "unrelated": not_contradictory_probability
            * equivalence_probabilities["unrelated"],
        }
        predicted_relation = max(
            ("equivalent", "unrelated"), key=relation_probabilities.get
        )
        if (
            contradiction_veto_enabled
            and contradictory_probability >= CONTRADICTION_VETO_THRESHOLD
        ):
            predicted_relation = "contradictory"
        evaluations.append(
            RelationEvaluation(
                relation_logits={
                    "equivalent": equivalence_logits["equivalent"],
                    "contradictory": contradiction_logits["contradictory"],
                    "unrelated": equivalence_logits["unrelated"],
                },
                relation_probabilities=relation_probabilities,
                contradiction_veto_enabled=contradiction_veto_enabled,
                label_probability_mass=min(contradiction_mass, equivalence_mass),
                top_token=RELATION_LABELS[predicted_relation],
                top_token_id=RELATION_TOKEN_IDS[predicted_relation],
                top_token_probability=relation_probabilities[predicted_relation],
            )
        )

    return evaluations


def check_medical_synonyms(pairs: list[tuple[str, str]]) -> list[float]:
    return [evaluation.score for evaluation in evaluate_medical_relations(pairs)]


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

    evaluations = evaluate_medical_relations(passing_cases)
    minimum = min(evaluation.score for evaluation in evaluations)
    maximum = max(evaluation.score for evaluation in evaluations)
    scored_cases = sorted(
        zip(passing_cases, evaluations),
        key=lambda item: item[1].score,
        reverse=True,
    )
    threshold = 0.5
    print(f"Score threshold: {threshold}\n")
    total_passing = sum(evaluation.score >= threshold for evaluation in evaluations)
    print(
        f"Positive examples ({total_passing}/{len(passing_cases)} passing, min={minimum:.4f}, max={maximum:.4f}):"
    )
    min_passing = minimum
    for (concept, definition), evaluation in scored_cases:
        probabilities = evaluation.relation_probabilities
        logits = evaluation.relation_logits
        print(
            f"Concept: {concept:<24} | Target: {definition:<40} "
            # f"| Relation: {evaluation.predicted_relation:<13} "
            f"| Score: {evaluation.score:.4f} "
            f"| Pass: {evaluation.score >= threshold} "
            # f"| P(A/B/C): {probabilities['equivalent']:.4f}/"
            # f"{probabilities['contradictory']:.4f}/{probabilities['unrelated']:.4f} "
            # f"| Logits(A/B/C): {logits['equivalent']:.2f}/"
            # f"{logits['contradictory']:.2f}/{logits['unrelated']:.2f} "
            # f"| Label mass: {evaluation.label_probability_mass:.4f} "
            # f"| Top token: {evaluation.top_token!r} "
            # f"(id={evaluation.top_token_id}, p={evaluation.top_token_probability:.4f})"
        )

    print("")
    evaluations = evaluate_medical_relations(failing_cases)
    minimum = min(evaluation.score for evaluation in evaluations)
    maximum = max(evaluation.score for evaluation in evaluations)
    max_failing = maximum
    scored_cases = sorted(
        zip(failing_cases, evaluations),
        key=lambda item: item[1].score,
        reverse=True,
    )
    total_failing = sum(evaluation.score < threshold for evaluation in evaluations)
    print(
        f"Negative examples ({total_failing}/{len(failing_cases)} passing, min={minimum:.4f}, max={maximum:.4f}):"
    )
    for (concept, definition), evaluation in scored_cases:
        probabilities = evaluation.relation_probabilities
        logits = evaluation.relation_logits
        # print(
        #    f"Concept: {concept:<24} | Target: {definition:<40} "
        #    f"| Relation: {evaluation.predicted_relation:<13} "
        #    f"| Score: {evaluation.score:.4f} "
        #    f"| P(A/B/C): {probabilities['equivalent']:.4f}/"
        #    f"{probabilities['contradictory']:.4f}/{probabilities['unrelated']:.4f} "
        #    f"| Logits(A/B/C): {logits['equivalent']:.2f}/"
        #    f"{logits['contradictory']:.2f}/{logits['unrelated']:.2f} "
        #    f"| Label mass: {evaluation.label_probability_mass:.4f} "
        #    f"| Top token: {evaluation.top_token!r} "
        #    f"(id={evaluation.top_token_id}, p={evaluation.top_token_probability:.4f})"
        # )
        print(
            f"Concept: {concept:<24} | Target: {definition:<40} "
            f"| Score: {evaluation.score:.4f} "
            f"| Pass: {evaluation.score >= threshold} "
        )
    print(
        f"\nTotal matching errors: {len(passing_cases) + len(failing_cases) - total_passing - total_failing}"
    )
    print(f"Minimum passing score: {min_passing:.4f}")
    print(f"Maximum failing score: {max_failing:.4f}")
    print(f"Spread: {min_passing - max_failing:.4f}")
