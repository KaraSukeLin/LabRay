"""Shared information, native questions and fixed composition rules."""

from .data import LABELS

SUFFICIENCY = "Is the available information sufficient to answer the user's initial question Yes or No without a follow-up question?"
CHOICE = {
    "Yes": "The rules and supplied facts establish that the answer is yes.",
    "No": "The rules and supplied facts establish that the answer is no.",
    "Inquire": "A follow-up question is required because a necessary fact is missing.",
}
LEVELS = [
    "Definitely insufficient",
    "Probably insufficient",
    "Uncertain whether sufficient",
    "Probably sufficient",
    "Definitely sufficient",
]


def state(row):
    # An explicit allowlist excludes the rule split, IDs, gold, and evidence.
    history = "\n".join(f"Question: {h['follow_up_question']}\nAnswer: {h['follow_up_answer']}" for h in row["history"])
    return (
        f"Rule:\n{row['oracle_rule']}\n\nInitial question:\n{row['question']}\n\n"
        f"Scenario:\n{row['scenario']}\n\nConversation history:\n{history or '(none)'}"
    )


def questions(task, strategy):
    if task == "benchmark" and strategy == "single_choice":
        return {
            "decision": {
                "type": "choice",
                "instructions": "What decision follows from the rule and user information?",
                "criteria": CHOICE.copy(),
            }
        }
    if task == "benchmark" and strategy == "hierarchical":
        return {
            "sufficient": {"type": "noul", "instructions": SUFFICIENCY},
            "terminal": {
                "type": "choice",
                "instructions": "Assuming all required information is available, is the answer to the initial question Yes or No?",
                "criteria": {k: CHOICE[k] for k in ("Yes", "No")},
            },
        }
    if task == "benchmark" and strategy == "parallel_noul":
        return {k: {"type": "noul", "instructions": f"Is {k} the correct decision? {CHOICE[k]}"} for k in LABELS}
    if task == "sufficiency" and strategy == "noul":
        return {"sufficient": {"type": "noul", "instructions": SUFFICIENCY}}
    if task == "sufficiency" and strategy == "choice2":
        return {
            "sufficient": {
                "type": "choice",
                "instructions": SUFFICIENCY,
                "criteria": {
                    "insufficient": "A necessary fact is missing; ask a follow-up.",
                    "sufficient": "Enough information is available to answer Yes or No.",
                },
            }
        }
    if task == "sufficiency" and strategy == "score5":
        return {"sufficient": {"type": "score", "instructions": SUFFICIENCY, "criteria": LEVELS.copy()}}
    raise ValueError(f"unsupported task/strategy: {task}/{strategy}")


def keys(q):
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def target(label, task):
    return label if task == "benchmark" else "insufficient" if label == "Inquire" else "sufficient"


def aggregate(task, strategy, answers, threshold=0.5):
    detail = {}
    if task == "sufficiency":
        p = answers["sufficient"]
        if strategy == "score5":
            positive = sum(int(k) * v for k, v in p.items()) / 4
            detail["score_expectation"] = positive * 4
        else:
            positive = p["true" if strategy == "noul" else "sufficient"]
        result = {"insufficient": 1 - positive, "sufficient": positive}
        prediction = "sufficient" if positive >= threshold else "insufficient"
    elif strategy == "single_choice":
        result = answers["decision"]
        prediction = max(LABELS, key=result.__getitem__)
    elif strategy == "hierarchical":
        p = answers["sufficient"]["true"]
        terminal = answers["terminal"]
        result = {"Yes": p * terminal["Yes"], "No": p * terminal["No"], "Inquire": 1 - p}
        prediction = "Inquire" if p < threshold else max(("Yes", "No"), key=terminal.__getitem__)
        detail = {
            "sufficient_prediction": p >= threshold,
            "terminal_prediction": max(("Yes", "No"), key=terminal.__getitem__),
            "routing": "hard gate; both stages measured to retain complete joint probabilities",
        }
    else:
        support = {k: answers[k]["true"] for k in LABELS}
        positives = [k for k in LABELS if support[k] >= threshold]
        prediction = positives[0] if len(positives) == 1 else "Inquire"
        total = sum(support.values())
        result = {k: support[k] / total if total else 1 / 3 for k in LABELS}
        detail = {
            "conflict": len(positives) > 1,
            "no_positive": not positives,
            "positive_decisions": positives,
            "probability_semantics": "normalized independent support scores; heuristic, not an independence model",
        }
    return {"probabilities": result, "prediction": prediction, "confidence": result[prediction], "composition": detail}


def training_example(row, label, task, strategy):
    if strategy == "score5":
        raise ValueError("Score has no five-level gold labels; evaluate only")
    qs = questions(task, strategy)
    if strategy == "hierarchical":
        gold = {"sufficient": label != "Inquire"}
        if label != "Inquire":
            gold["terminal"] = label
        else:
            del qs["terminal"]  # No invented terminal label for an Inquire example.
    elif strategy == "parallel_noul":
        gold = {k: label == k for k in LABELS}
    elif task == "benchmark":
        gold = {"decision": label}
    else:
        gold = {"sufficient": label != "Inquire" if strategy == "noul" else target(label, task)}
    return {"example_id": row["example_id"], "state": state(row), "questions": qs, "gold": gold}
