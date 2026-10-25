"""Pinned official data, oracle-rule inputs, sealed gold and nested samples."""

import collections
import itertools
import random
import urllib.request
from pathlib import Path

from .common import digest, read_json, rows, signature, write_json, write_rows

REVISION = "309c4ea3a7cc4c7a815d7f46292d91fc9a0de581"
SOURCE = "https://raw.githubusercontent.com/Yifan-Gao/open_retrieval_conversational_machine_reading"
COUNTS = {"train": 17936, "dev": 1105, "test": 2373}
LABELS = ["Yes", "No", "Inquire"]


def answer_label(answer):
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or answer.strip().casefold() in {"irrelevant", "unknown", "maybe"}
    ):
        raise ValueError(f"unexpected answer: {answer!r}")
    return {"yes": "Yes", "no": "No"}.get(answer.strip().casefold(), "Inquire")


def prepare(raw, out, download=False):
    raw, out = Path(raw), Path(out)
    names = ["id2snippet.json"] + [f"open_retrieval_sharc_{s}.json" for s in COUNTS]
    if download:
        raw.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not (raw / name).exists():
                with urllib.request.urlopen(f"{SOURCE}/{REVISION}/data/{name}", timeout=120) as response:
                    (raw / name).write_bytes(response.read())
    # Hashes come from the official pinned Git blobs, recorded during Phase 1.
    expected = read_json(Path(__file__).parents[2] / "configs" / "sources.json")["data_sha256"]
    hashes = {n: digest(raw / n) for n in names}
    if hashes != expected:
        raise ValueError("official data hash mismatch; do not silently accept changed data")
    if (out / "manifest.json").exists():
        validate_prepared(out)
        return read_json(out / "manifest.json")
    if out.exists() and any(out.iterdir()):
        raise ValueError("incomplete preparation exists; use a new output directory")
    snippets = read_json(raw / "id2snippet.json")
    splits = {s: rows(raw / f"open_retrieval_sharc_{s}.json") for s in COUNTS}
    if len(snippets) != 651 or {s: len(v) for s, v in splits.items()} != COUNTS:
        raise ValueError("official rule or split count mismatch")
    train_rules = {str(r["gold_snippet_id"]) for r in splits["train"]}
    ids, rules, trees, audit = {}, {}, {}, {}
    inputs, gold = {}, {}
    for split, records in splits.items():
        ids[split] = {r["utterance_id"] for r in records}
        if len(ids[split]) != len(records):
            raise ValueError(f"duplicate IDs in {split}")
        rules[split] = {str(r["gold_snippet_id"]) for r in records}
        trees[split] = {r["tree_id"] for r in records}
        inputs[split], gold[split] = [], []
        unexpected, labels, seen = collections.Counter(), collections.Counter(), collections.Counter()
        for r in records:
            answer = r["answer"]
            # Official decision evaluator maps every follow-up answer to 'more'.
            label = answer_label(answer)
            if label == "Inquire" and not answer.endswith("?"):
                unexpected[answer] += 1
            rule = str(r["gold_snippet_id"])
            if rule not in snippets:
                raise ValueError(f"missing oracle rule: {rule}")
            is_seen = rule in train_rules
            if split != "train" and (type(r.get("snippet_seen")) is not bool or r["snippet_seen"] != is_seen):
                raise ValueError(f"Seen/Unseen metadata disagrees with train rules: {r['utterance_id']}")
            history = [
                {"follow_up_question": h["follow_up_question"], "follow_up_answer": h["follow_up_answer"]}
                for h in r["history"]
            ]
            row = {
                "example_id": r["utterance_id"],
                "official_split": split,
                "tree_id": r["tree_id"],
                "rule_id": rule,
                "snippet_seen": r.get("snippet_seen"),
                "rule_split": "Seen" if is_seen else "Unseen",
                "source_url": r["source_url"],
                "oracle_rule": snippets[rule],
                "question": r["question"],
                "scenario": r["scenario"],
                "history": history,
            }
            inputs[split].append(row)
            gold[split].append({"example_id": row["example_id"], "gold": label, "original_answer": answer})
            labels[label] += 1
            seen[row["rule_split"]] += 1
        audit[split] = {
            "n": len(records),
            "labels": dict(labels),
            "rule_splits": dict(seen),
            "followups_without_question_mark": dict(unexpected),
            "rule_ids": sorted(rules[split]),
            "tree_count": len(trees[split]),
        }
    overlap = {}
    for a, b in itertools.combinations(COUNTS, 2):
        if ids[a] & ids[b]:
            raise ValueError(f"utterance IDs overlap: {a}/{b}")
        overlap[f"{a}/{b}"] = {"rule_ids": sorted(rules[a] & rules[b]), "tree_ids": sorted(trees[a] & trees[b])}
    order = sorted(ids["train"])
    random.Random(42).shuffle(order)
    dev_trees = sorted(trees["dev"], key=lambda t: signature([42, "dev-partition", t]))
    tune_trees = set(dev_trees[: len(dev_trees) // 2])
    partitions = {
        "train": order,
        "dev": [r["example_id"] for r in inputs["dev"]],
        "dev_select": [r["example_id"] for r in inputs["dev"] if r["tree_id"] in tune_trees],
        "dev_calibrate": [r["example_id"] for r in inputs["dev"] if r["tree_id"] not in tune_trees],
        "test": [r["example_id"] for r in inputs["test"]],
    }
    for n in (256, 1024, 4096):
        partitions[f"train_{n}"] = order[:n]
    train_gold = {r["example_id"]: r["gold"] for r in gold["train"]}
    demos = [next(i for i in order[:256] if train_gold[i] == label) for label in LABELS]
    demos.append(next(i for i in order[:256] if i not in demos))
    partitions["demos_4"] = demos
    for split in COUNTS:
        write_rows(out / "inputs" / f"{split}.jsonl", inputs[split])
        target = out / "sealed" / f"{split}.jsonl"
        write_rows(target, gold[split])
        target.chmod(0o600)
    write_json(out / "partitions.json", partitions)
    write_json(
        out / "audit.json",
        {
            "splits": audit,
            "overlap": overlap,
            "mapping": {"Yes": "Yes", "No": "No", "all other nonempty follow-up answers": "Inquire"},
            "oracle_context": True,
            "evidence_in_inputs": False,
            "seed": 42,
        },
    )
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {
        "source": SOURCE,
        "revision": REVISION,
        "raw_sha256": hashes,
        "files": files,
        "counts": COUNTS,
        "partition_counts": {k: len(v) for k, v in partitions.items()},
        "seed": 42,
    }
    write_json(out / "manifest.json", manifest)
    validate_prepared(out)
    return manifest


def validate_prepared(prepared, verify_gold=True):
    prepared = Path(prepared)
    m = read_json(prepared / "manifest.json")
    for name, sha in m["files"].items():
        if name.startswith("sealed/") and not verify_gold:
            continue
        if digest(prepared / name) != sha:
            raise ValueError(f"prepared data changed: {name}")
    p = read_json(prepared / "partitions.json")
    if not set(p["train_256"]) < set(p["train_1024"]) < set(p["train_4096"]) <= set(p["train"]):
        raise ValueError("training subsets are not nested")
    if set(p["dev_select"]) & set(p["dev_calibrate"]) or set(p["dev_select"] + p["dev_calibrate"]) != set(p["dev"]):
        raise ValueError("invalid development partitions")
    dev = {r["example_id"]: r for r in rows(prepared / "inputs/dev.jsonl")}
    if {dev[i]["tree_id"] for i in p["dev_select"]} & {dev[i]["tree_id"] for i in p["dev_calibrate"]}:
        raise ValueError("development tree leakage")
    for split in COUNTS:
        for r in rows(prepared / "inputs" / f"{split}.jsonl"):
            if {"answer", "gold", "evidence", "original_answer"} & r.keys():
                raise ValueError("label or evidence leaked into inference input")
    return m


def load_inputs(prepared, partition):
    p = read_json(Path(prepared) / "partitions.json")
    if partition not in p:
        raise ValueError(f"unknown partition: {partition}")
    split = (
        "train"
        if partition.startswith("train") or partition == "demos_4"
        else "dev"
        if partition.startswith("dev")
        else "test"
    )
    by_id = {r["example_id"]: r for r in rows(Path(prepared) / "inputs" / f"{split}.jsonl")}
    train = by_id if split == "train" else {r["example_id"]: r for r in rows(Path(prepared) / "inputs/train.jsonl")}
    exposed = {str(n): {train[i]["rule_id"] for i in p[f"train_{n}"]} for n in (256, 1024, 4096)}
    # Derived metadata only; preserve the prepared source bytes and official Seen split.
    return [
        {**by_id[i], "ft_exposed": {n: by_id[i]["rule_id"] in rules for n, rules in exposed.items()}}
        for i in p[partition]
    ]


def load_gold(prepared, split):
    return {r["example_id"]: r for r in rows(Path(prepared) / "sealed" / f"{split}.jsonl")}
