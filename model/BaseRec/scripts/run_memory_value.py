#!/usr/bin/env python3
"""Compare candidate-only, raw history and frozen user memory without retraining.

Point --memories to final_memories.json from an existing BaseRec training run.
All conditions share the same frozen LLM ranker and candidate permutations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.history_formatter import HISTORY_MODES, format_history
from src.models.base_rec import BaseRec
from scripts.run_baserec import (
    CompatibleLLM, atomic_json, canonical_json, evaluation_metrics,
    load_inputs, output_lock, retry_transaction, sha256_file,
)

MODES = (
    "candidate_only", "history_titles", "history_full",
    "history_structured", "real_memory", "shuffled_memory",
)
DEFAULT_MODES = (
    "candidate_only", "history_titles", "history_full",
    "real_memory", "shuffled_memory",
)


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(x.strip() for x in value.split(",") if x.strip())
    if not modes or len(set(modes)) != len(modes) or any(x not in MODES for x in modes):
        raise ValueError(f"Invalid --modes: choose unique modes from {MODES}")
    return modes


def donor_mapping(users: list[int], seed: int) -> dict[int, int]:
    """Deterministic derangement; no user receives their own memory."""
    if len(users) < 2:
        raise ValueError("Shuffled-memory control needs at least two users")
    shuffled_users = list(users)
    rng = random.Random(seed)
    rng.shuffle(shuffled_users)
    shift = rng.randrange(1, len(users))
    return {
        user: shuffled_users[(i + shift) % len(users)]
        for i, user in enumerate(shuffled_users)
    }


def context_for(mode: str, user: int, histories: dict, metadata: dict,
                memories: dict[int, str], donors: dict[int, int]) -> str:
    if mode == "candidate_only":
        return ""
    if mode == "real_memory":
        return memories[user]
    if mode == "shuffled_memory":
        return memories[donors[user]]
    if mode in HISTORY_MODES:
        # The final history element is always held out for evaluation.
        return format_history(histories[user][:-1], metadata, mode)
    raise ValueError(f"Invalid mode: {mode}")


def audit_history_coverage(metadata: dict, histories: dict) -> dict:
    """Summarize field availability before making any paid LLM calls."""
    from src.models.base_rec import positive_item_evidence, sanitize_item_description
    total, titles, descriptions, structured = 0, 0, 0, 0
    active_users, users_with_description = 0, 0
    repeated_targets = 0
    for history in histories.values():
        prefix = history[:-1]
        if prefix:
            active_users += 1
        repeated_targets += int(history[-1] in prefix)
        desc_found = False
        for item_id in prefix:
            total += 1
            facts = metadata[item_id]
            title = sanitize_item_description(facts.get("title") or facts.get("name"))
            description = sanitize_item_description(facts.get("description"))
            titles += int(isinstance(title, str) and bool(title))
            has_desc = isinstance(description, str) and bool(description)
            descriptions += int(has_desc)
            desc_found = desc_found or has_desc
            structured += int(bool(positive_item_evidence(facts)))
        users_with_description += int(desc_found)
    return {
        "training_interactions": total, "users_with_training_history": active_users,
        "interactions_with_title": titles, "interactions_with_description": descriptions,
        "interactions_with_structured_evidence": structured,
        "users_with_training_description": users_with_description,
        "users_with_heldout_item_previously_seen": repeated_targets,
        "history_full_identical_to_titles_for_users": len(histories) - users_with_description,
    }


def protocol(args, modes: tuple[str, ...]) -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = {
        "experiment": "memory-value-v1",
        "input_sha256": sha256_file(args.input),
        "memories_sha256": sha256_file(args.memories),
        "model": args.model,
        "endpoint": args.endpoint.rstrip("/"),
        "modes": list(modes),
        "seed": args.seed,
        "permutations": args.permutations,
        "include_candidate_description": args.include_candidate_description,
        "allow_repair": args.allow_repair,
        "allow_unverified_memories": args.allow_unverified_memories,
        "max_generation_tokens": args.max_generation_tokens,
        "source_sha256": {
            str(p.relative_to(root)): sha256_file(p)
            for p in (
                root / "src/history_formatter.py",
                root / "src/models/base_rec.py",
                root / "scripts/run_baserec.py",
                root / "scripts/run_memory_value.py",
            )
        },
    }
    return {"protocol": cfg, "fingerprint": hashlib.sha256(canonical_json(cfg)).hexdigest()}


def execute(args, llm=None) -> dict:
    modes = parse_modes(args.modes) if isinstance(args.modes, str) else tuple(args.modes)
    if not modes or any(m not in MODES for m in modes) or len(set(modes)) != len(modes):
        raise ValueError("Invalid modes")
    if args.permutations < 1:
        raise ValueError("At least one candidate permutation is required")
    metadata, histories, candidates = load_inputs(args.input)
    raw_memories = json.loads(args.memories.read_text(encoding="utf-8"))
    if not isinstance(raw_memories, dict) or set(raw_memories) != {str(u) for u in histories}:
        raise ValueError("final_memories.json must contain precisely the evaluation users")
    if any(not isinstance(v, str) for v in raw_memories.values()):
        raise ValueError("Memory values must be strings")

    # Fail closed for published results: a bare JSON memory snapshot is not proof
    # of which history/labels were used to build it.
    training_manifest = args.memories.parent / "manifest.json"
    snapshot_verified = training_manifest.exists()
    if not snapshot_verified and not args.allow_unverified_memories:
        raise ValueError(
            "Memory snapshot has no sibling manifest.json; provenance cannot be verified. "
            "For exploratory/legacy memory files only, use --allow-unverified-memories."
        )
    if snapshot_verified:
        original = json.loads(training_manifest.read_text(encoding="utf-8"))
        if original.get("protocol", {}).get("input_sha256") != sha256_file(args.input):
            raise ValueError("Memory snapshot was built from a different input dataset")

    input_audit = audit_history_coverage(metadata, histories)
    if "history_full" in modes and input_audit["interactions_with_description"] == 0:
        raise ValueError(
            "history_full has no usable descriptions in the training history; "
            "it would be identical to history_titles. Supply proper description metadata "
            "or remove history_full from --modes."
        )
    if "history_titles" in modes and input_audit["interactions_with_title"] == 0:
        raise ValueError("history_titles has no usable titles in the training history")

    memories = {u: raw_memories[str(u)] for u in histories}
    user_ids = list(histories)
    donors = donor_mapping(user_ids, args.seed) if "shuffled_memory" in modes else {}
    rng = random.Random(args.seed)
    fixed_orders = {}
    for user in user_ids:
        for permutation in range(args.permutations):
            order = list(candidates[user])
            rng.shuffle(order)
            fixed_orders[user, permutation] = order

    llm = llm or CompatibleLLM(
        args.endpoint, args.model, os.environ.get(args.api_key_env, ""), args.timeout
    )
    model = BaseRec(llm, metadata, user_ids, max_generation_tokens=args.max_generation_tokens)
    model.user_memories.update(memories)
    model.freeze()
    manifest = protocol(args, modes)
    jobs = [(user, permutation, mode)
            for user in user_ids
            for permutation in range(args.permutations)
            for mode in modes]

    output = args.output
    with output_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not args.resume or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Output already exists; use --resume with identical inputs")
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("Output directory is not empty")
            atomic_json(manifest_path, manifest)

        records_dir = output / "records"
        records_dir.mkdir(exist_ok=True)
        results = defaultdict(list)
        per_user = defaultdict(list)
        inference_cost = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0,
                                              "completion_tokens": 0, "total_tokens": 0})
        for index, (user, permutation, mode) in enumerate(jobs):
            path = records_dir / f"{index:08d}.json"
            order = fixed_orders[user, permutation]
            context = context_for(mode, user, histories, metadata, memories, donors)
            if path.exists():
                envelope = json.loads(path.read_text(encoding="utf-8"))
                row = envelope["result"]
                if (envelope.get("fingerprint") != manifest["fingerprint"]
                        or envelope.get("index") != index
                        or envelope.get("result_sha256") != hashlib.sha256(canonical_json(row)).hexdigest()
                        or row.get("user_id") != user or row.get("mode") != mode
                        or row.get("permutation") != permutation
                        or row.get("target_item_id") != histories[user][-1]
                        or set(row.get("ranking", [])) != set(order)
                        or len(row.get("ranking", [])) != len(order)):
                    raise ValueError(f"Invalid saved result at {path}")
                if hasattr(llm, "set_token_stats"):
                    llm.set_token_stats(row["token_stats_after"])
            else:
                if args.resume and any(p.name > path.name for p in records_dir.glob("*.json")):
                    raise ValueError("Noncontiguous evaluation journal")
                row = retry_transaction(
                    lambda: model.rank_candidates(
                        user, order, histories[user][-1], user_context=context,
                        strict_ranking=not args.allow_repair,
                        include_candidate_description=args.include_candidate_description
                    ), llm
                )
                row.update({
                    "mode": mode, "permutation": permutation,
                    "candidate_order": order, "context_utf8_bytes": len(context.encode("utf-8")),
                    "donor_user_id": donors.get(user) if mode == "shuffled_memory" else None,
                    "token_stats_after": llm.get_token_stats(),
                })
                atomic_json(path, {
                    "fingerprint": manifest["fingerprint"], "index": index, "result": row,
                    "result_sha256": hashlib.sha256(canonical_json(row)).hexdigest(),
                })
            results[mode].append(row)
            per_user[(mode, user)].append(row)
            costs = row.get("attempts_token_delta", row["token_delta"])
            for key in inference_cost[mode]:
                inference_cost[mode][key] += costs.get(key, 0)

        summary = {
            "fingerprint": manifest["fingerprint"],
            "users": len(user_ids), "permutations": args.permutations,
            "candidate_description_enabled": args.include_candidate_description,
            "strict_ranking": not args.allow_repair,
            "memory_generation_cost_included": False,
            "snapshot_provenance_verified": snapshot_verified,
            "input_audit": input_audit,
            "metrics": {
                mode: {
                    "overall": evaluation_metrics(rows),
                    "by_user": {
                        str(user): evaluation_metrics(per_user[(mode, user)])
                        for user in user_ids
                    },
                    "inference_usage": dict(inference_cost[mode]),
                    "mean_user_context_utf8_bytes": sum(r["context_utf8_bytes"] for r in rows) / len(rows),
                    "ranking_repair_count": sum(bool(r["ranking_repaired"]) for r in rows),
                }
                for mode, rows in results.items()
            },
        }
        atomic_json(output / "summary.json", summary)
        return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path,
                        help="Same dataset JSON used to generate final_memories.json")
    result.add_argument("--memories", required=True, type=Path,
                        help="Frozen BaseRec final_memories.json")
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--endpoint", default="http://localhost:8000/v1")
    result.add_argument("--model", required=True)
    result.add_argument("--api-key-env", default="OPENAI_API_KEY")
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--permutations", type=int, default=1)
    result.add_argument("--modes", default=",".join(DEFAULT_MODES),
                        help=f"Comma-separated choices: {','.join(MODES)}")
    result.add_argument("--include-candidate-description", action="store_true")
    result.add_argument("--allow-repair", action="store_true",
                        help="Legacy mode: append missing candidate ids in input order")
    result.add_argument("--allow-unverified-memories", action="store_true",
                        help="Exploratory only: accept memory snapshots lacking a training manifest")
    result.add_argument("--max-generation-tokens", type=int, default=1024)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        print(json.dumps(execute(args), ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Memory value evaluation: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
