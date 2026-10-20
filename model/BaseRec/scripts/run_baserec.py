#!/usr/bin/env python3
"""Run BaseRec using JSON input and an OpenAI-compatible chat endpoint."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.base_rec import BaseRec, build_update_schedule, token_delta

PROTOCOL_VERSION = "baserec-positive-v1"


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Durably replace one JSON file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Output directory is locked: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class UTF8TokenLimiter:
    """Conservative byte budget for byte-based tokenizers; not an exact tokenizer.

    A byte count upper-bounds token counts for ordinary byte-based BPE text.
    Supply a model tokenizer through --tokenizer-encoding for exact accounting.
    """
    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def truncate(self, text: str, limit: int) -> str:
        return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore").rstrip()


class TiktokenLimiter:
    def __init__(self, encoding: str):
        try:
            import tiktoken
        except ImportError as exc:
            raise ValueError("Install tiktoken to use --tokenizer-encoding") from exc
        self.encoding = tiktoken.get_encoding(encoding)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, limit: int) -> str:
        tokens = self.encoding.encode(text, disallowed_special=())
        # A token boundary may split a UTF-8 sequence. Decode only complete bytes.
        return self.encoding.decode_bytes(tokens[:limit]).decode("utf-8", errors="ignore").rstrip()


class JSONAwareLimiter:
    """Keep arrays valid when limiting JSON memory; never cut serialized JSON.

    Drop trailing points until the budget fits. The model then verifies its
    configured minimum point count. An impossible budget rejects the update.
    """
    def __init__(self, tokenizer, output_format: str):
        self.tokenizer, self.output_format = tokenizer, output_format

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def truncate(self, text: str, limit: int) -> str:
        if self.output_format != "json_points":
            return self.tokenizer.truncate(text, limit)
        points = json.loads(text)
        while points and self.count_tokens(json.dumps(points, ensure_ascii=False)) > limit:
            points.pop()
        return json.dumps(points, ensure_ascii=False)


class CompatibleLLM:
    """Small stdlib client. Retries belong to the interaction transaction."""
    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 120):
        self.url = endpoint.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_token_stats(self) -> dict:
        return dict(self.stats)

    def set_token_stats(self, stats: Mapping) -> None:
        if set(stats) != set(self.stats) or any(type(v) is not int or v < 0 for v in stats.values()):
            raise ValueError("Invalid checkpoint token counters")
        self.stats = dict(stats)

    def _request(self, messages, temperature, max_tokens, json_mode=False):
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=canonical_json(body), headers=headers, method="POST")
        self.stats["requests"] += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            # Do not print request headers or potentially sensitive response bodies.
            raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}") from exc
        usage = result.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.stats[key] += int(usage.get(key, 0))
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Endpoint did not return text content")
        return content

    def generate(self, messages, temperature, max_tokens) -> str:
        return self._request(messages, temperature, max_tokens)

    def generate_json(self, messages, properties, temperature, max_tokens) -> dict:
        schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        messages = copy.deepcopy(messages)
        messages[0]["content"] += " Return one JSON object matching this schema: " + json.dumps(schema)
        result = json.loads(self._request(messages, temperature, max_tokens, json_mode=True))
        if not isinstance(result, dict):
            raise ValueError("Endpoint returned a non-object JSON value")
        return result


def load_inputs(path: Path) -> tuple[dict, dict, dict]:
    """Input: {item_metadata: {id: facts}, histories: {user: [id]}, candidates: {user: [id]}}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    def integer_keys(raw, label):
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        result = {int(key): value for key, value in raw.items()}
        if len(result) != len(raw):
            raise ValueError(f"Duplicate normalized ids in {label}")
        return result
    metadata = integer_keys(data["item_metadata"], "item_metadata")
    histories = integer_keys(data["histories"], "histories")
    candidates = integer_keys(data["candidates"], "candidates")
    if not histories or set(histories) != set(candidates):
        raise ValueError("Every history needs a candidate list")
    if any(not isinstance(value, dict) for value in metadata.values()):
        raise ValueError("Item metadata must be objects")
    for user, items in histories.items():
        choices = candidates[user]
        for label, sequence in (("history", items), ("candidates", choices)):
            if not isinstance(sequence, list) or not sequence or any(type(x) is not int or x not in metadata for x in sequence):
                raise ValueError(f"Invalid {label} for user {user}")
        if len(choices) != len(set(choices)) or items[-1] not in choices:
            raise ValueError(f"Candidate list must be unique and contain held-out target for user {user}")
    return metadata, histories, candidates


def evaluation_metrics(results: list[dict]) -> dict:
    if not results:
        return {"MRR": 0.0, **{f"NDCG@{k}": 0.0 for k in (1, 3, 5, 10)}}
    return {"MRR": sum(1 / row["rank"] for row in results) / len(results), **{
        f"NDCG@{k}": sum(1 / math.log2(row["rank"] + 1) if row["rank"] <= k else 0 for row in results) / len(results)
        for k in (1, 3, 5, 10)}}


def retry_transaction(action, llm, attempts=3):
    start = copy.deepcopy(llm.get_token_stats())
    for attempt in range(1, attempts + 1):
        try:
            record = action()
            record["attempts"] = attempt
            record["attempts_token_delta"] = token_delta(start, llm.get_token_stats())
            return record
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"Transaction failed after {attempts} attempts: {exc}") from exc
    raise RuntimeError("Invalid retry count")


def protocol_manifest(args, model: BaseRec) -> dict:
    root = Path(__file__).resolve().parents[1]
    protocol = {"version": PROTOCOL_VERSION, "input_sha256": sha256_file(args.input),
                "model": args.model, "endpoint": args.endpoint.rstrip("/"),
                "memory_policy": model.get_memory_policy(), "ranking_temperature": 0,
                "tokenizer": args.tokenizer_encoding or "utf8_byte_upper_bound",
                "interaction_attempts": 3, "theme_generation_attempts": 3,
                "schedule": "position_round_robin_last_occurrence_held_out",
                "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in (
                    root / "src/models/base_rec.py", root / "scripts/run_baserec.py")}}
    return {"protocol": protocol, "fingerprint": hashlib.sha256(canonical_json(protocol)).hexdigest()}


def _read_record(path: Path, fingerprint: str, index: int) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("fingerprint") != fingerprint or envelope.get("index") != index:
        raise ValueError(f"Event protocol/index mismatch: {path.name}")
    record = envelope["record"]
    if envelope.get("record_sha256") != hashlib.sha256(canonical_json(record)).hexdigest():
        raise ValueError(f"Event integrity mismatch: {path.name}")
    return record


def _save_record(path: Path, fingerprint: str, index: int, record: dict) -> None:
    atomic_json(path, {"fingerprint": fingerprint, "index": index, "record": record,
                       "record_sha256": hashlib.sha256(canonical_json(record)).hexdigest()})


def execute(args, llm=None) -> dict:
    metadata, histories, candidates = load_inputs(args.input)
    llm = llm if llm is not None else CompatibleLLM(args.endpoint, args.model, os.environ.get(args.api_key_env, ""), args.timeout)
    tokenizer = TiktokenLimiter(args.tokenizer_encoding) if args.tokenizer_encoding else UTF8TokenLimiter()
    limiter = JSONAwareLimiter(tokenizer, args.memory_output_format)
    model = BaseRec(llm, metadata, list(histories), temperature=args.temperature,
                    max_memory_words=args.max_memory_words, memory_output_format=args.memory_output_format,
                    memory_token_limit_mode=args.memory_token_limit_mode, memory_token_limit=args.memory_token_limit,
                    token_limiter=limiter, json_point_format=args.json_point_format,
                    json_min_points=args.json_min_points, json_max_points=args.json_max_points,
                    max_memory_sentences=args.max_memory_sentences, max_generation_tokens=args.max_generation_tokens,
                    training_inference=not getattr(args, "skip_training_inference", False))
    manifest = protocol_manifest(args, model)
    fingerprint = manifest["fingerprint"]
    schedule = build_update_schedule(histories)
    output = args.output
    with output_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("Output already contains a run; use --resume or a new directory")
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Input or protocol fingerprint changed; refusing resume")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                raise ValueError("Output directory is not empty and has no manifest")
            atomic_json(manifest_path, manifest)
        events_dir, tests_dir = output / "events", output / "test_events"
        events_dir.mkdir(exist_ok=True)
        tests_dir.mkdir(exist_ok=True)
        checkpoint_path = output / "state.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        if checkpoint and checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        completed = checkpoint["completed_interactions"] if checkpoint else 0
        if type(completed) is not int or not 0 <= completed <= len(schedule):
            raise ValueError("Invalid checkpoint position")
        if checkpoint:
            model.set_state(checkpoint["model_state"])
            if model.frozen and completed != len(schedule):
                raise ValueError("Prematurely frozen checkpoint")
        # Rebuild an independent state from the authoritative event journal and
        # verify the checkpoint prefix before replaying any suffix.
        audit = BaseRec(llm, metadata, list(histories), token_limiter=limiter, **model.get_memory_policy())
        trace = []
        files = sorted(events_dir.glob("*.json"))
        if len(files) > len(schedule):
            raise ValueError("Too many events")
        for index, path in enumerate(files):
            if path.name != f"{index:08d}.json":
                raise ValueError("Event journal contains a gap or unexpected filename")
            record = _read_record(path, fingerprint, index)
            if record["event"] != schedule[index]:
                raise ValueError("Recorded event differs from update schedule")
            audit.apply_recorded_interaction(record)
            trace.append(record)
            if index + 1 == completed and audit.user_memories != model.user_memories:
                raise ValueError("Checkpoint memory differs from event journal")
            if index >= completed:
                model.apply_recorded_interaction(record)
        if len(trace) < completed:
            raise ValueError("Checkpoint is ahead of durable events")
        if completed == 0 and checkpoint and checkpoint["model_state"]["user_memories"] != {str(uid): "" for uid in histories}:
            raise ValueError("Nonempty initial checkpoint")
        # Restore cumulative successful + failed-attempt usage from durable records.
        durable_stats = trace[-1]["token_stats_after"] if trace else None
        def save_checkpoint():
            atomic_json(checkpoint_path, {"fingerprint": fingerprint, "completed_interactions": len(trace),
                                         "model_state": model.get_state(), "token_stats": llm.get_token_stats()})
        if durable_stats is not None and hasattr(llm, "set_token_stats"):
            llm.set_token_stats(durable_stats)
        for event in schedule[len(trace):]:
            record = retry_transaction(lambda: model.run_interaction(event), llm)
            record["token_stats_after"] = llm.get_token_stats()
            _save_record(events_dir / f"{event['event_id']:08d}.json", fingerprint, event["event_id"], record)
            trace.append(record)
            save_checkpoint()
        model.freeze()
        save_checkpoint()
        results = []
        test_files = sorted(tests_dir.glob("*.json"))
        if [p.name for p in test_files] != [f"{i:08d}.json" for i in range(len(test_files))] or len(test_files) > len(histories):
            raise ValueError("Invalid test journal")
        for index, (user, items) in enumerate(histories.items()):
            path = tests_dir / f"{index:08d}.json"
            if path.exists():
                result = _read_record(path, fingerprint, index)
                if (result["user_id"] != user or result["target_item_id"] != items[-1]
                        or set(result["ranking"]) != set(candidates[user])
                        or len(result["ranking"]) != len(candidates[user])
                        or result["rank"] != result["ranking"].index(items[-1]) + 1):
                    raise ValueError("Invalid saved test result")
                if hasattr(llm, "set_token_stats"):
                    llm.set_token_stats(result["token_stats_after"])
            else:
                result = retry_transaction(lambda: model.rank_candidates(user, candidates[user], items[-1]), llm)
                result["token_stats_after"] = llm.get_token_stats()
                _save_record(path, fingerprint, index, result)
            results.append(result)
        save_checkpoint()
        history = {str(uid): [{"item_position": None, "memory": ""}] for uid in histories}
        for record in trace:
            history[str(record["event"]["user_id"])].append({
                "item_position": record["event"]["item_position"], "memory": record["memory_after"],
                "stats": record["memory_stats"]["after"]})
        summary = {"fingerprint": fingerprint, "users": len(histories), "interactions": len(trace),
                   "test_users": len(results), "metrics": evaluation_metrics(results),
                   "token_stats": llm.get_token_stats(), "memory_policy": model.get_memory_policy(),
                   "memory_token_measure": args.tokenizer_encoding or "utf8_byte_upper_bound"}
        for name, data in (("interaction_trace", trace), ("memory_history", history),
                           ("final_memories", model.get_state()["user_memories"]),
                           ("test_results", results), ("summary", summary)):
            atomic_json(output / f"{name}.json", data)
        return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True, help="Dataset JSON")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--endpoint", default="http://localhost:8000/v1", help="API base URL or full chat/completions URL")
    result.add_argument("--model", required=True)
    result.add_argument("--api-key-env", default="OPENAI_API_KEY")
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--temperature", type=float, default=0)
    result.add_argument("--max-memory-words", type=int, default=180)
    result.add_argument("--max-memory-sentences", type=int, default=8)
    result.add_argument("--max-generation-tokens", type=int, default=1024)
    result.add_argument("--skip-training-inference", action="store_true",
                        help="Avoid diagnostic affinity calls that do not affect memory updates")
    result.add_argument("--memory-output-format", choices=["text", "json_points"], default="text")
    result.add_argument("--memory-token-limit-mode", choices=["none", "soft", "hard", "soft_hard"], default="none")
    result.add_argument("--memory-token-limit", type=int)
    result.add_argument("--json-point-format", choices=["sentence", "preference_theme"], default="sentence")
    result.add_argument("--json-min-points", type=int)
    result.add_argument("--json-max-points", type=int)
    result.add_argument("--tokenizer-encoding", help="Optional tiktoken encoding; otherwise conservative UTF-8 byte counts")
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        print(json.dumps(execute(args), ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"BaseRec: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
