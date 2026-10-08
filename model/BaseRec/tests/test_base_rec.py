"""Offline protocol, persistence, and compatible-endpoint tests."""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import BaseRec
from src.models.base_rec import (
    build_update_schedule, item_facts, memory_stats, positive_item_evidence,
    repair_ranking, sanitize_item_description, token_delta, transition_stats,
)
from scripts import run_baserec as runner


METADATA = {
    1: {"title": "Merchant Secret", "city": "Shanghai", "location": {"district": "Jing'an"},
        "latitude": 0, "categories": ["Cafe"], "price": "$$", "rating": 4.8,
        "hours": {"Monday": "09:00-18:00"}, "description": "untrusted free text",
        "attributes": {"WiFi": "free", "Outdoor": True, "Parking": False,
                       "Delivery": "none", "Alcohol": "no", "TV": "False", "Other": "null",
                       "Nested": {"yes": True, "no": False}}},
    2: {"title": "Second", "city": "Beijing", "categories": ["Tea"]},
    3: {"title": "Held Out Secret", "city": "Hangzhou", "categories": ["Noodles"]},
}
THEME = {"preference": "Enjoys cafes in Shanghai.", "evidence_count": 1,
         "confidence": "low", "contexts": ["Shanghai"]}


class FakeLLM:
    def __init__(self, updates=None, ranking=None):
        self.calls = []
        self.updates = list(updates or [])
        self.ranking = ranking if ranking is not None else [3, 2, 1]
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_token_stats(self):
        return dict(self.stats)

    def set_token_stats(self, stats):
        self.stats = dict(stats)

    def _call(self, kind, messages, temperature, properties=None):
        self.calls.append({"kind": kind, "messages": copy.deepcopy(messages),
                           "temperature": temperature, "properties": properties})
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += 7
        self.stats["completion_tokens"] += 3
        self.stats["total_tokens"] += 10

    def _update(self, default):
        value = self.updates.pop(0) if self.updates else default
        if isinstance(value, Exception):
            raise value
        return value

    def generate(self, messages, temperature, max_tokens):
        self._call("update", messages, temperature)
        return self._update("Enjoys cafes in Shanghai.")

    def generate_json(self, messages, properties, temperature, max_tokens):
        if "predicted_affinity" in properties:
            self._call("inference", messages, temperature, properties)
            return {"predicted_affinity": "uncertain", "rationale": "INFERENCE_SECRET_928"}
        if "ranking" in properties:
            self._call("ranking", messages, temperature, properties)
            return {"ranking": self.ranking}
        self._call("update", messages, temperature, properties)
        return self._update({"points": ["Enjoys cafes in Shanghai."]})


class WordLimiter:
    def count_tokens(self, value):
        return len(value.split())

    def truncate(self, value, limit):
        return " ".join(value.split()[:limit])


class BaseRecTests(unittest.TestCase):
    def model(self, llm=None, **kwargs):
        return BaseRec(llm or FakeLLM(), METADATA, [10, 20], **kwargs)

    def event(self, **kwargs):
        return {"user_id": 10, "item_id": 1, "item_position": 0, **kwargs}

    def test_inference_precedes_update_and_never_becomes_evidence(self):
        llm = FakeLLM()
        model = self.model(llm)
        self.assertEqual(model.user_memories, {10: "", 20: ""})
        record = model.run_interaction(self.event())
        self.assertEqual([call["kind"] for call in llm.calls], ["inference", "update"])
        inference_data = json.loads(llm.calls[0]["messages"][1]["content"])
        update_data = json.loads(llm.calls[1]["messages"][1]["content"])
        self.assertEqual(set(inference_data), {"old_memory", "item_facts"})
        self.assertEqual(set(update_data), {"old_memory", "positive_evidence"})
        self.assertEqual(update_data["old_memory"], "")
        self.assertNotIn("INFERENCE_SECRET_928", json.dumps(llm.calls[1]))
        self.assertNotIn("Merchant Secret", json.dumps(update_data))
        self.assertNotIn("rating", json.dumps(update_data))
        self.assertNotIn("hours", json.dumps(update_data))
        self.assertEqual(record["token_delta"]["total_tokens"], 20)
        self.assertEqual(record["inference_token_delta"]["total_tokens"], 10)
        self.assertFalse(hasattr(model, "item_memories"))

    def test_second_interaction_reads_old_memory(self):
        llm = FakeLLM(updates=["First memory.", "Second memory."])
        model = self.model(llm)
        model.run_interaction(self.event())
        model.run_interaction(self.event(item_id=2, item_position=1))
        for call in llm.calls[2:]:
            self.assertEqual(json.loads(call["messages"][1]["content"])["old_memory"], "First memory.")
        self.assertEqual(model.user_memories[10], "Second memory.")
        self.assertEqual(model.user_memories[20], "")

    def test_evidence_keeps_geography_and_affirmative_attributes(self):
        evidence = positive_item_evidence(METADATA[1])
        self.assertEqual(evidence["city"], "Shanghai")
        self.assertEqual(evidence["location"], {"district": "Jing'an"})
        self.assertEqual(evidence["latitude"], 0)
        self.assertEqual(evidence["affirmative_attributes"],
                         {"WiFi": "free", "Outdoor": True, "Nested": {"yes": True}})
        for key in ("title", "name", "rating", "hours", "description", "item_id"):
            self.assertNotIn(key, evidence)
        facts = item_facts(METADATA[1])
        for key in ("title", "rating", "hours", "city", "price", "categories", "attributes"):
            self.assertIn(key, facts)

    def test_sanitization(self):
        for value in (False, None, "False", " NONE ", "no", "null", "'none'", 0):
            self.assertIsNone(sanitize_item_description(value))
        self.assertEqual(sanitize_item_description([" cafe  ", None, "no"]), ["cafe"])

    def test_serialized_nested_attributes_and_forbidden_fields(self):
        metadata = {"attributes": {
            "Parking": "{'garage': False, 'street': True}",
            "Other": '{"rating": 5, "opening_hours": "all day", "WiFi": "free"}',
            "Merchant_Name": "Secret", "item_id": 123,
        }}
        self.assertEqual(positive_item_evidence(metadata), {"affirmative_attributes": {
            "Parking": {"street": True}, "Other": {"WiFi": "free"}}})

    def test_schedule_is_interleaved_and_holds_out_last_position(self):
        histories = {10: [1, 2, 3], 20: [2, 3], 30: [1], 40: []}
        events = build_update_schedule(histories)
        self.assertEqual([(e["user_id"], e["item_id"], e["item_position"]) for e in events],
                         [(10, 1, 0), (20, 2, 0), (10, 2, 1)])
        self.assertEqual(build_update_schedule({}), [])
        # A repeated id in earlier history remains a valid earlier interaction.
        self.assertEqual(len(build_update_schedule({1: [3, 3]})), 1)

    def test_text_word_and_sentence_limits(self):
        model = self.model(FakeLLM(updates=["One. Two. Three. Four."]), max_memory_sentences=2)
        model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "One. Two.")
        model = self.model(FakeLLM(updates=["one two three four"]), max_memory_words=2)
        model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "one two")
        model = self.model(FakeLLM(updates=["喜欢咖啡。偏好上海。喜欢茶。"]), max_memory_sentences=2)
        model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "喜欢咖啡。 偏好上海。")

    def test_json_sentences_are_deduplicated(self):
        model = self.model(FakeLLM(updates=[{"points": [" Cafes. ", "cafes.", "Tea."]}]),
                           memory_output_format="json_points")
        record = model.run_interaction(self.event())
        self.assertEqual(json.loads(record["memory_after"]), ["Cafes.", "Tea."])
        self.assertEqual(record["memory_stats"]["after"]["points"], 2)

    def test_preference_themes(self):
        model = self.model(FakeLLM(updates=[{"points": [THEME]}]), memory_output_format="json_points",
                           json_point_format="preference_theme", json_min_points=1, json_max_points=2)
        model.run_interaction(self.event())
        self.assertEqual(json.loads(model.user_memories[10]), [THEME])

    def test_theme_bounds_must_be_explicit(self):
        with self.assertRaises(ValueError):
            self.model(memory_output_format="json_points", json_point_format="preference_theme")

    def test_theme_structure_and_count_errors_retry_three_attempts(self):
        bad = dict(THEME, confidence="certain")
        llm = FakeLLM(updates=[{"points": [bad]}, {"points": []}, {"points": [THEME]}])
        model = self.model(llm, memory_output_format="json_points", json_point_format="preference_theme",
                           json_min_points=1, json_max_points=2)
        model.run_interaction(self.event())
        self.assertEqual([c["kind"] for c in llm.calls], ["inference", "update", "update", "update"])
        for call in llm.calls[1:]:
            self.assertNotIn("INFERENCE_SECRET_928", json.dumps(call))

    def test_invalid_theme_never_commits(self):
        llm = FakeLLM(updates=[{"points": []}] * 4)
        model = self.model(llm, memory_output_format="json_points", json_point_format="preference_theme",
                           json_min_points=1, json_max_points=2)
        with self.assertRaises(ValueError):
            model.run_interaction(self.event())
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(model.user_memories[10], "")

    def test_all_token_policies(self):
        for mode in ("none", "soft", "hard", "soft_hard"):
            with self.subTest(mode=mode):
                llm = FakeLLM(updates=["One. Two. Three. Four."])
                model = self.model(llm, memory_token_limit_mode=mode, memory_token_limit=3,
                                   token_limiter=WordLimiter(), max_memory_sentences=2)
                model.run_interaction(self.event())
                instruction = llm.calls[1]["messages"][0]["content"]
                self.assertEqual("at most 3 tokens" in instruction, mode in ("soft", "soft_hard"))
                self.assertEqual("short sentences" in instruction, mode != "soft_hard")
                self.assertEqual(model.user_memories[10], "One. Two. Three." if mode == "soft_hard" else "One. Two.")
        llm = FakeLLM(updates=["one two three four"])
        model = self.model(llm, memory_token_limit_mode="soft", memory_token_limit=1)
        model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "one two three four")

    def test_hard_limit_is_verified_not_just_delegated(self):
        class BrokenLimiter(WordLimiter):
            def truncate(self, text, limit):
                return text
        model = self.model(memory_token_limit_mode="hard", memory_token_limit=1, token_limiter=BrokenLimiter())
        with self.assertRaises(ValueError):
            model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "")

    def test_invalid_configuration(self):
        for kwargs in ({"memory_token_limit_mode": "hard", "memory_token_limit": 8},
                       {"memory_token_limit_mode": "soft"}, {"memory_output_format": "other"},
                       {"json_min_points": 3, "json_max_points": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)

    def test_failed_update_does_not_commit(self):
        llm = FakeLLM(updates=[RuntimeError("unavailable")])
        model = self.model(llm)
        with self.assertRaises(RuntimeError):
            model.run_interaction(self.event())
        self.assertEqual(model.user_memories[10], "")

    def test_bad_inference_does_not_call_update(self):
        llm = FakeLLM()
        llm.generate_json = lambda *args, **kwargs: {"predicted_affinity": "yes", "rationale": "bad"}
        model = self.model(llm)
        with self.assertRaises(ValueError):
            model.run_interaction(self.event())
        self.assertEqual(llm.calls, [])
        self.assertEqual(model.user_memories[10], "")

    def test_invalid_event_rejected_before_calling_llm(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.run_interaction(self.event(item_position=-1))
        self.assertEqual(model.llm.calls, [])

    def test_checkpoint_and_record_replay(self):
        model = self.model()
        record = model.run_interaction(self.event())
        restored = self.model()
        restored.set_state(json.loads(json.dumps(model.get_state())))
        self.assertEqual(restored.user_memories, model.user_memories)
        replayed = self.model()
        replayed.apply_recorded_interaction(record)
        self.assertEqual(replayed.user_memories, model.user_memories)
        self.assertEqual(replayed.llm.calls, [])
        with self.assertRaises(ValueError):
            replayed.apply_recorded_interaction(record)
        model.freeze()
        restored.set_state(model.get_state())
        with self.assertRaises(RuntimeError):
            restored.run_interaction(self.event())

    def test_checkpoint_validation_is_atomic(self):
        model = self.model()
        state = model.get_state()
        state["user_memories"]["20"] = 7
        with self.assertRaises(ValueError):
            model.set_state(state)
        self.assertEqual(model.user_memories, {10: "", 20: ""})
        state = model.get_state()
        state["memory_policy"]["temperature"] = 99
        with self.assertRaises(ValueError):
            model.set_state(state)

    def test_ranking_repair(self):
        self.assertEqual(repair_ranking([3, True, 3, 9, "2", None, {}, 1.0], [1, 2, 3]), [3, 1, 2])
        self.assertEqual(repair_ranking(None, [2, 1]), [2, 1])

    def test_ranking_is_frozen_deterministic_and_read_only(self):
        llm = FakeLLM(ranking=[2, 2, 99])
        model = self.model(llm, temperature=0.8)
        with self.assertRaises(RuntimeError):
            model.rank_candidates(10, [1, 2, 3], 3)
        model.run_interaction(self.event())
        model.freeze()
        before = copy.deepcopy(model.get_state())
        result = model.rank_candidates(10, [1, 2, 3], 3)
        self.assertEqual(result["ranking"], [2, 1, 3])
        self.assertEqual(result["rank"], 3)
        self.assertEqual(model.get_state(), before)
        self.assertEqual(llm.calls[-1]["temperature"], 0)
        payload = json.loads(llm.calls[-1]["messages"][1]["content"])
        self.assertNotIn("target_item_id", payload)
        self.assertIn("rating", payload["candidates"][0]["facts"])

    def test_statistics(self):
        self.assertEqual(memory_stats("a b", WordLimiter())["tokens"], 2)
        self.assertEqual(transition_stats("a", "a b")["delta"]["words"], 1)
        self.assertEqual(token_delta({"a": 2, "nested": {"b": 3}},
                                     {"a": 5, "nested": {"b": 4}, "model": "test"}), {"a": 3, "nested": {"b": 1}})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "input.json"
        self.data.write_text(json.dumps({"item_metadata": METADATA, "histories": {10: [1, 2, 3], 20: [2, 3]},
                                        "candidates": {10: [1, 2, 3], 20: [1, 2, 3]}}))
        self.args = runner.parser().parse_args(["--input", str(self.data), "--output", str(self.root / "out"),
                                               "--model", "offline-test"])

    def tearDown(self):
        self.temp.cleanup()

    def test_end_to_end_outputs_and_resume_without_llm_calls(self):
        llm = FakeLLM()
        summary = runner.execute(self.args, llm)
        self.assertEqual(summary["interactions"], 3)
        self.assertEqual(summary["metrics"], {"MRR": 1.0, "NDCG@1": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "NDCG@10": 1.0})
        for name in ("manifest", "state", "interaction_trace", "memory_history", "final_memories", "test_results", "summary"):
            self.assertTrue((self.args.output / f"{name}.json").exists())
        for call in llm.calls:
            if call["kind"] != "ranking":
                self.assertNotIn("Held Out Secret", json.dumps(call))
                self.assertNotIn("Hangzhou", json.dumps(call))
        self.args.resume = True
        resumed_llm = FakeLLM()
        resumed = runner.execute(self.args, resumed_llm)
        self.assertEqual(resumed_llm.calls, [])
        self.assertEqual(resumed, summary)

    def test_event_saved_before_checkpoint_failure_can_be_replayed(self):
        atomic = runner.atomic_json
        def fail_state(path, value):
            if path.name == "state.json":
                raise OSError("simulated crash after durable event")
            return atomic(path, value)
        with patch.object(runner, "atomic_json", side_effect=fail_state), self.assertRaises(OSError):
            runner.execute(self.args, FakeLLM())
        self.assertTrue((self.args.output / "events/00000000.json").exists())
        self.assertFalse((self.args.output / "state.json").exists())
        self.args.resume = True
        llm = FakeLLM()
        summary = runner.execute(self.args, llm)
        self.assertEqual(summary["interactions"], 3)
        self.assertEqual([c["kind"] for c in llm.calls].count("inference"), 2)
        self.assertEqual(summary["token_stats"]["requests"], 8)

    def test_stale_checkpoint_suffix_replay(self):
        atomic = runner.atomic_json
        saves = 0
        def fail_second_state(path, value):
            nonlocal saves
            if path.name == "state.json":
                saves += 1
                if saves == 2:
                    raise OSError("simulated crash")
            return atomic(path, value)
        with patch.object(runner, "atomic_json", side_effect=fail_second_state), self.assertRaises(OSError):
            runner.execute(self.args, FakeLLM())
        self.args.resume = True
        llm = FakeLLM()
        runner.execute(self.args, llm)
        self.assertEqual([c["kind"] for c in llm.calls].count("inference"), 1)

    def test_input_hash_and_protocol_protect_resume(self):
        runner.execute(self.args, FakeLLM())
        self.args.resume = True
        self.args.temperature = 0.1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.execute(self.args, FakeLLM())
        self.args.temperature = 0
        self.data.write_text(self.data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            runner.execute(self.args, FakeLLM())

    def test_checkpoint_tampering_rejected(self):
        runner.execute(self.args, FakeLLM())
        path = self.args.output / "state.json"
        state = json.loads(path.read_text())
        state["model_state"]["user_memories"]["10"] = "Tampered memory."
        runner.atomic_json(path, state)
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "Checkpoint memory"):
            runner.execute(self.args, FakeLLM())

    def test_event_tampering_rejected(self):
        runner.execute(self.args, FakeLLM())
        path = self.args.output / "events/00000000.json"
        envelope = json.loads(path.read_text())
        envelope["record"]["memory_after"] = "tampered"
        runner.atomic_json(path, envelope)
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "integrity"):
            runner.execute(self.args, FakeLLM())

    def test_missing_event_rejected(self):
        runner.execute(self.args, FakeLLM())
        (self.args.output / "events/00000001.json").unlink()
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "gap"):
            runner.execute(self.args, FakeLLM())

    def test_failure_retries_three_times_without_committing(self):
        llm = FakeLLM(updates=[RuntimeError("offline")] * 5)
        with self.assertRaisesRegex(RuntimeError, "3 attempts"):
            runner.execute(self.args, llm)
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(list((self.args.output / "events").glob("*.json")), [])

    def test_retry_usage_is_counted(self):
        llm = FakeLLM(updates=[RuntimeError("temporary"), "Likes cafes."])
        runner.execute(self.args, llm)
        record = json.loads((self.args.output / "events/00000000.json").read_text())["record"]
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(record["attempts_token_delta"]["requests"], 4)

    def test_output_lock_blocks_second_process(self):
        script = ("from pathlib import Path; from scripts.run_baserec import output_lock; "
                  f"lock = output_lock(Path({str(self.args.output)!r})); lock.__enter__()")
        with runner.output_lock(self.args.output):
            child = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn("is locked", child.stderr)

    def test_atomic_write_preserves_previous_file_on_replace_failure(self):
        path = self.root / "atomic.json"
        runner.atomic_json(path, {"old": True})
        with patch.object(runner.os, "replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
            runner.atomic_json(path, {"new": True})
        self.assertEqual(json.loads(path.read_text()), {"old": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_compatible_http_payload_and_token_stats(self):
        requests = []
        def respond(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"ranking": [2, 1]}'}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}}).encode())
        client = runner.CompatibleLLM("http://localhost:8000/v1", "test-model", "test-key")
        with patch.object(runner, "urlopen", side_effect=respond):
            result = client.generate_json([{"role": "system", "content": "Rank."}], {"ranking": {"type": "array"}}, 0, 100)
        self.assertEqual(result, {"ranking": [2, 1]})
        self.assertTrue(requests[0].full_url.endswith("/v1/chat/completions"))
        body = json.loads(requests[0].data)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(client.get_token_stats()["total_tokens"], 16)

    def test_json_hard_limiter_preserves_structure(self):
        limiter = runner.JSONAwareLimiter(runner.UTF8TokenLimiter(), "json_points")
        value = json.dumps(["Cafe", "Tea"])
        self.assertEqual(json.loads(limiter.truncate(value, len('["Cafe"]'))), ["Cafe"])
        model = BaseRec(FakeLLM(updates=[{"points": ["Cafe", "Tea"]}]), METADATA, [10],
                        memory_output_format="json_points", memory_token_limit_mode="hard",
                        memory_token_limit=8, token_limiter=limiter)
        model.run_interaction({"user_id": 10, "item_id": 1, "item_position": 0})
        self.assertEqual(json.loads(model.user_memories[10]), ["Cafe"])

    def test_json_impossible_hard_budget_rejects_update(self):
        limiter = runner.JSONAwareLimiter(runner.UTF8TokenLimiter(), "json_points")
        model = BaseRec(FakeLLM(updates=[{"points": ["Cafe"]}]), METADATA, [10],
                        memory_output_format="json_points", memory_token_limit_mode="hard",
                        memory_token_limit=2, token_limiter=limiter)
        with self.assertRaises(ValueError):
            model.run_interaction({"user_id": 10, "item_id": 1, "item_position": 0})
        self.assertEqual(model.user_memories[10], "")

    def test_singleton_histories_only_rank(self):
        self.data.write_text(json.dumps({"item_metadata": METADATA, "histories": {10: [3]},
                                        "candidates": {10: [1, 2, 3]}}))
        llm = FakeLLM()
        summary = runner.execute(self.args, llm)
        self.assertEqual(summary["interactions"], 0)
        self.assertEqual([call["kind"] for call in llm.calls], ["ranking"])
        self.assertEqual(json.loads((self.args.output / "final_memories.json").read_text()), {"10": ""})
        self.args.resume = True
        resumed = FakeLLM()
        self.assertEqual(runner.execute(self.args, resumed), summary)
        self.assertEqual(resumed.calls, [])

    def test_resume_partial_evaluation(self):
        atomic = runner.atomic_json
        def fail_second_test(path, value):
            if path.parent.name == "test_events" and path.name == "00000001.json":
                raise OSError("simulated evaluation crash")
            return atomic(path, value)
        with patch.object(runner, "atomic_json", side_effect=fail_second_test), self.assertRaises(OSError):
            runner.execute(self.args, FakeLLM())
        self.args.resume = True
        llm = FakeLLM()
        runner.execute(self.args, llm)
        self.assertEqual([call["kind"] for call in llm.calls], ["ranking"])

    def test_input_rejects_unknown_candidates_and_missing_target(self):
        data = json.loads(self.data.read_text())
        for values in ([1, 999], [1, 2], [3, 3], [True, 3]):
            with self.subTest(values=values):
                data["candidates"]["10"] = values
                self.data.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    runner.load_inputs(self.data)

    def test_metrics(self):
        metrics = runner.evaluation_metrics([{"rank": 1}, {"rank": 4}])
        self.assertEqual(metrics["MRR"], 0.625)
        self.assertEqual(metrics["NDCG@3"], 0.5)
        self.assertAlmostEqual(metrics["NDCG@5"], (1 + 1 / math.log2(5)) / 2)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_baserec.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--resume", result.stdout)


if __name__ == "__main__":
    unittest.main()
