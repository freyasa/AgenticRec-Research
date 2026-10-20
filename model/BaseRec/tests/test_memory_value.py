"""Offline tests for controlled memory-value experiments."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from test_base_rec import FakeLLM, METADATA
from src.history_formatter import format_history
from src.models.base_rec import BaseRec, item_facts
from scripts import run_memory_value as value


class HistoryFormatTests(unittest.TestCase):
    def test_titles_and_full_have_different_visibility(self):
        titles = json.loads(format_history([1, 2], METADATA, "history_titles"))
        full = json.loads(format_history([1, 2], METADATA, "history_full"))
        structured = json.loads(format_history([1, 2], METADATA, "history_structured"))
        self.assertEqual(len(titles), 2)
        self.assertEqual(titles[0]["observed_positive_item"], {"title": "Merchant Secret"})
        self.assertEqual(full[0]["observed_positive_item"]["description"], "untrusted free text")
        self.assertNotIn("rating", structured[0]["observed_positive_item"])
        self.assertNotIn("title", structured[0]["observed_positive_item"])
        self.assertNotIn("description", structured[0]["observed_positive_item"])
        with self.assertRaises(ValueError):
            format_history([1], METADATA, "unknown")

    def test_description_is_opt_in_for_candidate_facts(self):
        self.assertNotIn("description", item_facts(METADATA[1]))
        self.assertEqual(item_facts(METADATA[1], include_description=True)["description"],
                         "untrusted free text")


class RankerTests(unittest.TestCase):
    def test_external_context_is_passed_unchanged_without_test_label(self):
        llm = FakeLLM(ranking=[3, 2, 1])
        model = BaseRec(llm, METADATA, [10])
        model.user_memories[10] = "This is the stored memory."
        model.freeze()
        row = model.rank_candidates(10, [1, 2, 3], 3, user_context="Title only.",
                                    strict_ranking=True, include_candidate_description=True)
        self.assertTrue(row["ranking_valid"])
        self.assertFalse(row["ranking_repaired"])
        payload = json.loads(llm.calls[-1]["messages"][1]["content"])
        self.assertEqual(payload["user_context"], "Title only.")
        self.assertNotIn("user_memory", payload)
        self.assertNotIn("target_item_id", payload)
        self.assertIn("description", payload["candidates"][0]["facts"])

    def test_strict_mode_rejects_missing_candidates(self):
        llm = FakeLLM(ranking=[3])
        model = BaseRec(llm, METADATA, [10])
        model.freeze()
        with self.assertRaisesRegex(ValueError, "complete, unique"):
            model.rank_candidates(10, [1, 2, 3], 3, user_context="",
                                  strict_ranking=True)
        repaired = model.rank_candidates(10, [1, 2, 3], 3, user_context="",
                                         strict_ranking=False)
        self.assertEqual(repaired["ranking"], [3, 1, 2])
        self.assertTrue(repaired["ranking_repaired"])
        self.assertEqual(repaired["missing_candidates"], [1, 2])

    def test_training_inference_can_be_skipped_and_truncation_audited(self):
        llm = FakeLLM(updates=["Cafe and desserts in large cities."])
        model = BaseRec(llm, METADATA, [10], training_inference=False,
                        max_memory_words=3)
        row = model.run_interaction({"user_id": 10, "item_id": 1, "item_position": 0})
        self.assertIsNone(row["inference"])
        self.assertEqual([c["kind"] for c in llm.calls], ["update"])
        self.assertEqual(row["memory_after"], "Cafe and desserts")
        audit = row["update_audit"]
        self.assertEqual(audit["raw_generated_memory"], "Cafe and desserts in large cities.")
        self.assertTrue(audit["truncation_applied"])
        self.assertGreater(audit["truncation_chars_removed"], 0)

    def test_donor_derangement(self):
        for n in range(2, 12):
            donors = value.donor_mapping(list(range(n)), 42)
            self.assertEqual(set(donors), set(range(n)))
            self.assertEqual(set(donors.values()), set(range(n)))
            self.assertTrue(all(k != v for k, v in donors.items()))


class ValueRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.input = root / "input.json"
        self.memories = root / "final_memories.json"
        self.output = root / "result"
        self.input.write_text(json.dumps({
            "item_metadata": METADATA,
            "histories": {"10": [1, 2, 3], "20": [2, 1]},
            "candidates": {"10": [1, 2, 3], "20": [3, 1, 2]},
        }), encoding="utf-8")
        self.memories.write_text(json.dumps({
            "10": "Prefers coffee.", "20": "Prefers tea."
        }), encoding="utf-8")
        self.args = SimpleNamespace(
            input=self.input, memories=self.memories, output=self.output,
            endpoint="http://localhost:8000/v1", model="offline-test",
            api_key_env="OPENAI_API_KEY", timeout=120.0, seed=42,
            permutations=2, modes="candidate_only,history_titles,real_memory,shuffled_memory",
            include_candidate_description=False, allow_repair=False,
            max_generation_tokens=1024, resume=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_context_comparisons_and_resume(self):
        llm = FakeLLM(ranking=[3, 2, 1])
        result = value.execute(self.args, llm)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["permutations"], 2)
        self.assertEqual(len(llm.calls), 16)
        self.assertEqual(result["metrics"]["history_titles"]["ranking_repair_count"], 0)
        rows = [json.loads(p.read_text())["result"]
                for p in sorted((self.output / "records").glob("*.json"))]
        by_job = {(r["user_id"], r["permutation"], r["mode"]): r for r in rows}
        for user in [10, 20]:
            for permutation in [0, 1]:
                orders = {tuple(by_job[user, permutation, mode]["candidate_order"])
                          for mode in ("candidate_only", "history_titles", "real_memory", "shuffled_memory")}
                self.assertEqual(len(orders), 1)
        for row in rows:
            if row["mode"] == "shuffled_memory":
                self.assertNotEqual(row["donor_user_id"], row["user_id"])
        context = value.context_for("history_titles", 10, {10: [1, 2, 3]},
                                    METADATA, {10: "hello"}, {})
        observed = json.loads(context)
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[1]["observed_positive_item"]["title"], "Second")
        self.assertFalse(any("Held Out Secret" in json.dumps(x) for x in observed))

        self.args.resume = True
        resumed_llm = FakeLLM(ranking=[3, 2, 1])
        self.assertEqual(result, value.execute(self.args, resumed_llm))
        self.assertEqual(resumed_llm.calls, [])

    def test_snapshot_manifest_mismatch_rejected(self):
        (self.memories.parent / "manifest.json").write_text(
            json.dumps({"protocol": {"input_sha256": "wrong"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different input"):
            value.execute(self.args, FakeLLM())

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            value.parse_modes("real_memory,real_memory")
        with self.assertRaises(ValueError):
            value.parse_modes("unknown")


if __name__ == "__main__":
    unittest.main()
