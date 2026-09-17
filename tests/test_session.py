"""Tests for conversation state and persistence."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_cli.session import Session, SessionStore  # noqa: E402


class SessionTests(unittest.TestCase):
    def test_add_records_title_from_first_user_message(self) -> None:
        session = Session()
        session.add("user", "Explain recursion in one line please")
        self.assertEqual(session.title, "Explain recursion in one line please")

    def test_long_titles_are_truncated(self) -> None:
        session = Session()
        session.add("user", "x" * 200)
        self.assertLessEqual(len(session.title), 60)

    def test_newlines_are_flattened_in_titles(self) -> None:
        session = Session()
        session.add("user", "first line\nsecond line")
        self.assertNotIn("\n", session.title)

    def test_context_includes_system_prompt_first(self) -> None:
        session = Session(system_prompt="be terse")
        session.add("user", "hi")
        session.add("assistant", "hello")
        context = session.context()
        self.assertEqual(context[0], {"role": "system", "content": "be terse"})
        self.assertEqual([m["role"] for m in context[1:]], ["user", "assistant"])

    def test_context_omits_empty_system_prompt(self) -> None:
        session = Session()
        session.add("user", "hi")
        self.assertEqual(len(session.context()), 1)

    def test_context_limit_keeps_newest_messages(self) -> None:
        session = Session()
        for index in range(10):
            session.add("user", f"question {index}")
            session.add("assistant", f"answer {index}")
        context = session.context(limit=4)
        self.assertEqual(len(context), 4)
        self.assertEqual(context[-2]["content"], "question 9")
        self.assertEqual(context[0]["content"], "question 8")

    def test_usage_accumulates_across_turns(self) -> None:
        session = Session()
        session.record_usage({"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 2}, 0.001)
        session.record_usage({"prompt_tokens": 4, "completion_tokens": 1, "prompt_cache_hit_tokens": 1}, 0.002)
        self.assertEqual(session.usage["turns"], 2)
        self.assertEqual(session.usage["prompt_tokens"], 14)
        self.assertEqual(session.usage["completion_tokens"], 6)
        self.assertEqual(session.usage["prompt_cache_hit_tokens"], 3)
        self.assertAlmostEqual(session.usage["cost_usd"], 0.003, places=6)

    def test_record_usage_tolerates_missing_usage(self) -> None:
        session = Session()
        session.record_usage(None)
        self.assertEqual(session.usage["turns"], 1)
        self.assertEqual(session.usage["total_tokens"], 0)

    def test_drop_last_exchange_removes_pair(self) -> None:
        session = Session()
        session.add("user", "one")
        session.add("assistant", "1")
        session.add("user", "two")
        session.add("assistant", "2")
        self.assertTrue(session.drop_last_exchange())
        self.assertEqual([m["content"] for m in session.messages], ["one", "1"])

    def test_drop_last_exchange_on_empty_session(self) -> None:
        self.assertFalse(Session().drop_last_exchange())

    def test_roundtrip_through_dict(self) -> None:
        session = Session(model="deepseek-reasoner", system_prompt="sys")
        session.add("user", "hi")
        session.record_usage({"prompt_tokens": 3})
        restored = Session.from_dict(session.to_dict())
        self.assertEqual(restored.id, session.id)
        self.assertEqual(restored.model, "deepseek-reasoner")
        self.assertEqual(restored.system_prompt, "sys")
        self.assertEqual(restored.messages, session.messages)
        self.assertEqual(restored.usage["prompt_tokens"], 3)

    def test_from_dict_tolerates_missing_fields(self) -> None:
        restored = Session.from_dict({"id": "abc"})
        self.assertEqual(restored.id, "abc")
        self.assertEqual(restored.messages, [])
        self.assertEqual(restored.usage["turns"], 0)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_then_load(self) -> None:
        session = Session()
        session.add("user", "remember me")
        path = self.store.save(session)
        self.assertTrue(path.exists())
        loaded = self.store.load(session.id)
        self.assertEqual(loaded.messages[0]["content"], "remember me")

    def test_load_unknown_id_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.load("nope")

    def test_list_is_sorted_by_recency(self) -> None:
        first = Session()
        first.add("user", "older")
        self.store.save(first)
        second = Session()
        second.add("user", "newer")
        self.store.save(second)
        first.updated_at = 1.0
        self.store.save(first)
        ids = [s.id for s in self.store.list_sessions()]
        self.assertEqual(ids[0], second.id)

    def test_latest_returns_none_when_empty(self) -> None:
        self.assertIsNone(self.store.latest())

    def test_delete(self) -> None:
        session = Session()
        self.store.save(session)
        self.assertTrue(self.store.delete(session.id))
        self.assertFalse(self.store.delete(session.id))

    def test_corrupt_files_are_skipped(self) -> None:
        good = Session()
        self.store.save(good)
        (self.store.directory / "broken.json").write_text("{oops", encoding="utf-8")
        self.assertEqual([s.id for s in self.store.list_sessions()], [good.id])


if __name__ == "__main__":
    unittest.main()
