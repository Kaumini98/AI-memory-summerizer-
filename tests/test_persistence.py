import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from graph.persistence import memory_key, memories_namespace, save_memories
from graph.workflow import summarizer


USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"
SAMPLE_MEMORY = {
    "content": "Prefers Python for backend development.",
    "category": "preferences",
}


class FakeStore:
    def __init__(self) -> None:
        self.items: dict[tuple[tuple[str, ...], str], dict[str, str]] = {}

    def get(self, namespace: tuple[str, ...], key: str):
        value = self.items.get((namespace, key))
        return SimpleNamespace(value=value) if value is not None else None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, str]) -> None:
        self.items[(namespace, key)] = value


class PersistenceTests(unittest.TestCase):
    def test_user_namespaces_are_isolated(self) -> None:
        self.assertNotEqual(memories_namespace(USER_A), memories_namespace(USER_B))

    def test_repeated_memory_uses_same_key_and_preserves_created_at(self) -> None:
        store = FakeStore()
        with patch("graph.persistence.get_memory_store", return_value=store):
            first_count = save_memories([SAMPLE_MEMORY], USER_A)
            namespace = memories_namespace(USER_A)
            key = memory_key(SAMPLE_MEMORY["content"], SAMPLE_MEMORY["category"])
            original_created_at = store.get(namespace, key).value["created_at"]
            second_count = save_memories([SAMPLE_MEMORY], USER_A)

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 1)
        self.assertEqual(len(store.items), 1)
        self.assertEqual(
            store.get(namespace, key).value["created_at"], original_created_at
        )

    def test_same_memory_is_stored_in_separate_user_namespaces(self) -> None:
        store = FakeStore()
        with patch("graph.persistence.get_memory_store", return_value=store):
            save_memories([SAMPLE_MEMORY], USER_A)
            save_memories([SAMPLE_MEMORY], USER_B)

        self.assertEqual(len(store.items), 2)
        self.assertIn((memories_namespace(USER_A), memory_key(**{
            "content": SAMPLE_MEMORY["content"],
            "category": SAMPLE_MEMORY["category"],
        })), store.items)
        self.assertIn((memories_namespace(USER_B), memory_key(
            SAMPLE_MEMORY["content"], SAMPLE_MEMORY["category"]
        )), store.items)

    def test_extraction_and_summary_survive_storage_failure(self) -> None:
        client = Mock()

        def create(**kwargs):
            system_message = kwargs["messages"][0]["content"]
            content = (
                json.dumps({"memories": [SAMPLE_MEMORY]})
                if "Extract only useful information" in system_message
                else "A successful summary despite the storage outage."
            )
            return Mock(choices=[Mock(message=Mock(content=content))])

        client.chat.completions.create.side_effect = create
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            patch("graph.nodes.get_deepseek_client", return_value=client),
            patch("graph.nodes.save_memories", side_effect=ConnectionError("hidden URL")),
        ):
            result = summarizer.invoke(
                {
                    "paragraph": "I prefer Python for backend development.",
                    "summary": "",
                    "summary_length": "Short",
                }
            )

        self.assertEqual(
            result["summary"], "A successful summary despite the storage outage."
        )
        self.assertEqual(result["memories"], [SAMPLE_MEMORY])
        self.assertEqual(result["memory_storage_status"], "failed")
        self.assertIn("PostgreSQL", result["memory_storage_warning"])
        self.assertNotIn("hidden URL", result["memory_storage_warning"])


if __name__ == "__main__":
    unittest.main()
