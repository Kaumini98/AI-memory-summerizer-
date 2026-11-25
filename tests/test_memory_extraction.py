import json
import os
import unittest
from unittest.mock import Mock, patch

from graph.nodes import extract_memories


class ExtractMemoriesTests(unittest.TestCase):
    def call_extractor(self, payload: dict) -> list[dict[str, str]]:
        response = Mock(
            choices=[Mock(message=Mock(content=json.dumps(payload)))],
        )
        client = Mock()
        client.chat.completions.create.return_value = response
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            patch("graph.nodes.get_deepseek_client", return_value=client),
        ):
            result = extract_memories({"paragraph": "My name is John."})
        system_prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("must be category people", system_prompt)
        return result["memories"]

    def test_personal_name_and_all_categories_are_classified(self) -> None:
        memories = self.call_extractor(
            {
                "memories": [
                    {"content": "The person's name is John.", "category": "people"},
                    {
                        "content": "Works as a software engineer at ABC Technologies.",
                        "category": "important_facts",
                    },
                    {
                        "content": "Prefers Python for backend development.",
                        "category": "preferences",
                    },
                    {
                        "content": "Attended a project management workshop last month.",
                        "category": "events",
                    },
                ]
            }
        )

        self.assertEqual(
            [memory["category"] for memory in memories],
            ["people", "important_facts", "preferences", "events"],
        )
        self.assertEqual(memories[0]["category"], "people")

    def test_unknown_categories_are_ignored(self) -> None:
        memories = self.call_extractor(
            {
                "memories": [
                    {"content": "Unknown classification.", "category": "other"},
                    {"content": "Non-string category.", "category": []},
                    {"content": "Valid name fact.", "category": "people"},
                ]
            }
        )

        self.assertEqual(
            memories,
            [{"content": "Valid name fact.", "category": "people"}],
        )


if __name__ == "__main__":
    unittest.main()
