import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from graph.retrieval import answer_question, retrieve_memories
from graph.workflow import memory_qa


USER_NAMESPACE = ("users", "11111111-1111-4111-8111-111111111111", "memories")
OTHER_NAMESPACE = ("users", "22222222-2222-4222-8222-222222222222", "memories")


def memory_item(namespace: tuple[str, ...], content: str, category: str):
    return SimpleNamespace(
        namespace=namespace,
        value={
            "content": content,
            "category": category,
            "created_at": "2026-11-21T12:00:00+00:00",
        },
    )


class MemoryRetrievalTests(unittest.TestCase):
    def test_search_uses_only_the_configured_user_namespace(self) -> None:
        job = memory_item(
            USER_NAMESPACE,
            "Works as a software engineer at ABC Technologies.",
            "important_facts",
        )
        leaked_other_user_item = memory_item(
            OTHER_NAMESPACE, "Works at Other Company.", "important_facts"
        )
        store = Mock()
        # Include another namespace in the fake result to verify the node also
        # rejects any item whose full namespace is not the configured one.
        store.search.return_value = [leaked_other_user_item, job]
        with (
            patch("graph.retrieval.memories_namespace", return_value=USER_NAMESPACE),
            patch("graph.retrieval.get_memory_store", return_value=store),
        ):
            result = retrieve_memories({"question": "Where do I work?"})

        store.search.assert_called_once_with(USER_NAMESPACE, limit=100)
        self.assertEqual(result["retrieval_status"], "found")
        self.assertEqual(
            result["retrieved_memories"][0]["content"],
            "Works as a software engineer at ABC Technologies.",
        )
        self.assertNotIn(
            "Works at Other Company.",
            [item["content"] for item in result["retrieved_memories"]],
        )

    def test_keyword_matching_ranks_relevant_work_memory_first(self) -> None:
        store = Mock()
        store.search.return_value = [
            memory_item(USER_NAMESPACE, "Prefers Python for backend development.", "preferences"),
            memory_item(USER_NAMESPACE, "Attended a project management workshop.", "events"),
            memory_item(
                USER_NAMESPACE,
                "Works as a software engineer at ABC Technologies.",
                "important_facts",
            ),
        ]
        with (
            patch("graph.retrieval.memories_namespace", return_value=USER_NAMESPACE),
            patch("graph.retrieval.get_memory_store", return_value=store),
        ):
            result = retrieve_memories({"question": "Where do I work?"})

        self.assertEqual(
            result["retrieved_memories"][0]["content"],
            "Works as a software engineer at ABC Technologies.",
        )
        self.assertEqual(len(result["retrieved_memories"]), 1)

    def test_empty_results_return_not_found_without_calling_deepseek(self) -> None:
        store = Mock()
        store.search.return_value = []
        with (
            patch("graph.retrieval.memories_namespace", return_value=USER_NAMESPACE),
            patch("graph.retrieval.get_memory_store", return_value=store),
            patch("graph.retrieval.get_deepseek_client") as get_client,
        ):
            retrieved = retrieve_memories({"question": "Where do I work?"})
            answer = answer_question(
                {"question": "Where do I work?", **retrieved}
            )

        self.assertEqual(retrieved["retrieval_status"], "not_found")
        self.assertIn("couldn't find", answer["answer"])
        get_client.assert_not_called()

    def test_database_failure_is_reported_without_exposing_connection_details(self) -> None:
        with (
            patch("graph.retrieval.memories_namespace", return_value=USER_NAMESPACE),
            patch(
                "graph.retrieval.get_memory_store",
                side_effect=ConnectionError("postgresql://secret-password"),
            ),
        ):
            result = retrieve_memories({"question": "Where do I work?"})

        self.assertEqual(result["retrieval_status"], "error")
        self.assertIn("could not be accessed", result["retrieval_error"])
        self.assertNotIn("secret-password", result["retrieval_error"])
        with patch("graph.retrieval.get_deepseek_client") as get_client:
            answer = answer_question({"question": "Where do I work?", **result})
        self.assertIn("couldn't access", answer["answer"])
        get_client.assert_not_called()

    def test_answer_uses_only_the_retrieved_memories(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="You work at ABC Technologies."))]
        )
        retrieved = {
            "question": "Where do I work?",
            "retrieval_status": "found",
            "retrieved_memories": [
                {
                    "content": "Works as a software engineer at ABC Technologies.",
                    "category": "important_facts",
                }
            ],
        }
        with patch("graph.retrieval.get_deepseek_client", return_value=client):
            result = answer_question(retrieved)

        self.assertEqual(result["answer"], "You work at ABC Technologies.")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertIn("only the supplied saved memories", request["messages"][0]["content"])
        self.assertIn("ABC Technologies", request["messages"][1]["content"])

    def test_unsupported_detail_gets_not_available_answer(self) -> None:
        client = Mock()
        fallback = "I couldn't find that information in your saved memories."
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=fallback))]
        )
        state = {
            "question": "What is my dog's name?",
            "retrieval_status": "found",
            "retrieved_memories": [
                {"content": "The person's name is John.", "category": "people"}
            ],
        }
        with patch("graph.retrieval.get_deepseek_client", return_value=client):
            result = answer_question(state)

        self.assertEqual(result["answer"], fallback)
        self.assertIn("Do not use outside knowledge", client.chat.completions.create.call_args.kwargs["messages"][0]["content"])

    def test_separate_workflow_returns_answer_and_context(self) -> None:
        store = Mock()
        store.search.return_value = [
            memory_item(
                USER_NAMESPACE,
                "Works as a software engineer at ABC Technologies.",
                "important_facts",
            )
        ]
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="You work at ABC Technologies."))]
        )
        with (
            patch("graph.retrieval.memories_namespace", return_value=USER_NAMESPACE),
            patch("graph.retrieval.get_memory_store", return_value=store),
            patch("graph.retrieval.get_deepseek_client", return_value=client),
        ):
            result = memory_qa.invoke({"question": "Where do I work?"})

        self.assertEqual(result["answer"], "You work at ABC Technologies.")
        self.assertEqual(result["retrieval_status"], "found")
        self.assertEqual(len(result["retrieved_memories"]), 1)


if __name__ == "__main__":
    unittest.main()
