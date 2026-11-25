from typing import Literal, TypedDict


SummaryLength = Literal["Short", "Medium", "Detailed"]


class SummaryState(TypedDict):
    paragraph: str
    summary: str
    summary_length: SummaryLength
    memories: list[dict[str, str]]
    memory_storage_status: str
    memory_storage_warning: str
    memory_storage_count: int


RetrievalStatus = Literal["found", "not_found", "error"]


class MemoryQAState(TypedDict):
    question: str
    retrieved_memories: list[dict[str, str]]
    answer: str
    retrieval_status: RetrievalStatus
    retrieval_error: str
    answer_error: str
