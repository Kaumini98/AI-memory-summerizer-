from langgraph.graph import END, START, StateGraph

from graph.nodes import extract_memories, persist_memories, summarize_paragraph
from graph.retrieval import answer_question, retrieve_memories
from graph.state import MemoryQAState, SummaryState


graph_builder = StateGraph(SummaryState)
graph_builder.add_node("summarize", summarize_paragraph)
graph_builder.add_node("extract_memories", extract_memories)
graph_builder.add_node("persist_memories", persist_memories)
graph_builder.add_edge(START, "summarize")
graph_builder.add_edge(START, "extract_memories")
graph_builder.add_edge("summarize", END)
graph_builder.add_edge("extract_memories", "persist_memories")
graph_builder.add_edge("persist_memories", END)

summarizer = graph_builder.compile()


qa_builder = StateGraph(MemoryQAState)
qa_builder.add_node("retrieve_memories", retrieve_memories)
qa_builder.add_node("answer_question", answer_question)
qa_builder.add_edge(START, "retrieve_memories")
qa_builder.add_edge("retrieve_memories", "answer_question")
qa_builder.add_edge("answer_question", END)

memory_qa = qa_builder.compile()
