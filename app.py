import base64
import html

import streamlit as st
import streamlit.components.v1 as components

from graph.workflow import memory_qa, summarizer

MEMORY_CATEGORIES = {
    "people": ("👤", "People"),
    "events": ("📅", "Events"),
    "preferences": ("💡", "Preferences"),
    "important_facts": ("📌", "Important Facts"),
}


def group_memories(memories: list[dict[str, str]]) -> dict[str, list[str]]:
    """Group workflow memories by their recognized category for display."""
    grouped = {category: [] for category in MEMORY_CATEGORIES}
    for memory in memories:
        category = memory.get("category")
        content = memory.get("content")
        if category in grouped and isinstance(content, str) and content.strip():
            grouped[category].append(content.strip())
    return grouped


st.set_page_config(
    page_title="AI Paragraph Summarizer",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#172554; --blue:#244ED1; --soft:#84A5F2; --orange:#FF6600; --bg:#F3F6FB; }
      .stApp { background:var(--bg); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(243,246,251,.9); }
      [data-testid="stMainMenu"] { display:none !important; }
      [data-testid="stMainBlockContainer"] { max-width:900px; padding-top:2.2rem; padding-bottom:3rem; }
      html, body, [class*="st-"] { font-family:'DM Sans',sans-serif; }
      h1, h2, h3 { font-family:'Manrope','DM Sans',sans-serif !important; color:var(--ink); }
      .hero { text-align:center; padding:1rem .4rem 1.6rem; }
      .logo { display:inline-flex; width:52px; height:52px; align-items:center; justify-content:center; border-radius:17px; background:#e8efff; color:var(--blue); font-size:26px; margin-bottom:.8rem; }
      .hero h1 { font-size:clamp(1.8rem,4vw,2.55rem); font-weight:800; letter-spacing:-.045em; margin:0; }
      .hero p { color:#60708f; font-size:1rem; margin:.55rem 0 0; }
      .section-label { color:#35456c; font-size:.91rem; font-weight:700; margin:0 0 .45rem; }
      .stTextArea textarea { border:1px solid #dce4f2; border-radius:14px; background:#fff; color:var(--ink); line-height:1.65; padding:1rem; }
      .stTextArea textarea:focus { border-color:var(--soft); box-shadow:0 0 0 2px rgba(132,165,242,.18); }
      .count { color:#71809b; font-size:.82rem; margin-top:-.4rem; }
      div[data-testid="stButton"] > button { border-radius:11px; min-height:2.8rem; font-weight:700; transition:all .15s ease; }
      div[data-testid="stButton"] > button[kind="primary"] { background:var(--blue); border-color:var(--blue); color:white; }
      div[data-testid="stButton"] > button[kind="primary"]:hover { background:#193eaf; border-color:#193eaf; }
      div[data-testid="stButton"] > button:not([kind="primary"]) { border-color:#dce4f2; color:#344365; background:white; }
      div[data-testid="stRadio"] label { color:#344365; }
      .result-title { font-family:'Manrope',sans-serif; color:var(--ink); font-weight:800; font-size:1.15rem; margin:.3rem 0 .7rem; }
      .result-copy { background:white; border:1px solid #e1e8f4; border-radius:16px; padding:1.2rem 1.35rem; color:#263653; line-height:1.75; box-shadow:0 8px 24px rgba(28,52,100,.05); white-space:pre-wrap; }
      .stAlert { border-radius:12px; }
      @media (max-width:640px) {
        [data-testid="stMainBlockContainer"] { padding:1.1rem 1rem 2rem; }
        .hero { padding:.5rem 0 1.2rem; }
      }
    </style>
    <div class="hero">
      <div class="logo">✦</div>
      <h1>AI Paragraph Summarizer</h1>
      <p>Transform long paragraphs into clear and meaningful summaries.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def clear_input() -> None:
    st.session_state["paragraph"] = ""


def clear_all() -> None:
    st.session_state["paragraph"] = ""
    st.session_state["summary_text"] = ""
    st.session_state["summary_error"] = ""
    st.session_state["memories"] = []
    st.session_state["memory_storage_warning"] = ""
    st.session_state["memory_storage_count"] = 0
    st.session_state["summary_length"] = "Medium"


st.session_state.setdefault("paragraph", "")
st.session_state.setdefault("summary_text", "")
st.session_state.setdefault("summary_error", "")
st.session_state.setdefault("memories", [])
st.session_state.setdefault("memory_storage_warning", "")
st.session_state.setdefault("memory_storage_count", 0)
st.session_state.setdefault("summary_length", "Medium")
st.session_state.setdefault("app_mode", "Summarize a paragraph")
st.session_state.setdefault("memory_question", "")
st.session_state.setdefault("memory_answer", "")
st.session_state.setdefault("memory_answer_error", "")
st.session_state.setdefault("memory_retrieval_error", "")
st.session_state.setdefault("memory_retrieval_status", "")
st.session_state.setdefault("memory_context", [])

def render_summary_mode() -> None:
    st.markdown('<div class="section-label">Your text</div>', unsafe_allow_html=True)
    paragraph = st.text_area(
        "Paragraph to summarize",
        key="paragraph",
        height=230,
        placeholder="Paste or type one or more paragraphs here…",
        label_visibility="collapsed",
    )
    words = len(paragraph.split())
    st.markdown(
        f'<div class="count">{words:,} words <span style="padding:0 .35rem">·</span> {len(paragraph):,} characters</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label" style="margin-top:1.1rem">Summary length</div>', unsafe_allow_html=True)
    st.radio(
        "Choose summary length",
        ["Short", "Medium", "Detailed"],
        horizontal=True,
        key="summary_length",
        label_visibility="collapsed",
    )

    summarize_col, clear_input_col, _ = st.columns([1.4, 1, 3.2])
    with summarize_col:
        summarize_clicked = st.button("✦  Summarize", type="primary", use_container_width=True)
    with clear_input_col:
        st.button("Clear input", on_click=clear_input, use_container_width=True)

    if summarize_clicked:
        if not paragraph.strip():
            st.session_state["summary_error"] = "Enter some text before summarizing."
            st.session_state["summary_text"] = ""
        else:
            st.session_state["summary_error"] = ""
            st.session_state["summary_text"] = ""
            st.session_state["memories"] = []
            st.session_state["memory_storage_warning"] = ""
            st.session_state["memory_storage_count"] = 0
            try:
                with st.spinner("Creating your summary…"):
                    result = summarizer.invoke(
                        {
                            "paragraph": paragraph.strip(),
                            "summary": "",
                            "summary_length": st.session_state["summary_length"],
                        }
                    )
                st.session_state["summary_text"] = result["summary"]
                st.session_state["memories"] = result.get("memories", [])
                st.session_state["memory_storage_warning"] = result.get(
                    "memory_storage_warning", ""
                )
                st.session_state["memory_storage_count"] = result.get(
                    "memory_storage_count", 0
                )
            except Exception as exc:
                st.session_state["summary_error"] = str(exc)

    if st.session_state["summary_error"]:
        st.error(f"Could not create the summary. {st.session_state['summary_error']}")

    if st.session_state["summary_text"]:
        st.divider()
        st.markdown('<div class="result-title">📝 Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-copy">' + html.escape(st.session_state["summary_text"]) + "</div>",
            unsafe_allow_html=True,
        )
        summary_b64 = base64.b64encode(st.session_state["summary_text"].encode("utf-8")).decode("ascii")
        components.html(
            f"""
            <button id="copy-summary" style="margin-top:12px;padding:9px 15px;border:1px solid #dce4f2;border-radius:10px;background:#fff;color:#344365;font:600 14px 'DM Sans',sans-serif;cursor:pointer;">
              Copy summary
            </button>
            <span id="copy-status" style="margin-left:10px;color:#526482;font:13px 'DM Sans',sans-serif;"></span>
            <script>
              const summaryText = new TextDecoder().decode(Uint8Array.from(atob('{summary_b64}'), c => c.charCodeAt(0)));
              document.getElementById('copy-summary').addEventListener('click', async () => {{
                const status = document.getElementById('copy-status');
                try {{
                  await navigator.clipboard.writeText(summaryText);
                  status.textContent = 'Copied';
                }} catch (error) {{
                  status.textContent = 'Copy unavailable in this browser';
                }}
              }});
            </script>
            """,
            height=52,
        )

        st.markdown('<div class="result-title" style="margin-top:1.4rem">🧠 Extracted Memories</div>', unsafe_allow_html=True)
        grouped_memories = group_memories(st.session_state["memories"])
        if not any(grouped_memories.values()):
            st.info("No important memories were identified in this paragraph.")
        else:
            for category, (icon, title) in MEMORY_CATEGORIES.items():
                items = grouped_memories[category]
                if items:
                    st.markdown(f"### {icon} {title}")
                    for item in items:
                        st.markdown(
                            '<div class="result-copy" style="padding:.8rem 1rem;margin:.45rem 0">'
                            + html.escape(item)
                            + "</div>",
                            unsafe_allow_html=True,
                        )
        if st.session_state["memory_storage_warning"]:
            st.warning(st.session_state["memory_storage_warning"])
        elif st.session_state["memory_storage_count"]:
            st.caption("Extracted memories saved to your personal memory store.")
        st.button("Clear", on_click=clear_all, key="clear_all_button")



def clear_memory_qa() -> None:
    st.session_state["memory_question"] = ""
    st.session_state["memory_answer"] = ""
    st.session_state["memory_answer_error"] = ""
    st.session_state["memory_retrieval_error"] = ""
    st.session_state["memory_retrieval_status"] = ""
    st.session_state["memory_context"] = []


def render_memory_qa_mode() -> None:
    st.markdown('<div class="section-label">Ask about saved memories</div>', unsafe_allow_html=True)
    with st.form("memory_question_form"):
        question = st.text_input(
            "Your question",
            key="memory_question",
            placeholder="For example: Where do I work?",
        )
        ask_clicked = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.session_state["memory_answer_error"] = "Enter a question about your saved memories."
            st.session_state["memory_answer"] = ""
            st.session_state["memory_context"] = []
            st.session_state["memory_retrieval_error"] = ""
            st.session_state["memory_retrieval_status"] = ""
        else:
            st.session_state["memory_answer"] = ""
            st.session_state["memory_answer_error"] = ""
            st.session_state["memory_retrieval_error"] = ""
            st.session_state["memory_retrieval_status"] = ""
            st.session_state["memory_context"] = []
            try:
                with st.spinner("Searching your saved memories?"):
                    result = memory_qa.invoke({"question": question.strip()})
                st.session_state["memory_answer"] = result.get("answer", "")
                st.session_state["memory_answer_error"] = result.get("answer_error", "")
                st.session_state["memory_retrieval_error"] = result.get("retrieval_error", "")
                st.session_state["memory_retrieval_status"] = result.get("retrieval_status", "")
                st.session_state["memory_context"] = result.get("retrieved_memories", [])
            except Exception:
                st.session_state["memory_retrieval_error"] = (
                    "Saved memories could not be accessed. Check the PostgreSQL connection and try again."
                )
                st.session_state["memory_retrieval_status"] = "error"

    if st.session_state["memory_answer_error"]:
        st.error(st.session_state["memory_answer_error"])
    if st.session_state["memory_retrieval_error"]:
        st.error(st.session_state["memory_retrieval_error"])
    if st.session_state["memory_answer"]:
        st.divider()
        st.markdown('<div class="result-title">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-copy">' + html.escape(st.session_state["memory_answer"]) + '</div>',
            unsafe_allow_html=True,
        )
    if st.session_state["memory_context"]:
        with st.expander("Saved memory context used"):
            for memory in st.session_state["memory_context"]:
                category = html.escape(str(memory.get("category", "memory")).replace("_", " ").title())
                content = html.escape(str(memory.get("content", "")))
                st.markdown(
                    '<div class="result-copy" style="padding:.8rem 1rem;margin:.45rem 0">'
                    + f'<strong>{category}</strong><br>{content}</div>',
                    unsafe_allow_html=True,
                )
    elif st.session_state["memory_retrieval_status"] == "not_found":
        st.info("No relevant saved memories were found for this question.")

    st.button("Clear question and answer", on_click=clear_memory_qa)


mode = st.radio(
    "Choose a task",
    ["Summarize a paragraph", "Ask saved memories"],
    horizontal=True,
    key="app_mode",
    label_visibility="collapsed",
)
if mode == "Summarize a paragraph":
    render_summary_mode()
else:
    render_memory_qa_mode()
