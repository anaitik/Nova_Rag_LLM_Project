# pip install streamlit pymupdf langchain langchain-groq langchain-text-splitters python-dotenv
#
# Setup:
# 1. Create a folder named "documents" next to this file.
# 2. Place these PDFs inside ./documents/:
#    - 01_AI_Usage_Policy.pdf
#    - 02_Data_Privacy_Guideline.pdf
#    - 03_Customer_Data_Handling_Policy.pdf
#    - 04_Compliance_Approval_Process.pdf
#    - 05_EU_AI_Act_Internal_Briefing.pdf
# 3. Set your Groq API key:
#    PowerShell: $env:GROQ_API_KEY="your-api-key"
#    macOS/Linux: export GROQ_API_KEY="your-api-key"
# 4. Run: streamlit run app.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import fitz
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Nova Insurance Group - Compliance & Policy Assistant"
DOCUMENTS_DIR = Path(__file__).parent / "documents"
EXPECTED_PDFS = [
    "01_AI_Usage_Policy.pdf",
    "02_Data_Privacy_Guideline.pdf",
    "03_Customer_Data_Handling_Policy.pdf",
    "04_Compliance_Approval_Process.pdf",
    "05_EU_AI_Act_Internal_Briefing.pdf",
]

CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = 3
MAX_CONTEXT_CHARS = 4500
MAX_CHARS_PER_CHUNK = 1400
MAX_HISTORY_TURNS = 2
MAX_OUTPUT_TOKENS = 500
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "may",
    "must",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}

SYSTEM_PROMPT = """You are a helpful compliance assistant for Nova Insurance Group.
Answer the question using ONLY the provided policy excerpts.
If the answer cannot be found in the provided excerpts, say so clearly.
Cite the source documents by name and page number.
Do not use outside knowledge, assumptions, or unstated policy details."""


def get_api_key() -> str | None:
    """Load the Groq API key from the environment or a local .env file."""
    load_dotenv()
    return os.getenv("GROQ_API_KEY")


def validate_document_folder(documents_dir: Path) -> list[Path]:
    """Ensure the required PDF folder and files exist before building the index."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Missing documents folder: {documents_dir}. Create it and add the required PDFs."
        )

    missing = [
        filename
        for filename in EXPECTED_PDFS
        if not (documents_dir / filename).is_file()
    ]
    if missing:
        missing_list = "\n".join(f"- {filename}" for filename in missing)
        raise FileNotFoundError(f"Missing required PDF files:\n{missing_list}")

    return [documents_dir / filename for filename in EXPECTED_PDFS]


def load_pdf_pages(pdf_paths: Iterable[Path]) -> list[Document]:
    """Load PDFs page by page so page numbers can be cited in answers."""
    documents: list[Document] = []

    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_index,
                                "path": str(pdf_path),
                            },
                        )
                    )
        except Exception as exc:
            raise RuntimeError(f"Could not read {pdf_path.name}: {exc}") from exc

    if not documents:
        raise ValueError("No readable text was found in the required PDFs.")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split pages into retrieval-sized chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def document_signature(documents_dir: Path) -> tuple[tuple[str, float, int], ...]:
    """
    Create a lightweight cache key from file metadata.
    Streamlit will rebuild the vector store if one of the PDFs changes.
    """
    pdf_paths = validate_document_folder(documents_dir)
    return tuple(
        (pdf.name, pdf.stat().st_mtime, pdf.stat().st_size)
        for pdf in pdf_paths
    )


@st.cache_resource(show_spinner="Loading policy documents...")
def build_knowledge_base(
    documents_dir: str,
    signature: tuple[tuple[str, float, int], ...],
) -> list[Document]:
    """Load and split documents locally. This makes no Groq API calls."""
    _ = signature
    pdf_paths = validate_document_folder(Path(documents_dir))
    page_documents = load_pdf_pages(pdf_paths)
    return split_documents(page_documents)


def format_context(documents: list[Document]) -> str:
    """Render retrieved chunks into a compact prompt context."""
    formatted_chunks = []
    total_chars = 0

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "Unknown page")
        remaining_chars = MAX_CONTEXT_CHARS - total_chars
        if remaining_chars <= 0:
            break

        chunk_text = doc.page_content[:MAX_CHARS_PER_CHUNK]
        chunk = (
            f"Excerpt {index}\n"
            f"Source: {source}, page {page}\n"
            f"{chunk_text}"
        )
        chunk = chunk[:remaining_chars]
        formatted_chunks.append(chunk)
        total_chars += len(chunk)

    return "\n\n---\n\n".join(formatted_chunks)


def format_sources(documents: list[Document]) -> str:
    """Build a deduplicated source list from retrieved document metadata."""
    unique_sources = []
    seen = set()

    for doc in documents:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "Unknown page")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            unique_sources.append(f"- {source}, page {page}")

    return "\n".join(unique_sources) if unique_sources else "- No sources retrieved"


def tokenize_query(text: str) -> set[str]:
    """Create simple search terms for local retrieval without an API call."""
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {
        token
        for token in cleaned.split()
        if len(token) > 2 and token not in STOPWORDS
    }


def local_keyword_search(question: str, documents: list[Document], k: int = TOP_K) -> list[Document]:
    """
    Retrieve chunks locally from loaded document chunks.

    This avoids all embedding API calls. The only Groq call during
    question answering is the final chat generation request.
    """
    query_terms = tokenize_query(question)

    if not query_terms:
        return documents[:k]

    scored_documents = []
    for doc in documents:
        source = str(doc.metadata.get("source", "")).lower()
        page_text = doc.page_content.lower()
        score = 0

        for term in query_terms:
            score += page_text.count(term)
            if term in source:
                score += 2

        if score > 0:
            scored_documents.append((score, len(doc.page_content), doc))

    scored_documents.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [doc for _, _, doc in scored_documents[:k]]


def recent_chat_history(max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Include a short transcript for conversational continuity."""
    messages = st.session_state.messages[-max_turns * 2 :]
    transcript = []

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        transcript.append(f"{role}: {message['content']}")

    return "\n".join(transcript) if transcript else "No prior conversation."


def answer_question(
    question: str,
    documents: list[Document],
    api_key: str,
) -> tuple[str, str]:
    """Retrieve relevant policy chunks locally and generate a grounded answer."""
    retrieved_docs = local_keyword_search(question, documents, k=TOP_K)
    context = format_context(retrieved_docs)
    sources = format_sources(retrieved_docs)

    user_prompt = f"""Policy excerpts:
{context}

Recent conversation:
{recent_chat_history()}

User question:
{question}

Answer requirements:
- Use only the policy excerpts above.
- If the excerpts do not contain the answer, say that the answer cannot be found in the provided policy documents.
- Cite document names and page numbers in the answer where relevant.
- Be concise, practical, and compliance-focused."""

    llm = ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=0,
        api_key=api_key,
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = response.content if isinstance(response, AIMessage) else str(response)
    return answer, sources


def user_friendly_model_error(exc: Exception) -> str:
    """Convert common Groq quota/model errors into demo-friendly guidance."""
    error_text = str(exc)

    if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
        return (
            "Groq quota or rate limit was reached for this API key/project. "
            "Please wait a short time and try again, switch to a Groq key with available quota, "
            "or check your Groq billing and limits. The app makes only one generation request "
            "per question, but Groq is currently refusing generation requests for this key."
        )

    if "NOT_FOUND" in error_text or "404" in error_text:
        return (
            "The configured Groq model is not available for this API key. "
            "Check the CHAT_MODEL value in app.py or set GROQ_MODEL in your .env file "
            "to a model available in your Groq account."
        )

    return (
        "I could not generate an answer because an application error occurred. "
        "Please check the app logs and configuration."
    )


def initialize_session_state() -> None:
    """Create chat state the first time the app loads."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready = False


def render_sidebar(api_key: str | None, signature: tuple[tuple[str, float, int], ...] | None) -> None:
    """Render sidebar description, reset action, and disclaimer."""
    with st.sidebar:
        st.markdown(
            "Ask questions about Nova Insurance Group's internal data privacy, "
            "AI usage, and compliance policy documents."
        )

        st.divider()
        st.markdown("**Knowledge Base**")
        if st.session_state.knowledge_base_ready:
            st.success("Policy documents are loaded.")
        else:
            st.info("Click below to load policy documents before chatting.")

        disabled = not api_key or signature is None
        if st.button("Load Documents", use_container_width=True, disabled=disabled):
            try:
                with st.spinner("Loading policy documents..."):
                    st.session_state.knowledge_base = build_knowledge_base(
                        str(DOCUMENTS_DIR),
                        signature,
                    )
                    st.session_state.knowledge_base_ready = True
                st.rerun()
            except Exception as exc:
                st.session_state.knowledge_base = None
                st.session_state.knowledge_base_ready = False
                st.error(str(exc))

        if st.button("New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.warning(
            "This chatbot provides guidance based on internal policy documents. "
            "It does not constitute legal advice. Always verify critical information "
            "with the Legal & Compliance team."
        )


def render_chat_history() -> None:
    """Display prior user and assistant messages in chat bubbles."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**Sources**")
                st.markdown(message["sources"])


def render_startup_error(message: str) -> None:
    """Display setup errors without crashing the Streamlit page."""
    st.error(message)
    st.info(
        "Confirm that the five required PDFs are in ./documents/ and that "
        "GROQ_API_KEY is set in your environment before running the app."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="NIG", layout="centered")
    initialize_session_state()

    st.title(APP_TITLE)

    api_key = get_api_key()
    if not api_key:
        render_sidebar(api_key=None, signature=None)
        render_startup_error("GROQ_API_KEY is not set.")
        return

    try:
        signature = document_signature(DOCUMENTS_DIR)
    except Exception as exc:
        render_sidebar(api_key=api_key, signature=None)
        render_startup_error(str(exc))
        return

    render_sidebar(api_key=api_key, signature=signature)
    render_chat_history()

    if not st.session_state.knowledge_base_ready or st.session_state.knowledge_base is None:
        st.info("Click **Load Documents** in the sidebar to prepare the policy knowledge base.")
        st.chat_input("Load documents before asking a question", disabled=True)
        return

    question = st.chat_input("Ask a compliance or policy question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing policy documents..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state.knowledge_base,
                    api_key,
                )
            except Exception as exc:
                answer = user_friendly_model_error(exc)
                sources = "- No sources available"
                st.error(answer)

        st.markdown(answer)
        st.markdown("**Sources**")
        st.markdown(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()
