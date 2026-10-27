
import os
from io import BytesIO
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from dotenv import load_dotenv
from docx import Document as WordDocument
from pypdf import PdfReader

from google import genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(
    page_title="TPC School Calendar RAG",
    page_icon="📅",
    layout="wide",
)

st.title("📅 TPC School Calendar Assistant")
st.caption(
    "Ask questions about school schedules using the documents "
    "provided to this application."
)

st.info(
    "Answers are based on the uploaded or provided documents. "
    "Free-tier model availability and API quotas may limit usage."
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def extract_text(filename, file_bytes):
    """Extract text from supported document formats."""
    extension = Path(filename).suffix.lower()

    try:
        if extension == ".txt":
            return file_bytes.decode("utf-8", errors="replace")

        if extension == ".pdf":
            reader = PdfReader(BytesIO(file_bytes))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        if extension == ".docx":
            document = WordDocument(BytesIO(file_bytes))
            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            )

    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return ""

    return ""


def load_local_documents():
    """Load supported files from the project's Documents folder."""
    folder = Path(__file__).resolve().parent / "Documents"
    documents = []

    if not folder.exists():
        return documents

    for path in folder.iterdir():
        if path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue

        try:
            text = extract_text(path.name, path.read_bytes())
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name},
                    )
                )
        except OSError as exc:
            st.warning(f"Could not read {path.name}: {exc}")

    return documents


def fetch_tpc_news(website_url):
    """Fetch readable TPC articles from an official TPC website page."""
    parsed_url = urlparse(website_url)
    hostname = (parsed_url.hostname or "").lower()
    if (
        parsed_url.scheme not in {"http", "https"}
        or not (hostname == "tpc.edu.ph" or hostname.endswith(".tpc.edu.ph"))
    ):
        raise ValueError("Enter a URL from the official tpc.edu.ph website.")

    response = requests.get(website_url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup.select(
        "script, style, nav, footer, header, noscript, form, iframe"
    ):
        tag.decompose()

    article_urls = []
    if soup.find("article"):
        article_urls.append(website_url)
    else:
        content = soup.find("main") or soup.find("body")
        if content:
            for link in content.select("a[href]"):
                article_url = urljoin(website_url, link["href"])
                article_parsed = urlparse(article_url)
                article_hostname = (article_parsed.hostname or "").lower()
                if (
                    article_parsed.scheme not in {"http", "https"}
                    or not (
                        article_hostname == "tpc.edu.ph"
                        or article_hostname.endswith(".tpc.edu.ph")
                    )
                    or article_parsed.path.rstrip("/")
                    == parsed_url.path.rstrip("/")
                    or article_url in article_urls
                ):
                    continue
                article_urls.append(article_url)
                if len(article_urls) >= 10:
                    break

    documents = []
    for article_url in article_urls:
        try:
            article_response = requests.get(article_url, timeout=20)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(
                article_response.text, "html.parser"
            )
            for tag in article_soup.select(
                "script, style, nav, footer, header, "
                "noscript, form, iframe"
            ):
                tag.decompose()
            content = (
                article_soup.find("article")
                or article_soup.find("main")
                or article_soup.find("body")
            )
            if content is None:
                continue
            title_tag = content.find("h1") or article_soup.find("h1")
            title = (
                title_tag.get_text(" ", strip=True)
                if title_tag
                else article_url.rstrip("/").split("/")[-1]
            )
            text = content.get_text("\n", strip=True)
            if len(text) < 100:
                continue
            documents.append(
                Document(
                    page_content=(
                        f"Title: {title}\n"
                        f"URL: {article_url}\n\n"
                        f"{text}"
                    ),
                    metadata={
                        "source": article_url,
                        "title": title,
                    },
                )
            )
        except requests.RequestException as exc:
            print(f"Skipped article {article_url}: {exc}")

    if not documents:
        raise ValueError(
            "TPC pages were reached, but no readable article "
            "text was extracted. Try an individual article URL."
        )
    return documents


def build_vectorstore(documents):
    """Split documents, embed the chunks, and store them in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text was found in the documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def generate_answer(question, retrieved_docs):
    """Generate a grounded answer from retrieved RAG passages."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to your .env file."
        )
    passages = []
    for index, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        passages.append(
            f"[Source {index}: {source}]\n{doc.page_content}"
        )
    context = "\n\n".join(passages)
    system_instruction = """You are the TPC School Calendar Assistant.
Answer using only the supplied source passages.
Do not invent events, dates, deadlines, or policies.
If the sources do not support an answer, say:
"I cannot verify the answer from the available school documents."
Cite supporting passages using [Source 1], [Source 2], and so on.
Keep answers clear and concise."""
    client = genai.Client(api_key=api_key)
    response = client.interactions.create(
        model=model,
        system_instruction=system_instruction,
        input=(
            f"SOURCE PASSAGES:\n{context}\n\n"
            f"QUESTION:\n{question}"
        ),
    )
    answer = response.output_text
    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned an empty answer.")
    return answer.strip()


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "startup_knowledge_attempted" not in st.session_state:
    st.session_state.startup_knowledge_attempted = False

if not st.session_state.startup_knowledge_attempted:
    st.session_state.startup_knowledge_attempted = True
    try:
        startup_documents = load_local_documents()
        if startup_documents:
            with st.spinner("Loading the available school documents..."):
                vectorstore, count = build_vectorstore(startup_documents)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunk_count = count
            st.session_state.document_names = sorted(
                {doc.metadata["source"] for doc in startup_documents}
            )
    except Exception as exc:
        st.error(f"Could not load the startup knowledge base: {exc}")


with st.sidebar:
    st.header("Knowledge source")

    uploaded_files = st.file_uploader(
        "Upload school documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload official school-calendar documents when available.",
    )

    website_url = st.text_input(
        "Or import from official TPC website",
        placeholder="https://www.tpc.edu.ph/news",
    )

    if st.button("Build / Refresh Knowledge Base", type="primary"):
        documents = load_local_documents()
        names = [
            doc.metadata["source"]
            for doc in documents
        ]

        if website_url.strip():
            try:
                with st.spinner("Fetching TPC website articles..."):
                    website_docs = fetch_tpc_news(website_url.strip())
                documents.extend(website_docs)
                names.extend(
                    doc.metadata["source"]
                    for doc in website_docs
                )
                st.success(
                    f"Fetched {len(website_docs)} TPC article(s)."
                )
            except Exception as exc:
                st.error(f"Website import failed: {exc}")
                st.stop()

        if uploaded_files:
            for uploaded in uploaded_files:
                content = extract_text(
                    uploaded.name,
                    uploaded.getvalue(),
                )
                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": uploaded.name},
                        )
                    )
                    names.append(uploaded.name)

        if not documents:
            st.error(
                "No readable documents found. Add official files to "
                "the Documents folder or upload PDF, TXT, or DOCX files."
            )
        else:
            try:
                with st.spinner(
                    "Reading, chunking, embedding, and indexing documents..."
                ):
                    vectorstore, count = build_vectorstore(documents)

                st.session_state.vectorstore = vectorstore
                st.session_state.chunk_count = count
                st.session_state.document_names = sorted(set(names))

                st.success(
                    f"Knowledge base ready: {count} chunks from "
                    f"{len(set(names))} document(s)."
                )
            except Exception as exc:
                st.error(f"Could not build the knowledge base: {exc}")

    with st.expander("📚 Available Knowledge", expanded=True):
        st.caption(
            "Mga paksa at dokumentong kasalukuyang available "
            "sa chatbot knowledge base."
        )
        st.markdown("**Available topics**")
        available_topics = [
            "📶 Free Campus Wi-Fi and SSID",
            "🛡️ Campus Security Protocols and Gate Schedules",
            "📅 Start of Classes and Uniform Announcements",
            "📝 Enrollment Advisories and Requirements",
            "🎓 Grade Slips and Enrollment Schedules",
            "🏫 TPC Contact Information",
        ]
        for topic in available_topics:
            st.markdown(f"- {topic}")
        st.divider()
        st.markdown("**Indexed documents and sources**")
        indexed_names = st.session_state.get("document_names", [])
        if indexed_names:
            for document_name in indexed_names:
                st.markdown(f"- `{document_name}`")
        else:
            st.info("Loading available documents...")
        st.caption(
            "Note: Ang sagot ay limitado sa impormasyon "
            "na aktuwal na na-index. Maaaring historical na "
            "announcement ang ilang impormasyon."
        )

if st.session_state.vectorstore is None:
    st.warning(
        "Build the knowledge base from the sidebar to get document-grounded "
        "answers. You can enter a question below once it is ready."
    )
else:
    st.success(
        f"Knowledge base loaded: "
        f"{st.session_state.chunk_count} text chunks."
    )

question = st.chat_input("Ask a question about the school calendar")

if question:
    st.chat_message("user").write(question)

    if st.session_state.vectorstore is None:
        st.warning(
            "Build the knowledge base from the sidebar before asking "
            "questions."
        )
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving relevant passages..."):
                    results = (
                        st.session_state.vectorstore
                        .similarity_search_with_score(question, k=4)
                    )

                if not results:
                    st.warning(
                        "I cannot verify the answer from the available "
                        "school documents."
                    )
                else:
                    retrieved_docs = [doc for doc, score in results]

                    answer = generate_answer(question, retrieved_docs)
                    st.markdown(answer)

                    with st.expander("View retrieved source passages"):
                        for index, (doc, score) in enumerate(
                            results, start=1
                        ):
                            source = doc.metadata.get(
                                "source", "Unknown source"
                            )
                            st.markdown(f"**[Source {index}] {source}**")
                            st.caption(
                                f"Retrieval distance: {float(score):.4f}"
                            )
                            st.write(doc.page_content)
                            st.divider()

            except Exception as exc:
                st.error(
                    "The question could not be processed. Check your "
                    "Hugging Face token, model availability, and network "
                    f"connection. Details: {exc}"
                )
