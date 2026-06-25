import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

from adaptive_rag import build_adaptive_graph
from ingestion import chunk_documents
from rag_chain import format_docs, make_chain
from retriever import build_inmemory_retriever

st.set_page_config(page_title="Graph-RAG", page_icon="🔎", layout="centered")

@st.cache_resource(show_spinner="Initializing the search system (connect Neo4j, load Chroma, reranker)...")
def get_adaptive_app():
    return build_adaptive_graph()

@st.cache_resource
def get_chain():
    return make_chain()

@st.cache_resource(show_spinner="Index uploaded PDFs (Load → Chunk → Vector + BM25)…")
def get_uploaded_retriever(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try: 
        pages = PyPDFLoader(tmp_path).load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    for p in pages:
        p.metadata["source"] = file_name
    chunks = chunk_documents(pages)
    return build_inmemory_retriever(chunks), len(chunks)

def extract_sources(documents):
    out = []
    for d in documents:
        meta = d.metadata
        if meta.get("type") == "graph_facts":
            out.append({"label": "Knowledge graph facts", "content": d.page_content, "is_graph": True})
        else:
            src = meta.get("source", "knowledge-graph")
            page = meta.get("page")
            label = f"{src} p.{page + 1}" if page is not None else src
            out.append({"label": label, "content": d.page_content, "is_graph": False})
    return out

def render_meta(meta):
    if meta["mode"] == "uploaded":
        st.markdown(":orange[● uploaded PDF] · vector + BM25 + rerank (No graph, no routing)")
        st.caption(f"📤 {meta['n_chunks']} chunks indexed from your upload") 
    elif meta["route"] == "simple":
        st.markdown(":green[● simple route] · vector + BM25 + rerank")
        st.caption(f"router: {meta['reasoning']}")
    else:
        st.markdown(":blue[● complex route] · vector + BM25 + graph + rerank")
        st.caption(f"router: {meta['reasoning']}")

def render_sources(sources):
    chunks = [s for s in sources if not s["is_graph"]]
    facts = [s for s in sources if s["is_graph"]]
    suffix = ", including graph facts" if facts else ""
    with st.expander(f"📄 Search source ({len(chunks)} chunks{suffix})"):
        for s in chunks:
            st.markdown(f"**{s['label']}**")
            st.caption(s["content"][:500].strip())
        for s in facts:
            st.markdown(f"**🕸️ {s['label']}**")
            st.caption(s["content"])    

with st.sidebar:
    st.title("🔎 Graph-RAG")
    mode = st.radio("Question and Answer Target", ["📚 Pre-built paper library", "📤 Uploaded PDF"])

    uploaded_file = None
    if mode == "📤 Uploaded PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        st.caption(
            "Upload files to **real-time 2-way index** (vector + BM25 + rerank) - no knowledge graph is built,"
            "Because graph construction is a minute-level offline operation, it is not suitable for interactive requests. Complete GraphRAG +"
            "Adaptive routing is only available in the pre-built paper library."
        )

    st.divider()
    st.caption("**Stack:** LangChain · ChromaDB · Neo4j · FlashRank · LangGraph · LangSmith")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()    

# --- Main ---
st.title("Paper Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg["meta"])
            render_sources(msg["sources"])

upload_mode = mode == "📤 Uploaded PDF"
if upload_mode and uploaded_file is None:
    st.info("👈 Please upload a PDF on the left before starting your question.")

if question := st.chat_input("Enter your question..."):
    if upload_mode and uploaded_file is None:
        st.warning("Please upload the PDF first before asking any questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if upload_mode:
            retriever, n_chunks = get_uploaded_retriever(uploaded_file.getvalue(), uploaded_file.name)
            with st.spinner("Retrieving + Generating..."):
                docs = retriever.invoke(question)
                answer = get_chain().invoke({"context": format_docs(docs), "question": question})
            meta = {"mode": "uploaded", "n_chunks": n_chunks}
            sources = extract_sources(docs)
        else:
            with st.spinner("Retrieving + Generating..."):
                final = get_adaptive_app().invoke({"question": question})
            answer = final["answer"]
            meta = {"mode": "prebuilt", "route": final["route"], "reasoning": final["route_reasoning"]}
            sources = extract_sources(final["documents"])

        st.markdown(answer)
        render_meta(meta)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta, "sources": sources}
    )
