"""
app.py - Streamlit frontend for the RAG API.

    streamlit run app.py

Env:
    API_URL   backend base URL (default http://localhost:8000)
    API_KEY   sent as X-API-Key if set
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests
import streamlit as st

st.set_page_config(page_title="Multi-Doc RAG Chat", page_icon="📚", layout="wide")

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("API_KEY", "")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def stream_chat(
    api_url: str, api_key: str, question: str, history: list[dict[str, str]], holder: dict[str, Any]
) -> Iterator[str]:
    """
    Generator yielding answer tokens from the SSE stream. Side-channel data
    (sources, cache flag) is written into `holder` so the caller can render cards.
    """
    with requests.post(
        f"{api_url}/chat",
        json={"question": question, "history": history, "stream": True},
        headers=_headers(api_key),
        stream=True,
        timeout=(5, 180),
    ) as resp:
        if resp.status_code == 429:
            raise RuntimeError("Rate limit hit. Wait a moment and try again.")
        if resp.status_code == 401:
            raise RuntimeError("Unauthorized: check the API key in the sidebar.")
        resp.raise_for_status()
        resp.encoding = "utf-8"  # SSE has no charset header; requests would guess latin-1

        event = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "sources":
                    holder["sources"] = data.get("sources", [])
                elif event == "token":
                    yield data["text"]
                elif event == "done":
                    holder["cached"] = data.get("cached", False)
                elif event == "error":
                    raise RuntimeError(data.get("detail", "Unknown server error"))


def render_sources(sources: list[dict[str, Any]]) -> None:
    """One bordered card per cited chunk. The number matches [n] in the answer."""
    if not sources:
        return
    st.markdown(f"**Sources ({len(sources)})**")
    for s in sources:
        with st.container(border=True):
            loc = f" · p.{s['page']}" if s.get("page") else ""
            icon = "📄" if s.get("type") == "pdf" else "🌐"
            st.markdown(f"**[{s['id']}]** {icon} `{s['source']}`{loc}")
            if s.get("score") is not None:
                st.caption(f"rerank score: {s['score']}")
            st.caption(s.get("snippet", ""))


# --------------------------------------------------------------------------- #
# Sidebar: connection + ingestion
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Connection")
    api_url = st.text_input("API URL", DEFAULT_API_URL).rstrip("/")
    api_key = st.text_input("API key (optional)", DEFAULT_API_KEY, type="password")

    st.header("📥 Add documents")
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    urls = st.text_area("URLs (one per line)", height=90, placeholder="https://example.com/article")

    if st.button("Ingest", type="primary", use_container_width=True):
        if not uploads and not urls.strip():
            st.warning("Add a PDF or a URL first.")
        else:
            files = [("files", (u.name, u.getvalue(), "application/pdf")) for u in uploads]
            with st.spinner("Chunking and embedding... large PDFs take a while."):
                try:
                    r = requests.post(
                        f"{api_url}/ingest",
                        files=files or None,
                        data={"urls": urls},
                        headers=_headers(api_key),
                        timeout=600,
                    )
                    body = r.json()
                    for item in body.get("results", []):
                        if item["status"] == "ok":
                            st.success(f"{item['source']}: +{item['chunks_added']} chunks")
                        else:
                            st.error(f"{item['source']}: {item.get('detail')}")
                    if "detail" in body:
                        st.error(body["detail"])
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

    st.header("🗂️ Indexed")
    try:
        r = requests.get(f"{api_url}/sources", headers=_headers(api_key), timeout=5)
        if r.ok:
            items = r.json().get("sources", [])
            if not items:
                st.caption("Nothing ingested yet.")
            for it in items:
                st.caption(f"{'📄' if it['type'] == 'pdf' else '🌐'} {it['source']} ({it['chunks']} chunks)")
        else:
            st.caption(f"Backend responded {r.status_code}.")
    except Exception:
        st.caption("Backend unreachable.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
st.title("📚 Multi-Document RAG Chat")
st.caption("Hybrid retrieval (BM25 + vectors) → cross-encoder rerank → cited answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("cached"):
                st.caption("⚡ served from cache")
            render_sources(m.get("sources", []))

if question := st.chat_input("Ask something about your documents..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        holder: dict[str, Any] = {}
        try:
            answer = st.write_stream(stream_chat(api_url, api_key, question, history, holder))
            if holder.get("cached"):
                st.caption("⚡ served from cache")
            render_sources(holder.get("sources", []))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": holder.get("sources", []),
                    "cached": holder.get("cached", False),
                }
            )
        except Exception as exc:
            st.error(f"Request failed: {exc}")
