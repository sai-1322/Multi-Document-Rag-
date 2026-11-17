# Multi-Document RAG System

Hybrid retrieval (BM25 + FAISS) → cross-encoder reranking → streamed, cited answers.
FastAPI backend, Streamlit UI, Redis cache, Docker Compose.

```
PDF / URL ──► split ──► embed ──► FAISS (dense) ─┐
                    └─► BM25 (keyword) ──────────┼─► RRF fusion (top 20) ─► cross-encoder ─► top 4 ─► LLM ─► answer + [n] citations
                                                 ┘
```

| File | Purpose |
|---|---|
| `rag_pipeline.py` | Ingestion, hybrid retriever, reranker, conversational chain |
| `main.py` | FastAPI: `/ingest`, `/chat` (SSE), rate limiting, caching |
| `app.py` | Streamlit chat UI with uploads, streaming, source cards |
| `evaluate.py` | Baseline-vs-pipeline benchmark with confidence intervals |
| `Dockerfile` / `docker-compose.yml` | Multi-stage image (targets `api`, `ui`) + Redis |

---

## 1. Configure keys

```bash
cp .env.example .env
```

Edit `.env`:

- `OPENAI_API_KEY` – required (or point `OPENAI_BASE_URL` at any OpenAI-compatible server and put any non-empty value as the key).
- `API_KEY` – a long random string. It protects `/ingest` and `/chat`. Leaving it empty means **anyone who can reach port 8000 can burn your OpenAI credits**. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- `LLM_MODEL` – default `gpt-4o-mini`. If your provider has retired it, set whatever chat model you have access to.

## 2. Build and run with Docker Compose

```bash
docker compose up --build
```

First build takes several minutes (CPU torch + model download baked into the image). Then:

- Streamlit UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

The index persists in the `rag_data` volume, so it survives restarts. `docker compose down -v` deletes it.

## 3. Use it

**UI:** upload PDFs / paste URLs in the sidebar → *Ingest* → ask questions. Answers stream in; each `[n]` in the text maps to a source card.

**curl:**

```bash
# ingest
curl -X POST localhost:8000/ingest -H "X-API-Key: $API_KEY" \
  -F "files=@paper.pdf" -F "urls=https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

# streamed chat (SSE)
curl -N -X POST localhost:8000/chat -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is hybrid retrieval?", "history": []}'

# plain JSON instead of SSE
curl -X POST localhost:8000/chat -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is hybrid retrieval?", "stream": false}'
```

## 4. Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, avoids CUDA wheels
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
uvicorn main:app --port 8000          # terminal 1 (keep ONE worker)
streamlit run app.py                  # terminal 2
```

## 5. Evaluate

Ingest your documents first (the eval reads the persisted index).

```bash
# generate 100 synthetic questions from your indexed chunks AND evaluate them
docker compose run --rm api python evaluate.py --generate 100 --llm-judge

# later: re-run on the frozen question set (comparable across config changes)
docker compose run --rm api python evaluate.py --dataset /data/eval_set.jsonl --llm-judge
```

Output: a table comparing `baseline_dense`, `hybrid` (ablation) and `hybrid_rerank`, relative improvement with a 95% bootstrap CI, and a verdict against `--target-boost` (default 22). The report is saved to `/data/eval_report.json`.

### About the "22% boost"

I did not hard-code it and you should not claim it unless the script prints it. The script gives you the honest number for **your** corpus:

- Needs ≥100 questions (the default) or the CI will be too wide to mean anything.
- Synthetic questions favour keyword overlap with their source chunk. For a defensible claim, hand-write 100 realistic questions and label the gold chunk IDs in the same JSONL format.
- Using the same LLM to answer and judge is self-preference-biased; set `JUDGE_MODEL` to a different model.
- If you only measure `hit@k`/`MRR`, say "retrieval improvement", not "answer relevance".

## 6. Production caveats (know these before deploying)

- **One worker only.** The index is in process memory. `--workers 4` gives you four diverging indexes. To scale horizontally, move to a real vector store (Qdrant, pgvector, OpenSearch) and make the API stateless.
- **BM25 is rebuilt on every ingest**, O(corpus). Fine to ~100k chunks.
- **Scanned PDFs** have no text layer; `/ingest` returns an error for them. Add OCR if you need them.
- **JS-rendered pages** return little or no text through `WebBaseLoader`.
- **Prompt injection:** ingested web pages are untrusted input. The prompt mitigates it, nothing eliminates it. Don't give this system tools or private data you wouldn't show the page author.
- **SSRF:** `/ingest` blocks private IPs, but not DNS rebinding or redirects. Use an egress proxy in hostile environments.
- **Cache:** keyed on question + recent history + corpus fingerprint, so ingesting new docs invalidates old answers. Low hit rate for multi-turn chats is expected.
- **Not load-tested.** Reranking is CPU-bound; measure latency on your hardware before promising anything.
- LangChain is pinned to `<0.4` on purpose. Upgrading to 1.x needs a retest (`EnsembleRetriever` moved to `langchain-classic`).

## 7. Tuning knobs (`.env`)

| Variable | Effect |
|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Biggest single lever on quality. Re-ingest after changing. |
| `CANDIDATE_K` | Candidates passed to the reranker (higher = better recall, slower). |
| `FINAL_K` | Chunks sent to the LLM. |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | Fusion weights. Tune with `evaluate.py`, not by feel. |
| `MIN_RERANK_SCORE` | Noise floor; chunks scoring below it are dropped (and the bot says "not found"). |
