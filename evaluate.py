"""
evaluate.py - measure whether hybrid + rerank actually beats a plain dense baseline.

READ THIS FIRST (no sugar-coating):
    This script MEASURES. It does not, and cannot, "prove" a 22% boost. Whether you
    get +22%, +5% or -3% depends on your corpus, your questions and your models.
    Do not put a number on a resume or slide unless THIS script printed it on YOUR data,
    with a confidence interval that excludes zero.

Systems compared (all return the same final_k chunks):
    baseline_dense   FAISS similarity search only
    hybrid           BM25 + FAISS (EnsembleRetriever), no reranker  <- ablation
    hybrid_rerank    BM25 + FAISS + cross-encoder reranker          <- the full pipeline

Metrics
    hit@1, hit@k, MRR      retrieval ground truth = the chunk the question was written from
    answer_relevance       (--llm-judge) LLM grades the generated answer 1-5 against the
                           question + the gold passage, normalised to 0-1

Usage
    # 1) ingest your documents via the API first, then:
    python evaluate.py --generate 100 --llm-judge
    # 2) re-run later on the same frozen question set:
    python evaluate.py --dataset data/eval_set.jsonl --llm-judge

Caveats on synthetic questions:
    Questions are LLM-written from sampled chunks, so they overlap lexically with the gold
    chunk (biases toward BM25) and other chunks may also answer them (understates recall).
    For a claim you will defend, hand-label 100 real user questions into the same JSONL
    format: {"question": ..., "gold_chunk_id": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from rag_pipeline import RAGPipeline

QGEN_PROMPT = (
    "Below is a passage from a document. Write ONE question that a person who has NOT "
    "seen the passage might ask, which this passage answers.\n"
    "Rules: do not copy distinctive phrases verbatim - paraphrase; do not mention 'the "
    "passage' or 'the document'; the question must be self-contained and specific.\n"
    "Return only the question.\n\nPASSAGE:\n{passage}"
)

JUDGE_PROMPT = (
    "You are a strict evaluator. Score the ANSWER from 1 to 5 for how directly, correctly "
    "and completely it answers the QUESTION, using the REFERENCE PASSAGE as ground truth.\n"
    "1 = off-topic, wrong, or says it cannot find the answer although the reference contains it\n"
    "3 = partially answers or is partly wrong\n"
    "5 = directly and fully answers, consistent with the reference\n"
    "Reply with a single digit only.\n\n"
    "QUESTION: {question}\n\nREFERENCE PASSAGE:\n{reference}\n\nANSWER:\n{answer}"
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def sample_chunks(chunks: list[Document], n: int, seed: int) -> list[Document]:
    """Round-robin across sources so one big document doesn't dominate the test set."""
    rng = random.Random(seed)
    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        if len(c.page_content) >= 300:  # skip stubby chunks that make unanswerable questions
            by_source[c.metadata["source"]].append(c)
    for lst in by_source.values():
        rng.shuffle(lst)
    picked: list[Document] = []
    sources = sorted(by_source)
    while len(picked) < n and any(by_source.values()):
        for s in sources:
            if by_source[s] and len(picked) < n:
                picked.append(by_source[s].pop())
    return picked


def generate_dataset(pipeline: RAGPipeline, n: int, out: Path, seed: int, workers: int) -> list[dict]:
    picked = sample_chunks(pipeline.all_chunks(), n, seed)
    if len(picked) < n:
        print(f"WARNING: only {len(picked)} suitable chunks available (wanted {n}). "
              "Ingest more documents for a statistically meaningful test.", file=sys.stderr)

    llm = ChatOpenAI(
        model=os.getenv("JUDGE_MODEL", pipeline.settings.llm_model),
        temperature=0.3,
        base_url=pipeline.settings.llm_base_url,
    )

    def make(chunk: Document) -> dict | None:
        try:
            q = llm.invoke(QGEN_PROMPT.format(passage=chunk.page_content[:1500])).content.strip()
            if len(q) < 10:
                return None
            return {
                "question": q,
                "gold_chunk_id": chunk.metadata["chunk_id"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata.get("page"),
            }
        except Exception as exc:
            print(f"question generation failed: {exc}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in tqdm(pool.map(make, picked), total=len(picked), desc="Generating questions") if r]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} questions -> {out}")
    return rows


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rank_of(gold: str, docs: list[Document]) -> int | None:
    for i, d in enumerate(docs, 1):
        if d.metadata.get("chunk_id") == gold:
            return i
    return None


def judge_score(judge: ChatOpenAI, question: str, reference: str, answer: str) -> float:
    raw = judge.invoke(
        JUDGE_PROMPT.format(question=question, reference=reference[:2000], answer=answer)
    ).content
    m = re.search(r"[1-5]", raw)
    return (int(m.group()) - 1) / 4 if m else 0.0  # 1..5 -> 0..1; unparseable counts as worst


def bootstrap_ci(base: np.ndarray, cand: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% paired-bootstrap CI for the RELATIVE improvement (%) of mean(cand) over mean(base)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=(n, len(base)))
    b, c = base[idx].mean(axis=1), cand[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (c - b) / b * 100
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rel, [2.5, 97.5])
    return float(lo), float(hi)


def rel_improvement(base: float, cand: float) -> float:
    return float("nan") if base == 0 else (cand - base) / base * 100


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", type=int, metavar="N", help="synthesise N questions from the indexed chunks")
    ap.add_argument("--dataset", type=Path, help="JSONL question set (default: --out)")
    ap.add_argument("--out", type=Path, default=data_dir / "eval_set.jsonl")
    ap.add_argument("--report", type=Path, default=data_dir / "eval_report.json")
    ap.add_argument("--k", type=int, default=4, help="chunks given to the LLM / scored (default 4)")
    ap.add_argument("--llm-judge", action="store_true", help="also generate answers and grade relevance (costs API calls)")
    ap.add_argument("--target-boost", type=float, default=22.0, help="relative %% improvement to test against")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    pipeline = RAGPipeline()
    if pipeline.chunk_count == 0:
        print(f"No indexed chunks in {data_dir}. Ingest documents first.", file=sys.stderr)
        return 1

    if args.generate:
        rows = generate_dataset(pipeline, args.generate, args.out, args.seed, args.workers)
    else:
        path = args.dataset or args.out
        if not path.exists():
            print(f"{path} not found. Use --generate 100 first.", file=sys.stderr)
            return 1
        rows = load_dataset(path)
    if not rows:
        print("Empty question set.", file=sys.stderr)
        return 1

    k = args.k
    systems: dict[str, Callable[[str], list[Document]]] = {
        "baseline_dense": lambda q: pipeline.retrieve_dense(q, k),
        "hybrid": lambda q: pipeline.retrieve_hybrid(q)[:k],
        "hybrid_rerank": lambda q: pipeline.retrieve(q, final_k=k),
    }
    chunk_text = {c.metadata["chunk_id"]: c.page_content for c in pipeline.all_chunks()}

    judge = None
    if args.llm_judge:
        judge = ChatOpenAI(
            model=os.getenv("JUDGE_MODEL", pipeline.settings.llm_model),
            temperature=0,
            base_url=pipeline.settings.llm_base_url,
        )

    per_q: dict[str, dict[str, list[float]]] = {
        name: {"hit1": [], "hitk": [], "mrr": [], "latency_ms": [], "answer_relevance": []}
        for name in systems
    }

    for row in tqdm(rows, desc="Evaluating"):
        q, gold = row["question"], row["gold_chunk_id"]
        for name, retrieve in systems.items():
            t0 = time.perf_counter()
            docs = retrieve(q)
            per_q[name]["latency_ms"].append((time.perf_counter() - t0) * 1000)
            r = rank_of(gold, docs)
            per_q[name]["hit1"].append(1.0 if r == 1 else 0.0)
            per_q[name]["hitk"].append(1.0 if r else 0.0)
            per_q[name]["mrr"].append(1.0 / r if r else 0.0)
            if judge is not None:
                try:
                    answer = pipeline.generate(q, docs) if docs else "I could not find that."
                    per_q[name]["answer_relevance"].append(
                        judge_score(judge, q, chunk_text.get(gold, ""), answer)
                    )
                except Exception as exc:
                    print(f"judge failed: {exc}", file=sys.stderr)
                    per_q[name]["answer_relevance"].append(0.0)

    # ------------------------------ report ------------------------------
    metrics = ["hit1", "hitk", "mrr"] + (["answer_relevance"] if judge else []) + ["latency_ms"]
    means = {s: {m: float(np.mean(per_q[s][m])) for m in metrics} for s in systems}

    print(f"\n{len(rows)} questions, k={k}\n")
    header = f"{'metric':<18}" + "".join(f"{s:>18}" for s in systems)
    print(header + "\n" + "-" * len(header))
    for m in metrics:
        print(f"{m:<18}" + "".join(f"{means[s][m]:>18.3f}" for s in systems))

    primary = "answer_relevance" if judge else "mrr"
    print(f"\nRelative improvement over baseline_dense (primary metric: {primary})")
    report: dict[str, Any] = {"n_questions": len(rows), "k": k, "primary_metric": primary, "means": means, "improvement": {}}
    base_arr = np.array(per_q["baseline_dense"][primary])
    for s in ("hybrid", "hybrid_rerank"):
        imp = rel_improvement(means["baseline_dense"][primary], means[s][primary])
        lo, hi = bootstrap_ci(base_arr, np.array(per_q[s][primary]))
        report["improvement"][s] = {"relative_pct": imp, "ci95": [lo, hi]}
        print(f"  {s:<15} {imp:+7.1f}%   95% CI [{lo:+.1f}%, {hi:+.1f}%]")

    full = report["improvement"]["hybrid_rerank"]
    met = full["relative_pct"] >= args.target_boost
    significant = full["ci95"][0] > 0
    print(f"\nTarget +{args.target_boost:.0f}%: ", end="")
    if met and significant:
        print(f"MET on this dataset (measured {full['relative_pct']:+.1f}%, CI excludes 0).")
    elif met:
        print(f"point estimate {full['relative_pct']:+.1f}% reaches the target, but the CI includes 0: NOT reliable.")
    else:
        print(f"NOT MET (measured {full['relative_pct']:+.1f}%). Report that number, not the target.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
