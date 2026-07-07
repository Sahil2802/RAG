# Harness that drives the eval run: loops over the testset, calls retrieve()
# + generate() per question, and hands each (question, answer, contexts) off
# to ragas_eval.score_single() for the actual RAGAS judging. This file does
# not compute any RAGAS scores itself -- it only aggregates the per-question
# scores ragas_eval.py returns (mean/p50/p95) plus its own metrics that RAGAS
# doesn't cover (retrieval hit_rate, latency).
import os
import sys
import json
import time
import random
import statistics
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "backend"
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "results"
TESTSET_PATH = EVAL_DIR / "testset.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
STORE_DIR = str(BACKEND_DIR / "qdrant_storage")

sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

# guards against accidentally kicking off a slow, local-LLM-judged eval run --
# set RAGAS_EVAL_ENABLED=true (e.g. in backend/.env) when you actually want to run it
RAGAS_EVAL_ENABLED = os.getenv("RAGAS_EVAL_ENABLED", "false").lower() == "true"

import observability  # noqa: F401 - activates LangSmith tracing
from vectorstore.qdrant_store import load_store
from embedding.embedder import Embedder
from retriever.retriever import retrieve
from generation.generator import generate
from eval.ragas_eval import score_single

TOP_K = 5
MIN_SIMILARITY = 0.65

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _pct(data: list, p: int) -> float:
    # statistics.quantiles requires at least 2 points, so fall back to the
    # single value (or None) for smoke tests / tiny testsets.
    if len(data) < 2:
        return data[0] if data else None
    return statistics.quantiles(data, n=100)[p - 1]


def _agg_ragas(results: list[dict], key: str) -> dict:
    # Skip entries where the ragas judge failed/returned None for this metric,
    # so a single bad score doesn't skew or crash the aggregate.
    vals = [r["ragas_scores"][key] for r in results if (r["ragas_scores"] or {}).get(key) is not None]
    if not vals:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "p50": round(_pct(vals, 50), 4),
        "p95": round(_pct(vals, 95), 4),
    }


def _agg_lat(ms_list: list[int]) -> dict:
    if not ms_list:
        return {"p50": None, "p95": None}
    return {"p50": _pct(ms_list, 50), "p95": _pct(ms_list, 95)}


def _proportional_sample(testset: list[dict], n: int, seed: int = 42) -> list[dict]:
    # Proportionally draws from each retrieval_challenge category (largest-remainder
    # rounding so the shares sum exactly to n) instead of a flat testset[:n] slice,
    # which would skew toward whichever category is ordered first in the file.
    if n >= len(testset):
        return testset

    by_challenge: dict[str, list[dict]] = {}
    for entry in testset:
        by_challenge.setdefault(entry["retrieval_challenge"], []).append(entry)

    rng = random.Random(seed)
    raw_shares = {ch: n * len(entries) / len(testset) for ch, entries in by_challenge.items()}
    shares = {ch: int(share) for ch, share in raw_shares.items()}
    remainder = n - sum(shares.values())
    for ch, _ in sorted(raw_shares.items(), key=lambda kv: kv[1] - shares[kv[0]], reverse=True)[:remainder]:
        shares[ch] += 1

    sample = []
    for ch, entries in by_challenge.items():
        sample.extend(rng.sample(entries, min(shares[ch], len(entries))))
    rng.shuffle(sample)
    return sample


def _hit_rate(gt_ids: list[dict], retrieved_meta: list[tuple]) -> float | None:
    # Fraction of ground-truth chunks that appear anywhere in the retrieved set.
    # chunk_index is optional in the ground truth: when absent, a paper_id match
    # alone counts as a hit (paper-level, not chunk-level, ground truth).
    if not gt_ids:
        return None
    matched = 0
    for gt in gt_ids:
        for paper_id, chunk_index in retrieved_meta:
            paper_match = paper_id == gt.get("paper_id")
            chunk_match = gt.get("chunk_index") is None or chunk_index == gt.get("chunk_index")
            if paper_match and chunk_match:
                matched += 1
                break
    return round(matched / len(gt_ids), 4)


def main(smoke: int = 0, sample: int = 0):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    if smoke:
        testset = testset[:smoke]
        print(f"Smoke test: running {smoke} questions")
    elif sample:
        testset = _proportional_sample(testset, sample)
        print(f"Proportional sample: running {len(testset)} questions (seed=42)")

    client = load_store(STORE_DIR)
    embedder = Embedder()

    per_results = []
    retrieval_ms_list, generation_ms_list, total_ms_list = [], [], []

    for entry in testset:
        qid = entry["id"]
        question = entry["question"]
        print(f"[{qid}/{len(testset)}] {question[:70]}...")

        t0 = time.perf_counter()
        docs = retrieve(question, client, embedder, top_k=TOP_K, min_similarity=MIN_SIMILARITY)
        t1 = time.perf_counter()
        gen_result = generate(question, docs)
        t2 = time.perf_counter()

        r_ms = round((t1 - t0) * 1000)
        g_ms = round((t2 - t1) * 1000)
        total_ms = round((t2 - t0) * 1000)
        retrieval_ms_list.append(r_ms)
        generation_ms_list.append(g_ms)
        total_ms_list.append(total_ms)

        retrieved_contexts = [d["content"] for d in docs]
        retrieved_ids = [d["id"] for d in docs]
        retrieved_meta = [
            (d["metadata"].get("paper_id"), d["metadata"].get("chunk_index")) for d in docs
        ]

        gt_ids = entry.get("ground_truth_chunk_ids", [])
        hit_rate = _hit_rate(gt_ids, retrieved_meta)

        ragas = score_single(
            question=question,
            answer=gen_result["answer"],
            retrieved_contexts=retrieved_contexts,
            ground_truth_answer=entry["ground_truth_answer"],
        )

        per_results.append({
            "id": qid,
            "question": question,
            "retrieval_challenge": entry["retrieval_challenge"],
            "ground_truth_chunk_ids": gt_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_contexts": retrieved_contexts,
            "generated_answer": gen_result["answer"],
            "ground_truth_answer": entry["ground_truth_answer"],
            "retrieval_hit_rate": hit_rate,
            "ragas_scores": ragas,
            "latency_ms": {"retrieval": r_ms, "generation": g_ms, "total": total_ms},
        })

    # Per-challenge breakdown
    challenges: dict[str, dict[str, list]] = {}
    for r in per_results:
        ch = r["retrieval_challenge"]
        if ch not in challenges:
            challenges[ch] = {m: [] for m in METRIC_KEYS}
        for m in METRIC_KEYS:
            v = (r["ragas_scores"] or {}).get(m)
            if v is not None:
                challenges[ch][m].append(v)

    per_challenge_ragas = {
        ch: {m: round(sum(vs) / len(vs), 4) if vs else None for m, vs in metrics.items()}
        for ch, metrics in challenges.items()
    }

    hit_rates = [r["retrieval_hit_rate"] for r in per_results if r["retrieval_hit_rate"] is not None]

    baseline = {
        "metadata": {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "embedder_model": "BAAI/bge-small-en-v1.5",
            "generator_model": "llama3.2",
            "ragas_judge_model": "llama3.1:8b",
            "ragas_version": importlib.metadata.version("ragas"),
            "retrieval_top_k": TOP_K,
            "retrieval_min_similarity": MIN_SIMILARITY,
            "total_questions": len(testset),
        },
        "aggregate": {
            "ragas_scores": {m: _agg_ragas(per_results, m) for m in METRIC_KEYS},
            "latency_ms": {
                "retrieval": _agg_lat(retrieval_ms_list),
                "generation": _agg_lat(generation_ms_list),
                "total": _agg_lat(total_ms_list),
            },
            "retrieval_hit_rate": {
                "mean": round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else None,
                "p50": round(_pct(hit_rates, 50), 4) if len(hit_rates) >= 2 else None,
                "p95": round(_pct(hit_rates, 95), 4) if len(hit_rates) >= 2 else None,
            },
            "per_challenge_ragas": per_challenge_ragas,
        },
        "results": per_results,
    }

    if smoke:
        out_path = RESULTS_DIR / "smoke.json"
    elif sample:
        out_path = RESULTS_DIR / "sample.json"
    else:
        out_path = BASELINE_PATH
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("\nAggregate RAGAS scores:")
    for m, agg in baseline["aggregate"]["ragas_scores"].items():
        print(f"  {m}: mean={agg['mean']}, p50={agg['p50']}, p95={agg['p95']}")
    print("\nLatency (ms):")
    for phase, agg in baseline["aggregate"]["latency_ms"].items():
        print(f"  {phase}: p50={agg['p50']}, p95={agg['p95']}")


if __name__ == "__main__":
    if not RAGAS_EVAL_ENABLED:
        print(
            "Ragas eval is disabled by default (slow, runs a local LLM judge over the whole testset).\n"
            "Set RAGAS_EVAL_ENABLED=true in backend/.env to run it."
        )
        sys.exit(0)

    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", type=int, default=0, help="Run only the first N questions (smoke test)")
    group.add_argument("--sample", type=int, default=0, help="Run a sample of N questions, drawn proportionally across retrieval_challenge categories")
    args = parser.parse_args()
    main(smoke=args.smoke, sample=args.sample)
