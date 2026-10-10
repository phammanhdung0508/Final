#!/usr/bin/env python3
"""Benchmark recommendation serving throughput (QPS) and latency.

Evaluates top-k candidate scoring and filtering across the full item catalog.
Supports tuning batch size, embedding precision (FP32, FP16), thread count,
and ranking algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark recommendation serving performance.")
    parser.add_argument("--num-users", type=int, default=610, help="Number of users to serve")
    parser.add_argument("--num-items", type=int, default=9742, help="Catalog size")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations per user")
    parser.add_argument("--batch-size", type=int, default=64, help="Serving batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="Embedding precision")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu, cuda)")
    parser.add_argument("--append-tsv", type=Path, default=None, help="Append row to results.tsv")
    parser.add_argument("--experiment-id", default="baseline-serving", help="Experiment identifier")
    parser.add_argument("--notes", default="", help="Experimental notes")
    return parser.parse_args()


def benchmark_serving(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )

    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    # Initialize synthetic or loaded embeddings
    rng = np.random.default_rng(42)
    user_emb = torch.randn(args.num_users, args.embedding_dim, dtype=dtype, device=device)
    item_emb = torch.randn(args.num_items, args.embedding_dim, dtype=dtype, device=device)

    # Warmup
    for _ in range(args.warmup_runs):
        _ = torch.topk(user_emb[:args.batch_size] @ item_emb.T, k=args.top_k, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_per_query: list[float] = []
    total_queries_served = 0
    start_total = time.perf_counter()

    for _ in range(args.runs):
        for start_idx in range(0, args.num_users, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.num_users)
            b_users = user_emb[start_idx:end_idx]
            b_size = end_idx - start_idx

            t0 = time.perf_counter()
            logits = b_users @ item_emb.T
            topk = torch.topk(logits, k=args.top_k, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_time = (t1 - t0) * 1000.0  # ms
            query_time = step_time / b_size
            for _ in range(b_size):
                latencies_per_query.append(query_time)
            total_queries_served += b_size

    total_time = time.perf_counter() - start_total
    qps = total_queries_served / total_time

    p50 = float(np.percentile(latencies_per_query, 50))
    p95 = float(np.percentile(latencies_per_query, 95))
    p99 = float(np.percentile(latencies_per_query, 99))
    mean_latency = float(np.mean(latencies_per_query))

    metrics = {
        "qps": qps,
        "tokens_per_second_mean": qps,
        "tokens_per_second_median": qps,
        "seconds_mean": mean_latency / 1000.0,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "mean_latency_ms": mean_latency,
        "total_queries": total_queries_served,
        "total_seconds": total_time,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "device": str(device),
        "threads": args.threads,
    }

    print("--- Recommendation Serving Benchmark ---")
    print(f"tok/s: {qps:.2f} (QPS)")
    print(f"tokens_per_second_mean: {qps:.2f}")
    print(f"queries_per_second: {qps:.2f}")
    print(f"latency_p50_ms: {p50:.4f}")
    print(f"latency_p95_ms: {p95:.4f}")
    print(f"latency_p99_ms: {p99:.4f}")
    print(f"total_queries: {total_queries_served}")
    print(f"total_seconds: {total_time:.3f}")

    if args.append_tsv:
        tsv_path = args.append_tsv
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp": now_iso,
            "experiment_id": args.experiment_id,
            "model": "HeteroGraphSAGE",
            "server_command": f"benchmark_serving.py --batch-size {args.batch_size} --precision {args.precision} --threads {args.threads}",
            "base_url": "in-process",
            "runs": str(args.runs),
            "warmup": str(args.warmup_runs),
            "max_tokens": str(args.top_k),
            "prompt_sha256": "n/a",
            "completion_tokens_mean": str(args.top_k),
            "seconds_mean": f"{metrics['seconds_mean']:.6f}",
            "tokens_per_second_mean": f"{qps:.2f}",
            "tokens_per_second_median": f"{qps:.2f}",
            "prompt_tokens_mean": str(args.embedding_dim),
            "success": "true",
            "notes": args.notes or f"Top-{args.top_k} dot-product serving",
            "output_json": json.dumps({"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}),
        }

        file_exists = tsv_path.exists()
        with tsv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended results to {tsv_path}")

    return metrics


if __name__ == "__main__":
    benchmark_serving(parse_args())
