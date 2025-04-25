# -*- coding: utf-8 -*-
"""
Fast contamination checker for JSONL‑based code corpora (16.1 B tokens)
======================================================================
This script now accepts **benchmark files in either JSON or JSONL** format.
Each benchmark item can be a raw string (prompt), or an object with fields
`prompt` / `solution`.  Corpus shards must be JSONL where each line
contains a code snippet under the key specified by `--json_field`
(default: `text`).

Pipeline
--------
1. **Exact‑match scan (The‑Stack style)**
   • Multiprocess stream over all `*.jsonl` shards, flag any record whose
     code field contains a benchmark prompt/solution as an exact
     substring.
2. **MinHash sampling sanity check**
   • Reservoir‑sample up to `--sample N` code snippets; build 256‑perm
     LeanMinHash of 5‑token n‑grams and query against an LSHForest for
     Jaccard ≥ 0.8.
3. **Optional accuracy re‑compute**
   • If contamination is detected, recompute pass@1 excluding affected
     benchmark IDs when a result JSONL is supplied via `--results_json`.

Example
-------
```bash
python contamination_check.py \
  --corpus /mnt/datasets/SwallowCode-16.1B/*.jsonl \
  --bench_json benchmarks/humaneval_prompts.jsonl \
  --json_field text \
  --sample 1000000 \
  --results_json exp/best_run_results.json
```

Dependencies: `pip install datasketch tqdm orjson xxhash`
Author: ChatGPT (o3) 2025‑04‑25
"""
from __future__ import annotations

import argparse, glob, json, os, random, re, sys, multiprocessing as mp
from typing import List, Iterable, Set, Dict

import orjson
from datasketch import MinHash, LeanMinHash, MinHashLSHForest
from tqdm import tqdm

# ───────────────────────────────────────────────────────── Tokenizer ──
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", re.ASCII)

def tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)

# ───────────────────────────────────── Benchmark processing utils ──

def load_benchmark_texts(path: str) -> List[str]:
    """Load benchmark prompts/solutions from JSON or JSONL.

    • JSON   : list[str] **or** list[{"prompt": str, "solution": str}]
    • JSONL  : same as above but one item per line
    """
    texts: List[str] = []
    is_jsonl = path.endswith(".jsonl")
    with open(path, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except Exception:
                    continue
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    texts.append(obj.get("prompt", "") + obj.get("solution", ""))
        else:  # regular JSON
            data = json.load(f)
            if data and isinstance(data[0], str):
                texts = data
            else:
                texts = [d.get("prompt", "") + d.get("solution", "") for d in data]
    return texts

# ─────────────────────────────────────────── Exact‑match phase ──

def _exact_worker(args):
    shard, needles, json_field = args
    hits = []
    with open(shard, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            try:
                code = orjson.loads(line).get(json_field, "")
            except Exception:
                continue
            if any(n in code for n in needles):
                hits.append((shard, ln))
    return hits

# ────────────────────────────────────────────────── MinHash utils ──

def build_minhash(tokens: List[str], num_perm: int = 256) -> LeanMinHash:
    """Return a memory‑efficient LeanMinHash built from tokens."""
    mh = MinHash(num_perm=num_perm)
    for tok in tokens:
        mh.update(tok.encode())
    return LeanMinHash(mh)  # convert once – keeps update bug‑free


def build_bench_sigs(texts: List[str], num_perm=256, ngram=5):
    sigs = {}
    for i, doc in enumerate(texts):
        toks = tokenize(doc)
        grams = [" ".join(toks[j:j + ngram]) for j in range(len(toks) - ngram + 1)]
        sigs[i] = build_minhash(grams, num_perm)
    return sigs

# ───────────────────────────── Reservoir sampling helper ──

def reservoir_sample(reservoir: List[str], new_item: str, k: int, t: int):
    if len(reservoir) < k:
        reservoir.append(new_item)
    else:
        j = random.randint(0, t - 1)
        if j < k:
            reservoir[j] = new_item

# ──────────────────────────────── MinHash sampling phase ──

def minhash_sampling(shards: List[str], bench_sigs, sample_size=1_000_000, ngram=5, threshold=0.8, json_field="text"):
    # Build reservoir (unchanged)
    reservoir: List[str] = []
    t = 0
    for shard in shards:
        with open(shard, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                t += 1
                try:
                    code = orjson.loads(line).get(json_field, "")
                except Exception:
                    continue
                reservoir_sample(reservoir, code, sample_size, t)
    # Build LSHForest for benchmarks
    forest = MinHashLSHForest(num_perm=256)
    for bid, sig in bench_sigs.items():
        forest.add(f"b{bid}", sig)
    forest.index()

    flagged = []
    for rec in tqdm(reservoir, desc="MinHash querying"):
        toks = tokenize(rec)
        if len(toks) < ngram:
            continue
        grams = [" ".join(toks[j:j + ngram]) for j in range(len(toks) - ngram + 1)]
        sig = build_minhash(grams)
        for cid in forest.query(sig, 3):  # top‑3 candidates
            bid = int(cid[1:])
            j_val = sig.jaccard(bench_sigs[bid])
            if j_val >= threshold:
                flagged.append((bid, j_val))
                break
    return flagged

# ─────────────────────────────── Accuracy recomputation ──

def recompute_accuracy(results_jsonl: str, contaminated_ids: Set[str]):
    passed = total = 0
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = orjson.loads(line)
            if rec.get("task_id") in contaminated_ids:
                continue
            total += 1
            if rec.get("pass"):
                passed += 1
    return passed / max(total, 1)

# ───────────────────────────────────────────────────────── main ──

def main():
    ap = argparse.ArgumentParser(description="Fast contamination checker")
    ap.add_argument("--corpus", nargs="+", required=True, help="JSONL shards or glob patterns")
    ap.add_argument("--bench_json", required=True, help="Benchmark file (json or jsonl)")
    ap.add_argument("--json_field", default="text", help="Code field name in corpus JSONL")
    ap.add_argument("--sample", type=int, default=1_000_000)
    ap.add_argument("--results_json", help="Experiment results JSONL for pass@1 recomputation")
    ap.add_argument("--output", default="contamination_report.json")
    args = ap.parse_args()

    # Expand corpus globs
    shards = [p for pat in args.corpus for p in glob.glob(pat)]
    if not shards:
        sys.exit("No corpus shards found.")

    bench_texts = load_benchmark_texts(args.bench_json)
    print(f"Benchmarks: {len(bench_texts)} items")

    # Phase 1: exact match
    print("[1/3] Exact‑match scanning …")
    with mp.Pool() as pool:
        hits_nested = pool.map(_exact_worker, [(sh, bench_texts, args.json_field) for sh in shards])
    exact_hits = [hit for sub in hits_nested for hit in sub]
    print(f"  → {len(exact_hits)} exact hits")

    # Phase 2: MinHash sampling
    print("[2/3] MinHash sampling …")
    bench_sigs = build_bench_sigs(bench_texts)
    mh_hits = minhash_sampling(shards, bench_sigs, args.sample, json_field=args.json_field)
    print(f"  → {len(mh_hits)} collisions ≥0.8 (sample={args.sample})")

    # Phase 3: accuracy recompute (if contamination)
    acc = None
    if args.results_json and (exact_hits or mh_hits):
        contam_ids = {str(bid) for bid, _ in mh_hits}
        acc = recompute_accuracy(args.results_json, contam_ids)
        print(f"[3/3] Recomputed pass@1: {acc:.4f}")

    # Save JSON report
    report = {
        "exact_hits": exact_hits,
        "minhash_hits": mh_hits,
        "recomputed_accuracy": acc,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
