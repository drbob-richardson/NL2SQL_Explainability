# Server job: deeper-pool (N=500) faithful-BAGEL run — Paper A (AISTATS)

**Goal.** Re-run the faithful-BAGEL head-to-head at a top-**500** candidate pool (currently top-100), to
test whether the budget-invariant covariance gap (graph-GP vs BAGEL+prior, ~+0.11 at N=100) survives a
5x deeper pool. This retires the last "small-pool artifact" fairness objection and probes the deep-burial
regime where the graph covariance should matter most. Not GPU — it's an OpenAI judge (gpt-4o-mini) job.

## Prerequisites (confirm before running)
- Repo synced, including **new** `scripts/paperA_bagel_judge.py` and the edited `scripts/paperA_bagel.py`
  (now has `--out`).
- Data present:
  - `data/hotpot_emb.json` (482M), `data/twowiki_emb.json` (316M)
  - `data/hotpot/dev_distractor.parquet`, `data/twowiki/dev.parquet`
  - `data/graphrag_judge_hopaware_gpt-4o-mini.json` — the existing **90,663-label** judge cache. MUST be
    present; the new labels append to it (atomic write, so a crash won't corrupt it).
- `OPENAI_API_KEY` set; env has `openai`, `numpy`, `pyarrow`, `tiktoken`.
- Keep `--n 4000 --subset 300 --pool 500` **identical** across both scripts so the pools align. `--subset 300`
  is the established n=600 configuration (matches the N=100 curve — apples-to-apples).

## Steps

**1. Dry-run (free) — confirm the count and cost before spending.**
```
python scripts/paperA_bagel_judge.py --pool 500 --subset 300
```
Expect **242,549** uncached judge calls (~68.8M input tok), **~$10.61** (measured on the laptop dry-run
2026-08-20). It reprints the exact figure on the server; sanity-check it matches before spending.

**2. Judge the deeper pool (~$10-11, ~1-2 h).**
```
python scripts/paperA_bagel_judge.py --pool 500 --subset 300 --run --workers 16
```
- Bump `--workers` to 24-32 if you see no 429 rate-limit errors (the SDK already retries transient ones).
- **Resumable:** it only judges uncached passages, so if it dies just rerun the same command — it continues
  and retries anything skipped. Progress prints every 2000 calls.

**3. Run the comparison at N=500 ($0, CPU) — writes a SEPARATE results file (N=100 is preserved).**
```
python scripts/paperA_bagel.py --pool 500 --subset 300 \
    --budgets 1,2,3,5,10,20,40,50 \
    --out paper/4-graphrag-A/bagel_results_n500.json \
    | tee paper/4-graphrag-A/bagel_n500.log
```
This prints the recall/nDCG/completion table per budget with the two key CIs:
`graph-BAGEL` and `graph-(BAGEL+prior)` (the latter isolates the covariance, same calibrated mean).

## Sync back / report
- `paper/4-graphrag-A/bagel_n500.log` (the table) and `paper/4-graphrag-A/bagel_results_n500.json`.
- The updated `data/graphrag_judge_hopaware_gpt-4o-mini.json` (grows to ~15M) if we'll regenerate figures locally.

## What we're looking for
The headline is the **covariance gap**, `graph-(BAGEL+prior)`. At N=100 it was ~+0.11 and flat from B=1 to
native B=50. If at N=500 it stays ~+0.10-0.12 → fairness bulletproofed (the win is pool-invariant too, in the
realistic regime where the budget is 1-10% of the pool). If it shrinks materially → an honest pool-size
dependence; report the full curve either way. Also watch the total `graph-BAGEL` gap and completion.

(Do not pass a different `--subset`/`--n`/`--pool` to the two scripts — the judge cache must cover exactly the
pools the comparison builds, or unjudged passages silently default to relevance 0.)
