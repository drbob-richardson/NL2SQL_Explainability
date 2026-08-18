# Note for Server Claude — TMLR revision GPU runs (Paper 1: "What Predicts Correctness in Text-to-SQL?")

From: laptop Claude (on Rob's behalf). Date handed off: 2026-08-18.

## Mission (one deliverable)

Produce the **per-example test scores** for the three trained verifiers in **Table 4**, so we can bootstrap
**95% confidence intervals** on their in-distribution and LODO/transfer AUROCs. This is **TMLR reviewer #8**: the
paper currently reports Table-4 point estimates and a per-database "the frozen judge leads on *every* held-out
schema" claim (Fig. 1) with **no CIs** — the reviewer wants CIs, and we need to soften any per-DB lead whose CI
overlaps. Everything is prepped; these are **re-runs, not new science** (same models, same configs), with trainers
that now save per-example scores.

The canonical run-book is **`server_experiments/RUN_TABLE4_CIS.md`** — this note adds the *why*, the sanity
numbers, and the scope guard. If they ever disagree, RUN_TABLE4_CIS.md wins on mechanics.

## Scope guard — do ONLY this

- **Do NOT** run any Paper A / AISTATS work (the GraphRAG "structure-as-covariance" / BAGEL study). Rob explicitly
  **parked** its GPU items (e.g. the N=500 larger-pool BAGEL runs) — they are *not* for this server session.
- This session is **only** the three Table-4 CI re-runs below. Nothing here spends API budget (pure GPU; the
  bundled `server_experiments/data/verifier_data.jsonl`, 6,400 execution-labeled rows, is all the data needed).

## What changed in the trainers (the whole point)

`exp1_finetune_verifier.py` and `exp3_finetune_llm_judge.py` now write `indist_scores`/`labels` and
`lodo_per_db_scores`/`labels` into their results JSON. The *old* results on disk (Jun) lack these
(`indist_scores: False`), which is why we must re-run. Runtime is unchanged from the originals — only score-saving
was added.

## Steps (from `server_experiments/`)

```bash
pip install -r requirements.txt          # same env as before: torch, transformers>=4.48 (ModernBERT), peft (exp3)

# 0) smoke-test each first — ~1-3 min, confirms the box + score-saving before the real runs
python exp1_finetune_verifier.py --smoke
python exp3_finetune_llm_judge.py --smoke

# 1) encoder verifier  (Table 4: fine-tuned encoder, ModernBERT-base) — light GPU, quick
python exp1_finetune_verifier.py --mode both --model answerdotai/ModernBERT-base --epochs 3

# 2) generative judges (Table 4: Qwen2.5-1.5B and 7B LoRA) — default --epochs 2 is correct (matches the paper)
python exp3_finetune_llm_judge.py --mode both --model Qwen/Qwen2.5-1.5B-Instruct
python exp3_finetune_llm_judge.py --mode both --model Qwen/Qwen2.5-7B-Instruct
```

Notes:
- **Do not pass `--epochs` to exp3** — its default (2) is exactly what the paper's Table 4 used. exp1 takes
  `--epochs 3` as shown.
- `--mode both` = in-distribution + LODO (one training per held-out DB, 8 DBs). So exp3-7B is 9 LoRA fine-tunes of
  a 7B model — it's the heavy one; want a 24GB+ GPU (bf16) and budget a couple of hours. 1.5B and ModernBERT are
  quick. If 7B OOMs, drop `--bs` to 2 (grad-accum already 4) rather than changing epochs/LoRA rank.
- Each run writes `results/exp1_verifier_ModernBERT-base.json` and `results/exp3_judge_Qwen2.5-{1.5B,7B}-Instruct.json`.

## Sanity — confirm you reproduced the paper before trusting the CIs

Training is stochastic, so point estimates may drift a little; they should land **close** to these (in-dist / LODO
macro). If any is off by more than ~0.02, **flag it** — Rob updates the point estimate *and* the CI, honestly,
rather than pretending the old number stands.

| verifier | in-dist AUROC | LODO (macro) |
|---|---|---|
| ModernBERT-base (exp1) | ~0.785 | ~0.670 |
| Qwen2.5-1.5B LoRA (exp3) | ~0.766 | ~0.659 |
| Qwen2.5-7B LoRA (exp3) | ~0.798 | ~0.662 |

The story these must preserve: fine-tuned verifiers hit a **transfer wall (~0.66)** regardless of architecture or
scale (1.5B ≈ 7B ≈ ModernBERT on LODO), while the frozen GPT-4o judge sits at **0.71** on the same LODO — i.e.
*reasoning transfers, fitting does not*, and **scaling to 7B does not close the gap**. If 7B were to suddenly jump
to LODO ~0.72 that would be a real finding (and would *change* the paper's conclusion) — so double-check it's not a
data/label-leak bug before celebrating; the expected result is that 7B still plateaus.

## Hand back

Copy the three refreshed `server_experiments/results/*.json` (now containing the per-example
`indist_scores`/`lodo_per_db_scores`) back into the repo. Then on the laptop (no GPU, no API):

```bash
./.venv/bin/python scripts/paper1_table4_cis.py
```

It prints in-dist AUROC + 95% CI, LODO pooled + macro + CIs, and **per-database** AUROC + CIs for each model, plus
the frozen GPT-4o judge's per-DB CIs from cache so the LODO column and the Fig-1 per-schema comparison are directly
comparable. That output goes straight into Table 4 / the per-DB figure, softening any per-DB "lead" whose CI
overlaps the next verifier.

That's the entire job. Ping laptop Claude with the `paper1_table4_cis.py` output and we'll fold it into the tex.
