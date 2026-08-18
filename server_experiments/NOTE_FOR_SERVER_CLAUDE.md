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

---

# FOLLOW-UP — after your runs came back (2026-08-18)

Runs received, thank you — and good catch on the Fig. 1 "frozen judge leads on **every** held-out schema"
overstatement. You're right: it's 5/8, no per-DB difference is statistically separated, and it was already false
in June (not something your re-run introduced). We (laptop side) have handled all the **tex**; here's the state
and the one thing we still need from you.

## Already done AND PUSHED on our end — so `git pull` first, and do NOT re-edit the tex
`paper/1-text2sql-uq/paper1_correctness.tex` (note: the repo was reorganized into per-paper folders while you were
running — paper 1 now lives under `paper/1-text2sql-uq/`; `PAPERS.md` at the root is the map). We:
- Put your Table-4 numbers + 95% CIs into `\label{tab:transfer}` (now in-dist / macro / **pooled** columns), with
  an environment note (newer stack; ModernBERT in-dist 0.785→0.750; macro-LODO +0.01–0.02).
- Fixed the honesty issues: "every schema" (body §4.4 **and** the Fig. 1 caption) → "leads on average, 5/8, no
  per-DB difference statistically separated"; softened "above every fine-tuned alternative" → **directional**
  (macro CIs overlap + the per-question-vs-per-candidate units point you raised); **kept** the strong claim
  (fine-tuning doesn't transfer regardless of scale: 7B ≈ 1.5B ≈ encoder ≈ 0.68, with CIs). Also updated the
  abstract, the appendix transfer table (base 0.659→0.670, 7B 0.662→0.685), and the deployment range (0.77–0.79 →
  0.75–0.78). Compiles clean (11 pp).
- **=> The tex is finished and pushed. Please don't touch it — pull and you'll have the corrected version.**

## The ONE thing we need from you: push your result JSONs
Your re-run JSONs with per-example scores are still uncommitted on the box; our local copies are the June ones
(`indist_scores` absent), so we can't verify the CIs or compute the feature-classifier row. Please:
```
cd server_experiments
git add results/exp1_verifier_ModernBERT-base.json \
        results/exp3_judge_Qwen2.5-1.5B-Instruct.json \
        results/exp3_judge_Qwen2.5-7B-Instruct.json
git commit -m "Table-4 re-run results with per-example scores (for CIs + figure)"
git pull --rebase        # picks up our tex edits + this note (different files, no conflict)
git push
```
Once they land we'll (here) independently re-derive the CIs, fill the **feature-classifier CI** (via
`paper1_table4_cis.py`, which needs the trainer JSONs present), and finalize the numbers.

## Leave the figure to us — thanks for offering, but don't wire up `paper1_figures.py`
Once your JSONs are pushed we'll fix the `:122` hardcode to read from the JSON and regenerate
`paper1_lodo_perdb.png` so the picture matches the corrected caption. Keeping the tex + figure code + PNG in one
hand avoids us both editing the same script.

## Cross-check when you pull (flag any drift from your JSONs)
| verifier | in-dist | LODO macro | LODO pooled |
|---|---|---|---|
| ModernBERT | 0.750 [.724,.776] | 0.679 [.656,.706] | 0.716 [.704,.729] |
| Qwen 1.5B | 0.777 [.751,.800] | 0.670 [.638,.701] | 0.717 [.705,.730] |
| Qwen 7B | 0.783 [.757,.807] | 0.685 [.650,.722] | 0.699 [.686,.713] |
| frozen GPT-4o | — | 0.710 [.655,.764] | 0.770 [.737,.801] |

Your two deviations were both the right call — holding effective batch at 16 via `--grad-accum 8` (in-dist
reproduced at 0.783), and fast-forwarding the 111 commits. No concerns.
