# Re-run for Table 4 confidence intervals (TMLR revision #8)

The trainers `exp1_finetune_verifier.py` (encoder) and `exp3_finetune_llm_judge.py` (generative
Qwen judges) now **save per-example test scores** into their results JSONs, so we can bootstrap CIs
for the in-distribution and LODO/transfer AUROCs. Only these two need the GPU; the feature classifier
and the frozen GPT-4o judge get their CIs from cache locally (no GPU).

## On the GPU box

Same environment as before (`pip install -r server_experiments/requirements.txt`; needs only the
bundled `server_experiments/data/verifier_data.jsonl`). From `server_experiments/`:

```bash
# encoder verifier (Table 4 row: Fine-tuned encoder, ModernBERT-base)
python exp1_finetune_verifier.py --mode both --model answerdotai/ModernBERT-base --epochs 3

# generative judges (Table 4 rows: Qwen2.5-1.5B and 7B)
python exp3_finetune_llm_judge.py --mode both --model Qwen/Qwen2.5-1.5B-Instruct
python exp3_finetune_llm_judge.py --mode both --model Qwen/Qwen2.5-7B-Instruct
```

Each writes `results/exp1_verifier_<tag>.json` / `results/exp3_judge_<tag>.json` — now including
`indist_scores/labels` and `lodo_per_db_scores/labels`. (Runtime is the same as the original runs;
we only added score-saving. Use `--smoke` first to confirm it runs.)

## Back on your laptop

Copy the new `server_experiments/results/*.json` back into the repo, then:

```bash
./.venv/bin/python scripts/paper1_table4_cis.py
```

It prints, for each model: in-dist AUROC + 95% CI, LODO pooled + macro AUROC + CIs, and per-database
AUROC + CIs (for the "leads on every held-out schema" claim in Fig. 1). It also prints the frozen
GPT-4o judge's per-db CIs from cache so the LODO column is directly comparable. Paste those into
Table 4 / the per-db figure, and soften any per-db lead whose CI overlaps.

Nothing here spends API budget.
