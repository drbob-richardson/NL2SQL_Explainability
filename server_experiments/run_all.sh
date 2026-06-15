#!/usr/bin/env bash
# One-shot entry point. Run from inside server_experiments/.
# Outputs land in results/ as both .json and .log. Bring back the whole results/ folder.
set -e
cd "$(dirname "$0")"
mkdir -p results

echo "=================================================================="
echo " STEP 0: data check"
echo "=================================================================="
if [ ! -f data/verifier_data.jsonl ]; then
  echo "verifier_data.jsonl missing -> rebuilding (needs ../data/bird_samples.json + ../data/bird/db)"
  python prepare_data.py
else
  echo "data/verifier_data.jsonl present ($(wc -l < data/verifier_data.jsonl) rows) -- using bundled data"
fi

echo
echo "=================================================================="
echo " STEP 1: exp2 reranking (no GPU, ~seconds) -- accuracy headroom"
echo "=================================================================="
python exp2_rerank.py

echo
echo "=================================================================="
echo " STEP 2: exp1 SMOKE TEST (verifies torch/transformers/GPU work, ~1-2 min)"
echo "=================================================================="
python exp1_finetune_verifier.py --smoke

echo
echo "=================================================================="
echo " DONE with quick checks. For the real run, pick a model and launch:"
echo "   python exp1_finetune_verifier.py --mode both --model roberta-base --epochs 3"
echo "   # long-context (better for big schemas): --model answerdotai/ModernBERT-base"
echo " Then bring back the results folder:  tar czf results.tgz results/"
echo "=================================================================="
