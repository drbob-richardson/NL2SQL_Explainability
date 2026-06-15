# Server experiments — trained SQL verifier & verification-guided reranking

Self-contained package to run on a GPU server, then bring the `results/` folder back.
**No databases, embeddings, or API keys needed** — the execution-labeled data is bundled.

## TL;DR

```bash
# on the server, inside this folder:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                         # exp2 (free) + exp1 smoke test
# then the real run:
python exp1_finetune_verifier.py --mode both --model roberta-base --epochs 3
tar czf results.tgz results/            # bring this back
```

Every script prints to the screen AND writes `results/<name>.json` + `results/<name>.log`.
To bring results back, copy the whole `results/` folder (or `results.tgz`).

## The question these test

We established (off-server) that an **LLM verifier** is the best signal for whether generated SQL
is correct (AUROC 0.72 with gpt-4o-mini, 0.77 with gpt-4o, both zero-shot/transferable), while a
**cheap feature-trained classifier** matches it in-distribution (0.768) but **collapses on unseen
schemas** (0.661). Two open questions need a GPU:

1. **Does a *fine-tuned transformer* verifier transfer across schemas?** (exp1)
   Success = leave-one-DB-out AUROC **> 0.77** (beats the frozen gpt-4o verifier on transfer).
   Failure mode to watch = LODO << in-distribution (it overfit schemas, like the cheap classifier).
2. **Can verifier-guided reranking raise end accuracy?** (exp2, then verifier-as-reranker)
   exp2 shows the *headroom*; the verifier is the lever to capture it.

## Files

| file | what | needs GPU? |
|---|---|---|
| `data/verifier_data.jsonl` | 6,400 execution-labeled rows: question+evidence+schema+SQL → correct? | — |
| `data/bird_samples.json` | 800 questions × 8 samples (with logprobs, correctness) | — |
| `prepare_data.py` | rebuilds `verifier_data.jsonl` (only if you regenerate samples) | no |
| `exp1_finetune_verifier.py` | fine-tune verifier; in-distribution + leave-one-DB-out AUROC | **yes** |
| `exp2_rerank.py` | reranking accuracy & best-of-N headroom (no model needed) | no |
| `run_all.sh` | data check → exp2 → exp1 smoke test | partial |

## exp1 — fine-tuned verifier (the main GPU run)

```bash
python exp1_finetune_verifier.py --smoke            # ~1-2 min: confirms it runs end-to-end
python exp1_finetune_verifier.py --mode indist      # in-distribution AUROC only (1 training)
python exp1_finetune_verifier.py --mode lodo        # transfer: one training per held-out db (8x)
python exp1_finetune_verifier.py --mode both --model roberta-base --epochs 3
```
Useful flags: `--model` (try `roberta-base`, `microsoft/deberta-v3-base`, or long-context
`answerdotai/ModernBERT-base`), `--epochs`, `--bs`, `--max-len` (raise to 1024 with ModernBERT for
big schemas), `--max-dbs` (cap LODO trainings for speed).

**Reference baselines printed in the output** (BIRD, gpt-4o-mini generations):
frozen verifier mini 0.724 / gpt-4o 0.770 (transfer); self-consistency 0.616;
cheap classifier 0.768 in-dist / 0.661 transfer.

**Read the result:** report both numbers. If `indist` is high (~0.8) but `lodo` is low (~0.66),
the fine-tune overfits schemas and a *universal* verifier needs the LLM's reasoning (next step:
fine-tune a small generative LLM judge, or train on far more diverse schemas). If `lodo` > 0.77,
the bet is won: a fine-tuned verifier transfers and beats the frozen LLM judge.

## exp2 — verification-guided reranking headroom

```bash
python exp2_rerank.py
```
Seeded result (bundled data): modal/self-consistency 0.451, best-logprob 0.446,
**oracle best-of-8 = 0.506** → **+5.5 pts of headroom** a perfect reranker could capture, which
cheap signals (logprob) miss. Next step on the server: score every sample with the exp1 verifier
and pick the top-ranked one (verifier-guided selection), and **regenerate with larger K (32–64)**
to widen the oracle ceiling — more tokens → more headroom for the verifier to capture.

## Bigger directions (what GPUs unlock next, after exp1/exp2)

1. **Verifier-guided generation** — use the exp1 verifier to rerank best-of-K, then DPO the
   generator against verifier-preferred samples. Turns UQ into *accuracy* (the result top venues
   reward). exp2 measures the headroom this targets.
2. **Universal vs per-lake verifier** — exp1's LODO answers whether a verifier can be universal;
   if not, a per-deployment (in-distribution) verifier is still a near-free win (0.77 at ~0 cost).
3. **Scale the data** — regenerate samples across Spider + BIRD + Spider 2.0 with larger K to build
   a big execution-labeled correctness dataset (a contribution in its own right + training fuel).

## What to bring back

The `results/` folder (`*.json` for numbers, `*.log` for full output). Optionally:
```bash
tar czf results.tgz results/
```
Then we look at `exp1_verifier_*.json` (indist vs lodo AUROC) and `exp2_rerank.json` together.
