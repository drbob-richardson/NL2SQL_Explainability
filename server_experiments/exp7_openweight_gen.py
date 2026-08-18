"""TMLR revision #4 (broaden generators): generate K=8 SQL samples per BIRD question with an OPEN-WEIGHT,
non-OpenAI generator, so the correctness-prediction findings are checked outside the OpenAI family.

GPU-only, no API and no BIRD databases needed here: it reads the pre-exported prompts (data/bird_prompts.json,
byte-identical to what the OpenAI generators saw) and writes raw generations. Execution vs gold, signals, and
LLM-judging happen back on the laptop (which has the databases + API keys).

  # smoke (first 20 questions, ~1 min once the model is cached):
  python server_experiments/exp7_openweight_gen.py --smoke
  # full run (default Qwen2.5-Coder-7B-Instruct; ~6400 generations, a few minutes on one GPU):
  python server_experiments/exp7_openweight_gen.py --model Qwen/Qwen2.5-Coder-7B-Instruct
  # stronger generator option:
  python server_experiments/exp7_openweight_gen.py --model Qwen/Qwen2.5-Coder-32B-Instruct --tp 2

Needs: pip install vllm   (transformers/torch already in requirements.txt)
"""
from __future__ import annotations
import argparse, json, os, re, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def extract_sql(text):
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)   # strip a ```sql ... ``` fence if present
    if m:
        t = m.group(1)
    t = t.strip().strip("`").strip()
    if t[:3].lower() == "sql":
        t = t[3:].strip()
    return " ".join(t.split())                                 # collapse whitespace/newlines to one line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (use 2+ for 32B)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    prompts = json.load(open(os.path.join(ROOT, "data", "bird_prompts.json")))
    keys = list(prompts)
    if args.smoke:
        keys = keys[:20]
    tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_") + ("_smoke" if args.smoke else "")
    outpath = os.path.join(os.path.dirname(__file__), "results", f"bird_samples_{tag}_raw.json")
    print(f"model={args.model}  prompts={len(keys)}  K={args.k}  -> {os.path.basename(outpath)}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": prompts[k]["system"]},
         {"role": "user", "content": prompts[k]["user"]}],
        tokenize=False, add_generation_prompt=True) for k in keys]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.90)
    sp = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_tokens, logprobs=1)
    outs = llm.generate(texts, sp)

    result = {}
    for k, o in zip(keys, outs):
        samples, logp = [], []
        for co in o.outputs:
            samples.append(extract_sql(co.text))
            nt = max(len(co.token_ids), 1)
            logp.append(float(co.cumulative_logprob) / nt)     # mean per-token log-probability
        result[k] = {"db_id": prompts[k]["db_id"], "question_id": prompts[k]["question_id"],
                     "samples": samples, "logp": logp}
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    json.dump(result, open(outpath, "w"))
    print(f"wrote {len(result)} question generations ({args.k} samples each) -> {outpath}")
    print("Next: commit+push this file; on the laptop run scripts/bird_openweight_finish.py to execute vs gold,\n"
          "then bird_verify.py --samples <merged> for the judges, then paper1_openweight_verify.py.")


if __name__ == "__main__":
    main()
