# Data index

Every file in `data/` mapped to the paper that uses it. **Files are NOT moved** — many scripts build
`data/` paths dynamically (f-strings / helper functions), so physically relocating them would silently
break pipelines. Use this map to know what belongs where. (Generated; see `PAPERS.md`.)

## Text2SQL UQ (TMLR)

- `bird_graph_conf.json` — 36K, 2 refs
- `bird_samples_gpt_4o.json` — 1.6M, 1 refs
- `bird_selfcorrect.json` — 68K, 1 refs
- `bird_signals.json` — 100K, 12 refs
- `bird_verify.json` — 36K, 4 refs
- `bird_verify_anthropic_claude_sonnet_4_6_verbal.json` — 24K, 3 refs
- `bird_verify_gemini_gemini_2_5_flash_verbal.json` — 20K, 1 refs
- `bird_verify_gen-gpt_4o.json` — 36K, 1 refs
- `bird_verify_gen-gpt_4o_anthropic_claude_sonnet_4_6_verbal.json` — 24K, 1 refs
- `bird_verify_gen-gpt_4o_gpt_4o.json` — 36K, 1 refs
- `bird_verify_gpt_4o.json` — 36K, 2 refs
- `bird_verify_verbal.json` — 24K, 2 refs

## Subgraph (JASA A&CS)

- `ambrosia` — 62M, 9 refs
- `ambrosia_interp_gen_ctl_gpt_4o.json` — 108K, 1 refs
- `ambrosia_interp_gen_gpt_4o.json` — 92K, 1 refs
- `ambrosia_samples.json` — 4.1M, 4 refs
- `bird_value_tokens.json` — 160K, 2 refs
- `bridge_emb.json` — 35M, 4 refs
- `downstream_sql.json` — 592K, 1 refs
- `embeddings.json` — 281M, 2 refs
- `schema_growth_example.json` — 4.0K, 2 refs
- `scifact` — 4.3M, 1 refs
- `scifact_emb.json` — 170M, 1 refs
- `spider_db` — 101M, 7 refs
- `spider_multi_labels.json` — 180K, 1 refs
- `spider_samples.json` — 516K, 5 refs
- `spider_samples_k16.json` — 864K, 1 refs
- `spider_samples_lp.json` — 648K, 1 refs
- `spider_subgraph_emb.json` — 28M, 1 refs
- `spider_value_tokens.json` — 64K, 1 refs
- `uq_bird.json` — 4.0K, 1 refs
- `uq_spider.json` — 4.0K, 1 refs

## GraphRAG-A (AISTATS)

- `graphrag_judge_hopaware_gpt-4o-mini.json` — 3.4M, 8 refs
- `graphrag_judge_labels.json` — 124K, 1 refs
- `graphrag_qa_answers.json` — 208K, 1 refs
- `hotpot` — 26M, 10 refs
- `hotpot_emb.json` — 482M, 10 refs
- `musique` — 29M, 2 refs
- `musique_decomp.json` — 168K, 1 refs
- `musique_decomp_emb.json` — 49M, 2 refs
- `musique_emb.json` — 705M, 2 refs
- `musique_qa_answers.json` — 112K, 1 refs
- `twowiki` — 29M, 4 refs
- `twowiki_emb.json` — 316M, 4 refs

## Shared across papers

- `airbnb.sqlite` — 8.0K, 4 refs
- `airbnb_eval.json` — 8.0K, 2 refs
- `airbnb_eval_large.json` — 16K, 1 refs
- `beaver` — 1.2M, 2 refs
- `beaver_emb.json` — 6.5M, 2 refs
- `beir` — 75M, 5 refs
- `bird` — 146M, 20 refs
- `bird_samples.json` — 1.6M, 40 refs
- `bird_samples_gpt_4_1_mini.json` — 1.6M, 1 refs
- `bird_verify_gen-gpt_4_1_mini.json` — 36K, 1 refs
- `bird_verify_qsql.json` — 36K, 1 refs
- `bird_verify_qsql_schema.json` — 36K, 1 refs
- `cqa` — 466M, 4 refs
- `hotpot_ce.json` — 816K, 1 refs
- `openai_samples.json` — 64K, 4 refs
- `retrieval_emb.json` — 13M, 1 refs
- `spider_samples_multi.json` — 804K, 5 refs

## No literal reference found — likely dynamic-loaded or unused (VERIFY before archiving)

- `ambrosia_elicit_gpt_4o.json` — 132K, no literal refs
- `ambrosia_elicit_gpt_4o_mini.json` — 552K, no literal refs
- `ambrosia_interp_judge_gpt_4o.json` — 24K, no literal refs
- `ambrosia_realize_gpt_4o.json` — 124K, no literal refs
- `bird_samples_nolp_backup.json` — 368K, no literal refs
- `bird_verify_gemini_gemini_2_0_flash_verbal.json` — 4.0K, no literal refs
- `decision_bird.json` — 24K, no literal refs
- `decision_spider.json` — 24K, no literal refs
- `downstream_gpt4o.log` — 4.0K, no literal refs
- `downstream_sql_gpt4o.json` — 612K, no literal refs
- `gen_gpt4o.log` — 4.0K, no literal refs
- `musique_hopassign_gpt-4o-mini.json` — 1.8M, no literal refs
- `musique_judge_gpt-4o-mini.json` — 1.8M, no literal refs
- `musique_run_output.txt` — 4.0K, no literal refs
- `scale_bird.json` — 4.0K, no literal refs
- `scale_spider.json` — 4.0K, no literal refs
- `spider_samples_gpt_3_5_turbo.json` — 56K, no literal refs
- `spider_samples_gpt_4o.json` — 508K, no literal refs
- `spider_verify_gpt_4o.json` — 28K, no literal refs
- `spider_verify_gpt_4o_mini.json` — 28K, no literal refs
