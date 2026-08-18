# Scripts index

All 170 scripts in `scripts/` mapped to their paper. Scripts are kept **flat** (they cross-import
and use non-uniform `sys.path` bootstraps, so foldering them risks breaking imports). ★ = a module other
scripts import (a shared library rather than a runnable entry point). (Generated; see `PAPERS.md`.)

> **Cross-paper shared modules** (imported by more than one paper — edit with care): `table_selection`, `spider_benchmark`, `bird_error_analysis`, `bayes_subgraph_exact`, `bayes_subgraph_v2`, `graphrag_active_scale`, `graphrag_downstream_qa`, `graphrag_judge_hopaware`, `graphrag_n100`

## Text2SQL UQ (TMLR)

- `ambiguity_probe.py`
- `bird_abstention.py`
- `bird_column_posterior.py` ★
- `bird_correctness_final.py`
- `bird_correctness_uq.py`
- `bird_discovery.py`
- `bird_error_analysis.py` ★
- `bird_exec_uq.py`
- `bird_generate.py`
- `bird_graph_uq.py`
- `bird_join_posterior.py` ★
- `bird_judge_rationales.py`
- `bird_lib.py` ★
- `bird_openworld.py`
- `bird_perdeploy_abstention.py`
- `bird_prune_feasibility.py`
- `bird_selfcorrect.py`
- `bird_table_posterior.py`
- `bird_verifier_by_feature.py`
- `bird_verify.py`
- `bnp_equivclass.py` ★
- `bnp_novelty_complexity.py`
- `bnp_novelty_value.py`
- `bnp_probes.py`
- `breadth_compare.py`
- `cross_provider_analysis.py`
- `paper1_entropy_ksweep.py`
- `paper1_figures.py`
- `paper1_gen4o_verify.py`
- `paper1_table1_cis.py`
- `paper1_table4_cis.py`
- `plot_decision_frontier.py`
- `verifier_input_ablation.py`
- `verifier_probe.py`

## Subgraph (JASA A&CS)

- `ablate_scores.py`
- `active_pilot.py`
- `ambrosia_coverage.py`
- `ambrosia_decision_sim.py`
- `ambrosia_elicit.py`
- `ambrosia_generate.py`
- `ambrosia_interp.py`
- `ambrosia_probe.py`
- `ambrosia_realize.py`
- `ambrosia_rescore.py`
- `ambrosia_uq_coverage.py`
- `bayes_schema_growth.py`
- `bayes_subgraph.py`
- `bayes_subgraph_calib_metrics.py`
- `bayes_subgraph_corrected.py`
- `bayes_subgraph_decision.py`
- `bayes_subgraph_exact.py` ★
- `bayes_subgraph_hbayes.py` ★
- `bayes_subgraph_heldout.py`
- `bayes_subgraph_hmc_validate.py`
- `bayes_subgraph_purebayes.py`
- `bayes_subgraph_review2.py`
- `bayes_subgraph_scale.py`
- `bayes_subgraph_spaghetti.py`
- `bayes_subgraph_tier2.py`
- `bayes_subgraph_uq.py`
- `bayes_subgraph_v2.py` ★
- `beir_encode.py`
- `beir_hier.py`
- `beir_uq.py`
- `compare_baselines.py` ★
- `compare_single_multi.py`
- `connector_analysis.py`
- `downstream_ex.py`
- `extend_k16.py`
- `laplace_calib_gate.py`
- `logprob_experiment.py`
- `model_sweep.py`
- `multitable_diagnostic.py`
- `phase1_adaptive.py`
- `phase1_cosinecoupling.py`
- `phase1_fkbaseline.py`
- `phase1_validate.py`
- `phase3_selective.py`
- `plot_concept_figure.py`
- `plot_coupling_posterior.py`
- `plot_schema_growth.py`
- `plot_subgraph_figs.py`
- `retrieval_probe.py`
- `review_fixes.py`
- `s1a_graphgp.py`
- `s2_beaver.py`
- `s3_scifact.py`
- `s3_sql_hopgate.py`
- `s3_sql_singlehop.py`
- `s_adapt_beaver.py`
- `schema_linking_uq.py`
- `sim_calibration_study.py`
- `spider_benchmark.py` ★
- `spider_correctness.py`
- `table_selection.py` ★
- `verify_global_map.py`

## GraphRAG-A (AISTATS)

- `active_pilot2.py`
- `graphrag_active_analysis.py`
- `graphrag_active_scale.py` ★
- `graphrag_ccvoi.py`
- `graphrag_chain_completion.py` ★
- `graphrag_downstream_qa.py` ★
- `graphrag_evoi.py` ★
- `graphrag_judge_comparison.py`
- `graphrag_judge_fix.py` ★
- `graphrag_judge_hopaware.py` ★
- `graphrag_lambda_ceiling.py` ★
- `graphrag_lambda_learn.py` ★
- `graphrag_lambda_mixed.py` ★
- `graphrag_llm_judge.py` ★
- `graphrag_lookahead.py`
- `graphrag_n100.py` ★
- `graphrag_n100_judge.py`
- `graphrag_n100_normalized.py`
- `graphrag_n100_qa.py`
- `musique_decomp_graph.py` ★
- `musique_diagnose.py`
- `musique_embed.py`
- `musique_entity_graph.py` ★
- `musique_hopassign_graph.py`
- `musique_implgraph.py`
- `musique_n100.py` ★
- `musique_run.py` ★
- `paperA_alignment_sim.py` ★
- `paperA_assortativity.py`
- `paperA_bagel.py`
- `paperA_fig_alignment.py`
- `paperA_fig_bagel.py`
- `paperA_metrics.py` ★
- `paperA_mixture.py`
- `paperA_negative_analysis.py`
- `paperA_perdataset.py`
- `paperA_routing_normalized.py`
- `s4_adaptive.py`
- `s4_diversity_uq.py`
- `s4_hotpot.py`
- `s5_twowiki.py`
- `s_graph_posterior.py`
- `s_reranker.py`
- `s_sensitivity.py`

## De-aliasing-B (JASA T&M)

- `paperB_boundary_sim.py`
- `paperB_branch_sim.py`
- `paperB_correction_sim.py`
- `paperB_dealiasing_sim.py`
- `paperB_field_sim.py`
- `paperB_identify_sim.py`
- `paperB_lan_sim.py`
- `paperB_lowerbound_sim.py`
- `paperB_rate_sim.py`
- `paperB_realdata_correction.py`
- `paperB_realdata_correction2.py`
- `paperB_subtree_sim.py`
- `paperB_unknown_sim.py`

## Cross-cutting / uncertain (shared utilities or ambiguous)

- `ablate_mechanism.py`
- `bridge_probe.py`
- `build_diverse_verifier_data.py`
- `calibration_check.py`
- `demo.py`
- `demo_calibration.py`
- `demo_model_a.py`
- `discovery_detection.py`
- `enumerate_space.py`
- `gen_eval.py`
- `make_db.py`
- `polish_stats.py`
- `review2_experiments.py`
- `run_benchmark.py`
- `sample_openai.py`
- `sim_bridge_recovery.py`
- `where_bayes_matters.py`
