# TBI-GF: Disease-Adapted Single-Cell Foundation Modeling for Traumatic Brain Injury

Official analysis, multi-task fine-tuning, out-of-distribution (CS-OOD) benchmarking, white-box self-attention gene regulatory network (GRN) extraction, *in silico* perturbation (ISP) desaturation, single-nucleus dropout calibration, and figure reproduction repository for:

> **Disease-Adapted Single-Cell Foundation Modeling Resolves Cross-Species Glial Pathology, Chronic Temporal Persistence, and Effector Hub Perturbations in Traumatic Brain Injury**
> Jiangtao Sheng\*, Xueer Liu, Runmin Lin, Xiaoxuan Chen, Kangsheng Li\*, Weiqiang Chen\*
> Shantou University Medical College & First Affiliated Hospital of Shantou University Medical College

---

## Overview

**TBI-GF** adapts the 12-layer human-pretrained `Geneformer-V2-104M` Transformer backbone (`d_model = 768`, 12 attention heads, 4,096-token rank-value context window) to **536,619 singlet mouse traumatic brain injury (TBI) nuclei** across four cohorts (`GSE269748`, `GSE277487`, `GSE271769`, `GSE290150`) via 1:1 mouse-to-human Ensembl orthology (`16,320` matched tokens), and evaluates generalization and mechanism across **759,749 total mouse and human nuclei**:

1. **Multi-Task TBI-GF (`run2`)**: Freezes lower syntactic layers (`0–1`) and fine-tunes upper semantic layers (`2–11`) with dual classification heads (14-lineage identity: `97.67%` held-out test accuracy; binary injury state: `98.58%` accuracy on `n = 53,674` test nuclei).
2. **Exam 1 — Temporal Extrapolation & Classical Baseline Benchmark**: `TBI-GF-le7d` trained strictly on `<= 7d` nuclei (`n = 365,295`) extrapolates zero-shot to unseen Day 28 (`GSE271769`, AUROC `0.9654`; microglia `0.9869`, `+0.3509` AUROC over 5 classical HVG/PCA/Harmony/GBDT/module baselines) and chronic 6-month (`GSE269748`, embedding AUROC `0.8966`; major glia `0.9148`) brains with `1.0000` sample-level AUROC.
3. **Exam 2 — Zero-Shot Mouse-to-Human Clinical Transfer**: Evaluated zero-shot on `24,241` human neurosurgical TBI frontal-cortex nuclei across 17 donors (`GSE209552`), achieving `93.93%` canonical cell-type accuracy, `1.0000` donor AUROC (`p = 3.23e-4`), and `0.9032` microglial embedding AUROC (`+0.3559` over classical baselines).
4. **Exam 3 — Zero-Shot Cross-Disease AD Projection & Genetic *Trem2*-KO Rescue**: Generalizes zero-shot to human post-mortem AD entorhinal cortex (`GSE138852`, astrocyte AUROC `0.9220`) and mouse 5xFAD vs `Trem2-KO 5xFAD` brains (`GSE140511`), where genetic *Trem2* deletion rescues predicted microglial injury by `124.3%–126.1%` (`p = 3.71e-5`).
5. **Two-Phase Softmax-Jacobian Desaturated *In Silico* Perturbation & Unseen *Tgfbr1*-cKO Validation (`GSE333879`)**: Resolves softmax Jacobian attenuation (`P(1-P) -> 0`) to restore `31.7x` (*APOE*) and `51.1x` (*IL1B*) sensitivity, and validates *SPP1* (`z = +48.97`) and *APOE* (`z = +35.91`) zero-shot in an unseen Day 30 astrocytic *Tgfbr1*-cKO cohort (`GSE333879`, `35,047` nuclei; `5.6x` microglial and `8.1x` astrocytic paracrine exacerbation).
6. **White-Box Self-Attention GRN & 24,515-Gene Nuclear Sparsity Calibration**: Extracts Layer 8–11 multi-head self-attention weights (`1.41x–3.19x` hub amplification onto `LPL–SPP1–IL1B–TREM2–APOE–HEXB`) and calibrates the `<0.85%` droplet snRNA-seq capture floor (`Spearman rho = 0.996`) of low-copy cell-cycle inhibitors (*Cdkn2a*, *Cdkn1a*).

---

## Repository Structure

- `scripts/01_training_and_tokenization/`
  - `train_tbi_gf_full_v2.py` — Multi-task bfloat16 SDPA fine-tuning of `TBI-GF (run2)` across 536,619 mouse TBI singlet nuclei
  - `eval_test.py` — Held-out test evaluation (`n = 53,674` nuclei) and lineage-stratified calibration
  - `train_exam1_temporal_le7d.py` — Exam 1 temporally restricted `<= 7d` training (`TBI-GF-le7d`) and zero-shot Day 28 / 6-month evaluation
  - `tokenize_routeA_exam2_isp2.py` — 1:1 Ensembl ortholog rank-value tokenization for human clinical TBI (`GSE209552`) and unseen Day 30 `GSE333879`
- `scripts/02_ood_exams_and_isp/`
  - `kernel_routeA_exam2_isp2.py` — Exam 2 zero-shot human TBI migration and ISP Phase 2 execution on `GSE333879`
  - `isp_p16_arm.py` — ISP Phase 1 single-gene deletion screen (`69,600` random-token null forward passes)
  - `isp_phase2_nonsat_recompute.py` — Softmax-Jacobian desaturation (`0.5 < P_orig < 0.99`) and combinatorial hub perturbation recomputation
  - `verify_routeA_consensus.py` — Five-trajectory consensus (`T1–T5`) and biological-replicate / clinical-donor aggregation
  - `run_tier1_baselines.py` — Head-to-head benchmark against 5 classical single-cell baselines (`HVG2000+LogReg`, `HVG+GBDT`, `PCA50+MLP`, `Harmony_ZPCA50`, `DAM_SenMayo_Sig`)
  - `run_tier1_attention_grn.py` — White-box Layer 8–11 and Layer 11 multi-head self-attention gene regulatory network extraction
  - `run_tier1_exam3_ad.py` — Exam 3 zero-shot projection to human AD (`GSE138852`), mouse `Trem2-KO 5xFAD` (`GSE140511`), and AD microglial ISP
- `scripts/03_dropout_calibration_R/`
  - `sd01_probe_release.R` to `sd06c_diag_actb.R` — Genome-wide 24,515-gene single-nucleus abundance-dropout calibration, abundance-matched controls, and 107-gene SenMayo truncation / leave-one-out sensitivity battery (`R >= 4.3`, `Seurat v5`)
- `scripts/04_figures/`
  - `make_paper2_fm_figures.py` — Reproducible generation of Main Figures 1–6
- `summary_tables/`
  - `Supplementary_Tables_S1_to_S7.xlsx` and constituent machine-readable CSV summary tables (`Table S1A`–`Table S7G`)

---

## System Requirements & License

- **Operating system**: Platform-independent (Linux / Windows / macOS; GPU fine-tuning tested on Linux Ubuntu 22.04 with NVIDIA RTX 4090 24 GB VRAM, CUDA 12.1)
- **Programming languages**: Python (`>= 3.10`) and R (`>= 4.3`)
- **Dependencies**: Listed in `requirements.txt` plus `Geneformer` (`GF-V2-104M`) and R packages `Seurat (>= 5.0)`, `Matrix`, `scDblFinder`
- **Data availability**: Public cohorts in NCBI GEO (`GSE269748`, `GSE277487`, `GSE271769`, `GSE290150`, `GSE209552`, `GSE138852`, `GSE140511`); in-house Day 30 mouse CCI & astrocytic *Tgfbr1*-cKO cohort in NCBI GEO under accession **`GSE333879`** (`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE333879`)
- **License**: OSI-approved MIT License (no restrictions to use by academic or non-academic researchers)
