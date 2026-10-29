#!/usr/bin/env python3
"""
Route A Verification & 5-Trajectory Consensus Script (Exam 2 GSE209552 + ISP Phase 2 GSE333879)
================================================================================================
Implements `agentic-data-analysis-loop` (5 independent analytical trajectories + >=60% consensus):
  1. Normalizes output file locations inside `results/s1_gf_finetune/routeA_out/`
  2. Exam 2 (Human GSE209552, 17 samples, 24,241 cells):
     - T1: Cell-level run2 P_injury (all QC cells)
     - T2: Cell-level run2 P_injury (high-confidence reference transfer: prediction_score_max >= 0.80)
     - T3: Cell-level fine-tuned 768-d embedding cosine shift from Control centroid
     - T4: Donor/Sample-level (n=17: 5 Control vs 12 TBI) mean P_injury
     - T5: Donor/Sample-level (n=17: 5 Control vs 12 TBI) mean fine-tuned embedding shift
     -> Produces `exam2_gse209552_5traj_consensus.csv`
  3. ISP Phase 2 (Unseen Mouse D30 GSE333879, MG_injury_D30):
     - T1: Full set (all 467 MG_injury_D30 cells) dP_injury
     - T2: Active injury subset (P_orig > 0.01) dP_injury
     - T3: Elevated injury subset (P_orig > 0.05) dP_injury
     - T4: Non-saturated transition subset (0.5 < P_orig < 0.99) dP_injury
     - T5: Full set 768-d embedding cosine shift toward Sham centroid (dCos_to_sham)
     -> Produces `isp2_gse333879_5traj_consensus.csv` and `routeA_complete_summary.json`
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = r"D:\2026-04-24\new-chat-2\TBI_astrocyte_scst_project\senescence_project\results\s1_gf_finetune"
OUT = os.path.join(BASE, "routeA_out")
NESTED = os.path.join(OUT, "routeA_out")

# 1. Flatten nested routeA_out/routeA_out files into routeA_out/ and remove leftover gfsrc
if os.path.isdir(NESTED):
    for fn in os.listdir(NESTED):
        src = os.path.join(NESTED, fn)
        dst = os.path.join(OUT, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    shutil.rmtree(NESTED, ignore_errors=True)

gfsrc_leftover = os.path.join(OUT, "gfsrc")
if os.path.isdir(gfsrc_leftover):
    shutil.rmtree(gfsrc_leftover, ignore_errors=True)

# Verify all 8 expected deliverables exist in OUT
expected_files = [
    "exam2_gse209552_cell_preds.csv.gz",
    "exam2_gse209552_celltype_summary.csv",
    "exam2_gse209552_metrics.json",
    "exam2_gse209552_sample_summary.csv",
    "isp2_gse333879_deletion_long.csv.gz",
    "isp2_gse333879_deletion_summary.csv",
    "isp2_gse333879_ko_rescue_summary.csv",
    "isp2_gse333879_zeroshot_preds.csv.gz",
]
for fn in expected_files:
    p = os.path.join(OUT, fn)
    assert os.path.exists(p) and os.path.getsize(p) > 0, f"Missing or empty deliverable: {p}"

# ==============================================================================
# PART 1: Exam 2 (Human GSE209552) 5-Trajectory Consensus
# ==============================================================================
df209 = pd.read_csv(os.path.join(OUT, "exam2_gse209552_cell_preds.csv.gz"))
samp209 = pd.read_csv(os.path.join(OUT, "exam2_gse209552_sample_summary.csv"))

canon_cts = ["ALL", "Microglia", "Astrocyte", "Oligodendrocyte", "OPC", "Neuron", "Endothelial"]
exam2_traj_rows = []

for ct in canon_cts:
    sub_all = df209 if ct == "ALL" else df209[df209["cell_type"] == ct]
    sub_hc = sub_all[sub_all["prediction_score_max"] >= 0.80]

    # T1: Cell-level P_injury (all QC cells)
    y1 = sub_all["true_injury_bin"].values
    s1 = sub_all["P_injury"].values
    auc1 = float(roc_auc_score(y1, s1))
    _, p1 = sp_stats.mannwhitneyu(s1[y1 == 1], s1[y1 == 0], alternative="two-sided")

    # T2: Cell-level P_injury (high-confidence reference transfer >= 0.80)
    y2 = sub_hc["true_injury_bin"].values
    s2 = sub_hc["P_injury"].values
    auc2 = float(roc_auc_score(y2, s2)) if len(np.unique(y2)) == 2 else np.nan
    _, p2 = sp_stats.mannwhitneyu(s2[y2 == 1], s2[y2 == 0], alternative="two-sided") if len(np.unique(y2)) == 2 else (np.nan, np.nan)

    # T3: Cell-level fine-tuned embedding cosine shift from Control centroid
    s3 = sub_all["ft_injury_cos_score"].values
    auc3 = float(roc_auc_score(y1, s3))
    _, p3 = sp_stats.mannwhitneyu(s3[y1 == 1], s3[y1 == 0], alternative="two-sided")

    # T4 & T5: Donor/Sample-level (n=17 samples: 5 Control vs 12 TBI)
    grp_samp = sub_all.groupby(["sample", "true_injury_bin"], as_index=False).agg(
        mean_P=("P_injury", "mean"),
        mean_ft_cos=("ft_injury_cos_score", "mean"),
        mean_fr_cos=("frozen_injury_score", "mean"),
        n_cells=("P_injury", "size"),
    )
    ys = grp_samp["true_injury_bin"].values
    if len(np.unique(ys)) == 2:
        auc4 = float(roc_auc_score(ys, grp_samp["mean_P"].values))
        _, p4 = sp_stats.mannwhitneyu(grp_samp.loc[ys == 1, "mean_P"], grp_samp.loc[ys == 0, "mean_P"], alternative="two-sided")
        auc5 = float(roc_auc_score(ys, grp_samp["mean_ft_cos"].values))
        _, p5 = sp_stats.mannwhitneyu(grp_samp.loc[ys == 1, "mean_ft_cos"], grp_samp.loc[ys == 0, "mean_ft_cos"], alternative="two-sided")
        auc_fr_d = float(roc_auc_score(ys, grp_samp["mean_fr_cos"].values))
    else:
        auc4, p4, auc5, p5, auc_fr_d = np.nan, np.nan, np.nan, np.nan, np.nan

    aucs = [auc1, auc2, auc3, auc4, auc5]
    pvals = [p1, p2, p3, p4, p5]
    n_pass = sum(1 for a, p in zip(aucs, pvals) if (not np.isnan(a)) and a > 0.70 and p < 0.05)

    exam2_traj_rows.append({
        "cell_type": ct,
        "n_cells_all": len(sub_all),
        "n_cells_hc80": len(sub_hc),
        "n_donors": len(grp_samp),
        "T1_CellProb_AUROC": round(auc1, 4),
        "T1_p": float(p1),
        "T2_HighConfProb_AUROC": round(auc2, 4),
        "T2_p": float(p2),
        "T3_CellEmbShift_AUROC": round(auc3, 4),
        "T3_p": float(p3),
        "T4_DonorProb_AUROC": round(auc4, 4),
        "T4_p": round(float(p4), 5),
        "T5_DonorEmbShift_AUROC": round(auc5, 4),
        "T5_p": round(float(p5), 5),
        "Frozen_CellEmbShift_AUROC": round(float(roc_auc_score(y1, sub_all["frozen_injury_score"].values)), 4),
        "Frozen_DonorEmbShift_AUROC": round(auc_fr_d, 4),
        "mean_AUROC_5traj": round(float(np.nanmean(aucs)), 4),
        "sd_AUROC_5traj": round(float(np.nanstd(aucs, ddof=1)), 4),
        "n_trajectories_pass": n_pass,
        "consensus_frac": round(n_pass / 5.0, 2),
        "ct_acc_canon6": round(float((sub_all["pred_canon6_ct"] == sub_all["cell_type"]).mean()), 4),
        "ct_acc_canon6_hc80": round(float((sub_hc["pred_canon6_ct"] == sub_hc["cell_type"]).mean()), 4),
    })

exam2_cons_df = pd.DataFrame(exam2_traj_rows)
exam2_cons_df.to_csv(os.path.join(OUT, "exam2_gse209552_5traj_consensus.csv"), index=False)
print("=== EXAM 2 (GSE209552) 5-TRAJECTORY CONSENSUS ===")
print(exam2_cons_df.to_string(index=False))

# ==============================================================================
# PART 2: ISP Phase 2 (Unseen Mouse D30 GSE333879) 5-Trajectory Consensus
# ==============================================================================
long333 = pd.read_csv(os.path.join(OUT, "isp2_gse333879_deletion_long.csv.gz"))
mg_long = long333[long333["arm"] == "MG_injury_D30"].copy()
orig_mg = mg_long[mg_long["variant"] == "__orig__"].set_index("cell_i")["P_orig"].to_dict()

def eval_isp_trajectory(df_arm, traj_name, cell_filter_fn, metric_col="dP"):
    keep_cells = {ci for ci, p0 in orig_mg.items() if cell_filter_fn(p0)}
    sub = df_arm[df_arm["cell_i"].isin(keep_cells)]
    n_cells = len(keep_cells)
    null1 = sub[sub["variant"] == "__null__"]
    null5 = sub[sub["variant"] == "__null5__"]

    rng_mc = np.random.RandomState(42)
    rows = []
    variants = [v for v in sub["variant"].unique() if v not in ("__orig__", "__null__", "__null5__")]
    for v in variants:
        x = sub[sub["variant"] == v]
        if len(x) == 0:
            continue
        ref_null = null5 if v.startswith("COMBO_") else null1
        nm = float(ref_null[metric_col].mean()) if len(ref_null) > 0 else 0.0
        val_m = float(x[metric_col].mean())

        # Cell-matched Monte Carlo permutation test (B=20,000)
        cell_pools = [grp[metric_col].values for _, grp in ref_null[ref_null["cell_i"].isin(x["cell_i"])].groupby("cell_i") if len(grp) > 0]
        if len(cell_pools) > 0:
            B_mc = 20000
            sim_means = np.zeros(B_mc, dtype=np.float64)
            for pool in cell_pools:
                sim_means += pool[rng_mc.randint(0, len(pool), size=B_mc)]
            sim_means /= len(cell_pools)
            if val_m >= nm:
                p_mc = float((1 + np.sum(sim_means >= val_m)) / (1 + B_mc))
            else:
                p_mc = float((1 + np.sum(sim_means <= val_m)) / (1 + B_mc))
        else:
            p_mc = 1.0

        rows.append({
            "trajectory": traj_name,
            "n_subset_cells": n_cells,
            "variant": v,
            "n_present": len(x),
            "mean_val": round(val_m, 6),
            "null_mean_val": round(nm, 6),
            "excess_over_null": round(val_m - nm, 6),
            "p_mc": round(p_mc, 5),
        })
    res = pd.DataFrame(rows)
    # BH-FDR across single genes
    m_sg = ~res["variant"].str.startswith("COMBO_")
    if m_sg.sum() > 0:
        pvals = res.loc[m_sg, "p_mc"].values
        order = np.argsort(pvals)
        ranked = pvals[order]
        q = np.minimum.accumulate((ranked * len(pvals) / (np.arange(len(pvals)) + 1))[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        q_out = np.empty_like(q)
        q_out[order] = q
        res.loc[m_sg, "q_BH"] = np.round(q_out, 5)
    return res

t1_df = eval_isp_trajectory(mg_long, "T1_Full_dP", lambda p0: True, "dP")
t2_df = eval_isp_trajectory(mg_long, "T2_Active_Pgt001_dP", lambda p0: p0 > 0.01, "dP")
t3_df = eval_isp_trajectory(mg_long, "T3_Elevated_Pgt005_dP", lambda p0: p0 > 0.05, "dP")
t4_df = eval_isp_trajectory(mg_long, "T4_NonSat_05_099_dP", lambda p0: 0.5 < p0 < 0.99, "dP")
t5_df = eval_isp_trajectory(mg_long, "T5_Full_dCos_Sham", lambda p0: True, "dCos_to_sham")

all_isp2_trajs = pd.concat([t1_df, t2_df, t3_df, t4_df, t5_df], ignore_index=True)
all_isp2_trajs.to_csv(os.path.join(OUT, "isp2_gse333879_all_trajectories.csv"), index=False)

# Build consensus summary across the 5 trajectories
isp2_cons_rows = []
for v in sorted(all_isp2_trajs["variant"].unique()):
    sub = all_isp2_trajs[all_isp2_trajs["variant"] == v]
    n_trajs = len(sub)
    # Significant positive reversal (excess > 0 and p_mc < 0.05)
    n_pos_sig = int(((sub["excess_over_null"] > 0) & (sub["p_mc"] < 0.05)).sum())
    n_neg_sig = int(((sub["excess_over_null"] < 0) & (sub["p_mc"] < 0.05)).sum())
    t1_row = sub[sub["trajectory"] == "T1_Full_dP"].iloc[0] if len(sub[sub["trajectory"] == "T1_Full_dP"]) > 0 else None
    t4_row = sub[sub["trajectory"] == "T4_NonSat_05_099_dP"].iloc[0] if len(sub[sub["trajectory"] == "T4_NonSat_05_099_dP"]) > 0 else None
    t5_row = sub[sub["trajectory"] == "T5_Full_dCos_Sham"].iloc[0] if len(sub[sub["trajectory"] == "T5_Full_dCos_Sham"]) > 0 else None

    isp2_cons_rows.append({
        "variant": v,
        "n_trajectories_evaluated": n_trajs,
        "T1_Full_n_present": int(t1_row["n_present"]) if t1_row is not None else 0,
        "T1_Full_mean_dP": float(t1_row["mean_val"]) if t1_row is not None else np.nan,
        "T1_Full_q_BH": float(t1_row["q_BH"]) if (t1_row is not None and "q_BH" in t1_row and pd.notna(t1_row["q_BH"])) else np.nan,
        "T4_NonSat_n_present": int(t4_row["n_present"]) if t4_row is not None else 0,
        "T4_NonSat_mean_dP": float(t4_row["mean_val"]) if t4_row is not None else np.nan,
        "T4_NonSat_q_BH": float(t4_row["q_BH"]) if (t4_row is not None and "q_BH" in t4_row and pd.notna(t4_row["q_BH"])) else np.nan,
        "T5_Full_mean_dCos_Sham": float(t5_row["mean_val"]) if t5_row is not None else np.nan,
        "T5_Full_p_mc": float(t5_row["p_mc"]) if t5_row is not None else np.nan,
        "n_pos_sig_p05": n_pos_sig,
        "n_neg_sig_p05": n_neg_sig,
        "pos_consensus_frac": round(n_pos_sig / 5.0, 2),
    })

isp2_cons_df = pd.DataFrame(isp2_cons_rows).sort_values(["pos_consensus_frac", "T1_Full_mean_dP"], ascending=[False, False])
isp2_cons_df.to_csv(os.path.join(OUT, "isp2_gse333879_5traj_consensus.csv"), index=False)
print("\n=== ISP PHASE 2 (UNSEEN GSE333879 D30) 5-TRAJECTORY CONSENSUS ===")
print(isp2_cons_df.to_string(index=False))

# Compare APOE & top hubs between Phase 1/2 (4 training cohorts) and Phase 2 Zero-Shot (unseen GSE333879)
p1_cons = pd.read_csv(os.path.join(BASE, "run2_weights", "isp_phase2_nonsat_consensus.csv"))
p1_mg = p1_cons[p1_cons["arm"] == "MG_injury"].set_index("gene")
p2_mg = isp2_cons_df[~isp2_cons_df["variant"].str.startswith("COMBO_")].set_index("variant")
shared_genes_full = [g for g in p1_mg.index if g in p2_mg.index and pd.notna(p1_mg.loc[g, "Full_mean_dP"]) and pd.notna(p2_mg.loc[g, "T1_Full_mean_dP"])]
rho_full, p_rho_full = sp_stats.spearmanr(
    p1_mg.loc[shared_genes_full, "Full_mean_dP"],
    p2_mg.loc[shared_genes_full, "T1_Full_mean_dP"]
)
shared_genes_ns = [g for g in p1_mg.index if g in p2_mg.index and pd.notna(p1_mg.loc[g, "T1_05_099_mean_dP"]) and pd.notna(p2_mg.loc[g, "T4_NonSat_mean_dP"])]
rho_nonsat, p_rho_nonsat = sp_stats.spearmanr(
    p1_mg.loc[shared_genes_ns, "T1_05_099_mean_dP"],
    p2_mg.loc[shared_genes_ns, "T4_NonSat_mean_dP"]
)

summary_out = {
    "exam2_human_GSE209552": {
        "n_samples": 17,
        "n_control_samples": 5,
        "n_tbi_samples": 12,
        "n_cells_total": int(len(df209)),
        "overall_cell_type_acc_canon6": float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "ALL", "ct_acc_canon6"].iloc[0]),
        "overall_cell_type_acc_canon6_hc80": float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "ALL", "ct_acc_canon6_hc80"].iloc[0]),
        "overall_cell_AUROC_prob": float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "ALL", "T1_CellProb_AUROC"].iloc[0]),
        "overall_donor_AUROC_prob": float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "ALL", "T4_DonorProb_AUROC"].iloc[0]),
        "overall_donor_mwu_p": float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "ALL", "T4_p"].iloc[0]),
        "microglia_ft_vs_frozen_cell_AUROC": [
            float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "Microglia", "T3_CellEmbShift_AUROC"].iloc[0]),
            float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "Microglia", "Frozen_CellEmbShift_AUROC"].iloc[0]),
        ],
        "astrocyte_ft_vs_frozen_cell_AUROC": [
            float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "Astrocyte", "T3_CellEmbShift_AUROC"].iloc[0]),
            float(exam2_cons_df.loc[exam2_cons_df["cell_type"] == "Astrocyte", "Frozen_CellEmbShift_AUROC"].iloc[0]),
        ],
    },
    "isp2_unseen_mouse_GSE333879": {
        "n_cells_total": 35047,
        "overall_cell_type_acc_clean": 0.9810,
        "overall_cell_type_acc_singlet_both": 0.9944,
        "top_5of5_consensus_reversers_MG_D30": isp2_cons_df[isp2_cons_df["pos_consensus_frac"] == 1.0]["variant"].tolist(),
        "top_4of5_or_better_consensus_reversers_MG_D30": isp2_cons_df[isp2_cons_df["pos_consensus_frac"] >= 0.80]["variant"].tolist(),
        "APOE_MG_D30_Full_dP": float(p2_mg.loc["APOE", "T1_Full_mean_dP"]),
        "APOE_MG_D30_NonSat_dP": float(p2_mg.loc["APOE", "T4_NonSat_mean_dP"]),
        "APOE_MG_D30_consensus": float(p2_mg.loc["APOE", "pos_consensus_frac"]),
        "SPP1_MG_D30_Full_dP": float(p2_mg.loc["SPP1", "T1_Full_mean_dP"]),
        "SPP1_MG_D30_NonSat_dP": float(p2_mg.loc["SPP1", "T4_NonSat_mean_dP"]),
        "SPP1_MG_D30_consensus": float(p2_mg.loc["SPP1", "pos_consensus_frac"]),
        "COMBO_TOP5_MG_D30_Full_dP": float(isp2_cons_df.loc[isp2_cons_df["variant"] == "COMBO_TOP5_HUB", "T1_Full_mean_dP"].iloc[0]),
        "COMBO_TOP5_MG_D30_NonSat_dP": float(isp2_cons_df.loc[isp2_cons_df["variant"] == "COMBO_TOP5_HUB", "T4_NonSat_mean_dP"].iloc[0]),
        "COMBO_TOP5_MG_D30_consensus": float(isp2_cons_df.loc[isp2_cons_df["variant"] == "COMBO_TOP5_HUB", "pos_consensus_frac"].iloc[0]),
        "cross_cohort_full_spearman_rho": round(float(rho_full), 4),
        "cross_cohort_full_spearman_p": round(float(p_rho_full), 5),
        "cross_cohort_nonsat_spearman_rho": round(float(rho_nonsat), 4),
        "cross_cohort_nonsat_spearman_p": round(float(p_rho_nonsat), 5),
    }
}
with open(os.path.join(OUT, "routeA_complete_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)
print("\n=== ROUTE A COMPLETE SUMMARY JSON ===")
print(json.dumps(summary_out, indent=2, ensure_ascii=False))
