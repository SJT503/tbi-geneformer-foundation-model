#!/usr/bin/env python3
"""ISP Phase 2 (Item 1, Free Offline): Non-saturated cell subset recomputation & 5-trajectory consensus.

Purpose:
  In ISP Phase 1 (all 800 MG_injury cells, 240 Astro_injury cells), the task2 classifier head
  exhibits ceiling saturation (mean P_injury = 0.9962 in MG_injury, 221/800 cells at P_injury = 1.00000),
  compressing single-gene deletion dP = P_orig - P_del via the sigmoid Jacobian P*(1-P).

  This script executes N=5 analytical trajectories (Robin Finch / agentic-data-analysis-loop standard)
  on run2_weights/isp_isp_cells_long.csv.gz:
    - T0 (Baseline Reference):     All cells (bit-for-bit reproduction of isp_stats_recomputed.csv)
    - T1 (Canonical Phase 2 spec): 0.5 < P_orig < 0.99  (MG_injury n=23, Astro_injury n=40)
    - T2 (Moderate window):        0.5 < P_orig < 0.995 (MG_injury n=29, Astro_injury n=47)
    - T3 (Broad window):           0.5 < P_orig < 0.999 (MG_injury n=61, Astro_injury n=74)
    - T4 (All non-1.00000 cells):  0.5 < P_orig < 1.0   (MG_injury n=577, Astro_injury n=220)
    - T5 (Logit-space + Cell-matched Monte Carlo permutation of sample mean, B=50,000)

Outputs (saved in run2_weights/):
  - isp_phase2_nonsat_T1_05_099.csv
  - isp_phase2_nonsat_all_trajectories.csv
  - isp_phase2_nonsat_consensus.csv
  - isp_phase2_nonsat_summary.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = "D:/2026-04-24/new-chat-2/TBI_astrocyte_scst_project/senescence_project/results/s1_gf_finetune/run2_weights"
IN_CSV = os.path.join(BASE_DIR, "isp_isp_cells_long.csv.gz")
REF_CSV = os.path.join(BASE_DIR, "isp_stats_recomputed.csv")

df = pd.read_csv(IN_CSV)
assert len(df) == 79452, f"Expected 79,452 rows in isp_isp_cells_long.csv.gz, got {len(df)}"

orig_df = df[df["variant"] == "__orig__"].copy()
assert len(orig_df) == 1160, f"Expected 1,160 orig cells, got {len(orig_df)}"
orig_map = orig_df.set_index("cell_i")["P_injury"]
df["P_orig"] = df["cell_i"].map(orig_map)

# Clip probabilities to [1e-5, 1 - 1e-5] for logit transform
eps = 1e-5
p_orig_clip = np.clip(df["P_orig"].values, eps, 1.0 - eps)
p_var_clip = np.clip(df["P_injury"].values, eps, 1.0 - eps)
df["logit_orig"] = np.log(p_orig_clip / (1.0 - p_orig_clip))
df["logit_var"] = np.log(p_var_clip / (1.0 - p_var_clip))
df["dLogit"] = df["logit_orig"] - df["logit_var"]

# Pre-index per-cell null vectors for cell-matched Monte Carlo permutation
null_all = df[df["variant"] == "__null__"]
cell_null_dp_dict = {cid: grp["dP"].values for cid, grp in null_all.groupby("cell_i")}
cell_null_dl_dict = {cid: grp["dLogit"].values for cid, grp in null_all.groupby("cell_i")}

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

N_PERM = 50000

def compute_window_stats(sub_df, trajectory_name, min_p=0.5, max_p=0.99, seed=42):
    """Exact statistical protocol matching isp_stats_recomputed.csv + cell-matched sample-mean permutation."""
    rng = np.random.RandomState(seed)
    valid_cells = orig_df[(orig_df["P_injury"] > min_p) & (orig_df["P_injury"] < max_p)]["cell_i"].unique()
    wdf = sub_df[sub_df["cell_i"].isin(valid_cells)].copy()

    null_rows = wdf[wdf["variant"] == "__null__"]
    cell_null_mean_dp = null_rows.groupby("cell_i")["dP"].mean()

    records = []
    for arm in ["MG_injury", "Astro_injury", "MG_sham"]:
        arm_df = wdf[wdf["arm"] == arm]
        arm_orig = arm_df[arm_df["variant"] == "__orig__"]
        n_cells_arm = len(arm_orig)
        if n_cells_arm == 0:
            continue
        p_orig_mean = float(arm_orig["P_injury"].mean())
        arm_nulls = arm_df[arm_df["variant"] == "__null__"]["dP"].values
        arm_nulls_dl = arm_df[arm_df["variant"] == "__null__"]["dLogit"].values
        null_mean = float(arm_nulls.mean()) if len(arm_nulls) else np.nan
        null_sd = float(arm_nulls.std(ddof=1)) if len(arm_nulls) > 1 else np.nan

        cand_df = arm_df[(arm_df["variant"] != "__orig__") & (arm_df["variant"] != "__null__")]
        arm_recs = []
        for gene, gdf in cand_df.groupby("variant"):
            n_pres = len(gdf)
            if n_pres == 0:
                continue
            cids = gdf["cell_i"].values
            dp_vals = gdf["dP"].values
            dl_vals = gdf["dLogit"].values
            mean_dp = float(dp_vals.mean())
            median_dp = float(np.median(dp_vals))
            mean_dl = float(dl_vals.mean())

            # 1. Phase-1 identical single-deletion empirical p (comparing rounded mean_dP vs raw pooled arm_nulls)
            mean_dp_r5 = round(mean_dp, 5)
            p_emp = round(float((arm_nulls >= mean_dp_r5).mean()), 5)
            sd_g = float(dp_vals.std(ddof=1)) if n_pres > 1 else np.nan
            z_canon = (mean_dp - null_mean) / (sd_g / np.sqrt(n_pres)) if (n_pres > 1 and sd_g > 0) else np.nan
            z_nullsd = (mean_dp - null_mean) / (null_sd / np.sqrt(n_pres)) if (n_pres > 0 and null_sd > 0) else np.nan

            # 2. Cell-matched sample-mean permutation test (B=50,000 draws of 1 null per expressing cell)
            null_mat = np.array([rng.choice(cell_null_dp_dict[c], size=N_PERM, replace=True) for c in cids])
            perm_means = null_mat.mean(axis=0)
            p_perm_pos = float((np.sum(perm_means >= mean_dp) + 1) / (N_PERM + 1))
            p_perm_neg = float((np.sum(perm_means <= mean_dp) + 1) / (N_PERM + 1))
            p_perm_2s = float(min(1.0, 2.0 * min(p_perm_pos, p_perm_neg)))
            z_cellperm = float((mean_dp - perm_means.mean()) / (perm_means.std(ddof=1) + 1e-12))

            # 3. Logit-space cell-matched sample-mean permutation
            null_mat_dl = np.array([rng.choice(cell_null_dl_dict[c], size=N_PERM, replace=True) for c in cids])
            perm_means_dl = null_mat_dl.mean(axis=0)
            p_perm_dl_pos = float((np.sum(perm_means_dl >= mean_dl) + 1) / (N_PERM + 1))
            p_perm_dl_neg = float((np.sum(perm_means_dl <= mean_dl) + 1) / (N_PERM + 1))
            p_perm_dl_2s = float(min(1.0, 2.0 * min(p_perm_dl_pos, p_perm_dl_neg)))

            # Paired within-cell excess dP
            c_nm = gdf["cell_i"].map(cell_null_mean_dp).values
            paired_mean = float((dp_vals - c_nm).mean())

            arm_recs.append({
                "trajectory": trajectory_name,
                "window": f"{min_p}<P<{max_p}",
                "arm": arm,
                "n_cells_arm": n_cells_arm,
                "gene": gene,
                "n_present": n_pres,
                "presence_rate": round(n_pres / n_cells_arm, 4),
                "mean_dP": round(mean_dp, 5),
                "median_dP": round(median_dp, 5),
                "paired_excess_dP": round(paired_mean, 5),
                "mean_dLogit": round(mean_dl, 4),
                "p_emp": round(p_emp, 5),
                "z_canon": round(float(z_canon), 2) if not np.isnan(z_canon) else np.nan,
                "z_vs_nullsd": round(float(z_nullsd), 2) if not np.isnan(z_nullsd) else np.nan,
                "z_cellperm": round(z_cellperm, 2),
                "p_perm_1s_pos": round(p_perm_pos, 5),
                "p_perm_2s": round(p_perm_2s, 5),
                "p_perm_dl_2s": round(p_perm_dl_2s, 5),
                "P_injury_orig_mean": round(p_orig_mean, 4),
                "task1_flip_rate": round(float(gdf["task1_flip"].mean()), 4),
            })

        if arm_recs:
            adf = pd.DataFrame(arm_recs)
            adf = adf.sort_values("mean_dP", ascending=False).reset_index(drop=True)
            adf["q_BH"] = np.round(bh_fdr(adf["p_emp"].values), 5)
            adf["q_BH_perm_1s"] = np.round(bh_fdr(adf["p_perm_1s_pos"].values), 5)
            adf["q_BH_perm_2s"] = np.round(bh_fdr(adf["p_perm_2s"].values), 5)
            adf["q_BH_dl_2s"] = np.round(bh_fdr(adf["p_perm_dl_2s"].values), 5)
            adf["rank_in_arm"] = np.arange(1, len(adf) + 1)
            records.append(adf)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# Execute T0 (Full) + T1..T4 non-saturated windows
t0_full = compute_window_stats(df, "T0_Full_AllCells", min_p=-0.01, max_p=1.01, seed=42)
t1 = compute_window_stats(df, "T1_Canonical_05_099", min_p=0.5, max_p=0.99, seed=101)
t2 = compute_window_stats(df, "T2_Moderate_05_0995", min_p=0.5, max_p=0.995, seed=102)
t3 = compute_window_stats(df, "T3_Broad_05_0999", min_p=0.5, max_p=0.999, seed=103)
t4 = compute_window_stats(df, "T4_NonCeiling_05_100", min_p=0.5, max_p=1.0, seed=104)

# Verify T0_Full bit-for-bit match with isp_stats_recomputed.csv on MG_injury APOE
ref_df = pd.read_csv(REF_CSV)
t0_mg_apoe = t0_full[(t0_full["arm"] == "MG_injury") & (t0_full["gene"] == "APOE")].iloc[0]
ref_mg_apoe = ref_df[(ref_df["arm"] == "MG_injury") & (ref_df["gene"] == "APOE")].iloc[0]
assert t0_mg_apoe["mean_dP"] == ref_mg_apoe["mean_dP"] == 0.00233
assert t0_mg_apoe["p_emp"] == ref_mg_apoe["p_emp"] == 0.01562

t1.to_csv(os.path.join(BASE_DIR, "isp_phase2_nonsat_T1_05_099.csv"), index=False)
all_traj = pd.concat([t0_full, t1, t2, t3, t4], ignore_index=True)
all_traj.to_csv(os.path.join(BASE_DIR, "isp_phase2_nonsat_all_trajectories.csv"), index=False)

# Build 5-Trajectory Consensus Table for MG_injury and Astro_injury
# 5 Independent Trajectories for Consensus:
#   Traj 1: T1 (0.5<P<0.99) cell-matched perm q_BH_perm_2s < 0.05
#   Traj 2: T2 (0.5<P<0.995) cell-matched perm q_BH_perm_2s < 0.05
#   Traj 3: T3 (0.5<P<0.999) cell-matched perm q_BH_perm_2s < 0.05
#   Traj 4: T4 (0.5<P<1.0) cell-matched perm q_BH_perm_2s < 0.05
#   Traj 5: T3 Logit-space (0.5<P<0.999) q_BH_dl_2s < 0.05
cons_rows = []
for arm in ["MG_injury", "Astro_injury"]:
    genes = sorted(all_traj[all_traj["arm"] == arm]["gene"].unique())
    for g in genes:
        sub = all_traj[(all_traj["arm"] == arm) & (all_traj["gene"] == g)].set_index("trajectory")
        def get_val(tr, col):
            return float(sub.loc[tr, col]) if tr in sub.index else np.nan

        pos_votes = 0
        neg_votes = 0
        for tr in ["T1_Canonical_05_099", "T2_Moderate_05_0995", "T3_Broad_05_0999", "T4_NonCeiling_05_100"]:
            if tr in sub.index:
                if get_val(tr, "mean_dP") > 0 and get_val(tr, "q_BH_perm_2s") < 0.05:
                    pos_votes += 1
                if get_val(tr, "mean_dP") < 0 and get_val(tr, "q_BH_perm_2s") < 0.05:
                    neg_votes += 1
        if "T3_Broad_05_0999" in sub.index:
            if get_val("T3_Broad_05_0999", "mean_dLogit") > 0 and get_val("T3_Broad_05_0999", "q_BH_dl_2s") < 0.05:
                pos_votes += 1
            if get_val("T3_Broad_05_0999", "mean_dLogit") < 0 and get_val("T3_Broad_05_0999", "q_BH_dl_2s") < 0.05:
                neg_votes += 1

        dp_full = get_val("T0_Full_AllCells", "mean_dP")
        dp_t1 = get_val("T1_Canonical_05_099", "mean_dP")
        dp_t3 = get_val("T3_Broad_05_0999", "mean_dP")
        fold_t1 = round(dp_t1 / dp_full, 2) if (not np.isnan(dp_t1) and not np.isnan(dp_full) and abs(dp_full) > 1e-6) else np.nan
        fold_t3 = round(dp_t3 / dp_full, 2) if (not np.isnan(dp_t3) and not np.isnan(dp_full) and abs(dp_full) > 1e-6) else np.nan

        cons_rows.append({
            "arm": arm,
            "gene": g,
            "Full_n": get_val("T0_Full_AllCells", "n_present"),
            "Full_mean_dP": dp_full,
            "Full_p_emp_single": get_val("T0_Full_AllCells", "p_emp"),
            "Full_q_BH_single": get_val("T0_Full_AllCells", "q_BH"),
            "Full_q_BH_cellperm_2s": get_val("T0_Full_AllCells", "q_BH_perm_2s"),
            "T1_05_099_n": get_val("T1_Canonical_05_099", "n_present"),
            "T1_05_099_mean_dP": dp_t1,
            "T1_amplification_vs_Full": fold_t1,
            "T1_p_emp_single": get_val("T1_Canonical_05_099", "p_emp"),
            "T1_q_BH_single": get_val("T1_Canonical_05_099", "q_BH"),
            "T1_z_cellperm": get_val("T1_Canonical_05_099", "z_cellperm"),
            "T1_q_BH_cellperm_2s": get_val("T1_Canonical_05_099", "q_BH_perm_2s"),
            "T3_05_0999_n": get_val("T3_Broad_05_0999", "n_present"),
            "T3_05_0999_mean_dP": dp_t3,
            "T3_amplification_vs_Full": fold_t3,
            "T3_z_cellperm": get_val("T3_Broad_05_0999", "z_cellperm"),
            "T3_q_BH_cellperm_2s": get_val("T3_Broad_05_0999", "q_BH_perm_2s"),
            "T4_05_100_n": get_val("T4_NonCeiling_05_100", "n_present"),
            "T4_05_100_mean_dP": get_val("T4_NonCeiling_05_100", "mean_dP"),
            "T4_q_BH_cellperm_2s": get_val("T4_NonCeiling_05_100", "q_BH_perm_2s"),
            "pos_pillar_votes_of_5": pos_votes,
            "pos_pillar_robustness": round(pos_votes / 5.0, 2),
            "neg_anchor_votes_of_5": neg_votes,
            "neg_anchor_robustness": round(neg_votes / 5.0, 2),
        })

cons_df = pd.DataFrame(cons_rows).sort_values(
    ["arm", "pos_pillar_votes_of_5", "T1_05_099_mean_dP"], ascending=[True, False, False]
)
cons_df.to_csv(os.path.join(BASE_DIR, "isp_phase2_nonsat_consensus.csv"), index=False)

# Dynamically extract key findings from cons_df (no hardcoded literals)
mg_cons = cons_df[cons_df["arm"] == "MG_injury"].set_index("gene")
def _kf(gene_sym):
    r = mg_cons.loc[gene_sym]
    return {
        "Full_mean_dP": float(r["Full_mean_dP"]),
        "T1_05_099_n": int(r["T1_05_099_n"]),
        "T1_05_099_mean_dP": float(r["T1_05_099_mean_dP"]),
        "T1_amplification_vs_Full": float(r["T1_amplification_vs_Full"]),
        "T1_p_emp_single": float(r["T1_p_emp_single"]),
        "T1_z_cellperm": float(r["T1_z_cellperm"]),
        "T1_q_BH_cellperm_2s": float(r["T1_q_BH_cellperm_2s"]),
        "T3_05_0999_n": int(r["T3_05_0999_n"]),
        "T3_05_0999_mean_dP": float(r["T3_05_0999_mean_dP"]),
        "T3_q_BH_cellperm_2s": float(r["T3_q_BH_cellperm_2s"]),
        "pos_pillar_consensus": f"{int(r['pos_pillar_votes_of_5'])}/5 ({float(r['pos_pillar_robustness'])})",
        "neg_anchor_consensus": f"{int(r['neg_anchor_votes_of_5'])}/5 ({float(r['neg_anchor_robustness'])})",
    }

summary_meta = {
    "script": "isp_phase2_nonsat_recompute.py",
    "input_rows": len(df),
    "n_orig_cells": len(orig_df),
    "n_permutations_cellmatched": N_PERM,
    "cell_counts_by_window": {
        "T0_Full_AllCells": {"MG_injury": 800, "Astro_injury": 240, "MG_sham": 120},
        "T1_Canonical_05_099": {"MG_injury": 23, "Astro_injury": 40, "MG_sham": 3},
        "T2_Moderate_05_0995": {"MG_injury": 29, "Astro_injury": 47, "MG_sham": 3},
        "T3_Broad_05_0999": {"MG_injury": 61, "Astro_injury": 74, "MG_sham": 3},
        "T4_NonCeiling_05_100": {"MG_injury": 577, "Astro_injury": 220, "MG_sham": 3},
    },
    "key_findings_MG_injury": {
        "APOE": _kf("APOE"),
        "IL1B": _kf("IL1B"),
        "CSF1R": _kf("CSF1R"),
    }
}
with open(os.path.join(BASE_DIR, "isp_phase2_nonsat_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_meta, f, indent=2, ensure_ascii=False)

print("=== DONE: Saved T1, All-Trajectories, Consensus, and Summary JSON ===")
mg_c = cons_df[cons_df["arm"] == "MG_injury"]
print(mg_c[[
    "gene", "Full_mean_dP", "T1_05_099_n", "T1_05_099_mean_dP", "T1_amplification_vs_Full",
    "T1_p_emp_single", "T1_q_BH_cellperm_2s", "T3_05_0999_mean_dP", "T3_q_BH_cellperm_2s",
    "pos_pillar_votes_of_5", "neg_anchor_votes_of_5"
]].to_string(index=False))
