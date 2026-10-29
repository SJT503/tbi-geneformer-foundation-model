#!/usr/bin/env python3
"""
Tier-1 Upgrade #1: Conventional & Lightweight ML Baselines vs. TBI-Geneformer
=============================================================================
Benchmarks 5 standard non-FM single-cell methods against Frozen GF-V2-104M,
TBI-GF-le7d (Exam 1 specialist), and TBI-GF run2 (Full specialist) on the
EXACT same splits:
  1. Exam 1 In-Domain <=7d Test (n = 45,675 nuclei)
  2. Exam 1 Subacute D28 Temporal Extrapolation (GSE271769, n = 39,001 nuclei, 8 samples)
  3. Exam 1 Chronic 6mo Temporal Extrapolation (GSE269748, n = 48,595 nuclei, 11 samples)
  4. Exam 2 Zero-Shot Mouse-to-Human Clinical Transfer (GSE209552, n = 24,241 nuclei, 17 donors)

Five Conventional Baselines (trained strictly on the mouse <=7d training pool):
  - B1_HVG2000_LogReg: Top 2,000 HVGs (1:1 orthologs) + L2-Regularized Logistic Regression
  - B2_HVG2000_GBDT:   Top 2,000 HVGs + HistGradientBoosting Tree Ensemble
  - B3_PCA50_MLP:      50-component TruncatedSVD/PCA + 2-Layer Shallow MLP (128-64)
  - B4_Harmony_ZPCA50: Cohort/Domain Z-Standardized PCA-50 (Harmony-style linear batch alignment) + L2 Classifier
  - B5_DAM_SenMayo_Sig: Curated DAM + SenMayo minus Homeostatic Knowledge Signature Score

Follows agentic-data-analysis-loop: 5 trajectories (T1-T5) + consensus verification.
"""

import os
import sys
import glob
import time
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

PROJ = r"D:\2026-04-24\new-chat-2\TBI_astrocyte_scst_project"
SEN = os.path.join(PROJ, "senescence_project")
OUT_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "tier1_upgrades")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load GC-104M dictionaries
DICT_DIR = r"F:\fm_tbi_training\kaggle_data"
TOK_PKL = os.path.join(DICT_DIR, "token_dictionary_gc104M.pkl")
NAME_PKL = os.path.join(DICT_DIR, "gene_name_id_dict_gc104M.pkl")
tok_dict = pickle.load(open(TOK_PKL, "rb"))
name_dict = pickle.load(open(NAME_PKL, "rb"))
inv_tok = {v: k for k, v in tok_dict.items()}
inv_name = {v: k for k, v in name_dict.items()}

# 2. Load Exam 1 split manifest & predictions
EXAM1_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "exam1_out")
man_df = pd.read_csv(os.path.join(EXAM1_DIR, "exam1_split_manifest.csv.gz"))
preds_exam1 = pd.read_csv(os.path.join(EXAM1_DIR, "exam1_all_eval_preds.csv.gz"))
log(f"Loaded Exam 1 manifest ({len(man_df):,} cells) & eval preds ({len(preds_exam1):,} cells)")

# 3. Load tokenized 562k mouse Arrow files and filter is_dbl == 0 (536,619 cells)
ARROW_DIR = r"F:\fm_tbi_training\tokenized\kernel_out5\tbi_562k.dataset"
arrow_files = sorted(glob.glob(os.path.join(ARROW_DIR, "data-*.arrow")))
assert len(arrow_files) == 7, f"Expected 7 arrow shards, found {len(arrow_files)}"

tables = []
for af in arrow_files:
    with pa.memory_map(af, "r") as source:
        reader = ipc.open_stream(source)
        tables.append(reader.read_all())
full_tbl = pa.concat_tables(tables)
assert len(full_tbl) == 562633, f"Expected 562,633 total rows, got {len(full_tbl)}"

is_dbl_arr = full_tbl["is_dbl"].to_numpy()
clean_orig_idx = np.nonzero(is_dbl_arr == 0)[0]
assert len(clean_orig_idx) == 536619, f"Expected 536,619 clean rows, got {len(clean_orig_idx)}"
log("Verified 536,619 clean mouse singlets from Arrow shards.")

# Zero-copy flat numpy representation of all sequences across the 7 shards
shard_offsets = []
shard_values = []
shard_row_starts = [0]
for t_shard in tables:
    col_c = t_shard["input_ids"].combine_chunks()
    shard_offsets.append(col_c.offsets.to_numpy())
    shard_values.append(col_c.values.to_numpy())
    shard_row_starts.append(shard_row_starts[-1] + len(t_shard))
shard_row_starts = np.array(shard_row_starts, dtype=np.int64)

def get_seq_numpy(clean_row_idx):
    raw_idx = int(clean_orig_idx[clean_row_idx])
    s_i = int(np.searchsorted(shard_row_starts[1:], raw_idx, side="right"))
    local_r = raw_idx - int(shard_row_starts[s_i])
    offs = shard_offsets[s_i]
    return shard_values[s_i][offs[local_r]:offs[local_r + 1]]

train_rows_all = man_df.index[man_df["split"] == "train"].values
rng = np.random.RandomState(42)
train_sub_rows = np.sort(rng.choice(train_rows_all, size=60000, replace=False))
eval_rows_all = preds_exam1["orig_row"].values  # 125,667 rows in exact order of preds_exam1

log(f"Extracting zero-copy numpy token slices for {len(train_sub_rows):,} <=7d training cells and {len(eval_rows_all):,} Exam 1 evaluation cells...")
train_seqs = [get_seq_numpy(r) for r in train_sub_rows]
train_y = (man_df["grp"].iloc[train_sub_rows].values == "injury").astype(np.int32)
train_ct = man_df["cell_type"].iloc[train_sub_rows].values
train_ds_name = man_df["dataset"].iloc[train_sub_rows].values

eval_seqs = [get_seq_numpy(r) for r in eval_rows_all]
log("Zero-copy numpy token sequences ready.")

# 4. Load Human GSE209552 tokenized data & Route A predictions
ROUTEA_TOK = os.path.join(SEN, "results", "s1_gf_finetune", "routeA_tokenized", "gse209552_tokenized.pkl")
ROUTEA_PREDS = os.path.join(SEN, "results", "s1_gf_finetune", "routeA_out", "exam2_gse209552_cell_preds.csv.gz")
d209 = pickle.load(open(ROUTEA_TOK, "rb"))
preds_209 = pd.read_csv(ROUTEA_PREDS)
seqs_209 = d209["input_ids"]
assert len(seqs_209) == 24241 and len(preds_209) == 24241
log(f"Loaded Human GSE209552 ({len(seqs_209):,} nuclei across 17 clinical donors).")

# 5. Select Top 2,000 Highly Variable Ortholog Genes (HVG-2000) strictly on <=7d training pool
MAX_TOK_ID = max(tok_dict.values()) + 1
CLS_ID = tok_dict.get("<cls>", 2)
EOS_ID = tok_dict.get("<eos>", 3)
PAD_ID = tok_dict.get("<pad>", 0)
SPECIAL_IDS = {CLS_ID, EOS_ID, PAD_ID, 1}

log("Computing gene frequency and rank-expression variance across <=7d training pool...")
sum_x = np.zeros(MAX_TOK_ID, dtype=np.float64)
sum_x2 = np.zeros(MAX_TOK_ID, dtype=np.float64)
count_x = np.zeros(MAX_TOK_ID, dtype=np.int64)

for seq in train_seqs:
    L = len(seq)
    if L <= 2:
        continue
    arr = np.asarray(seq[1:-1], dtype=np.int64)
    n_g = len(arr)
    # Rank-normalized expression in (0, 1]: highest rank = 1.0, lowest = 1/n_g
    vals = 1.0 - (np.arange(n_g, dtype=np.float64) / float(n_g))
    sum_x[arr] += vals
    sum_x2[arr] += vals * vals
    count_x[arr] += 1

N_tr = float(len(train_seqs))
mean_x = sum_x / N_tr
var_x = (sum_x2 / N_tr) - (mean_x * mean_x)
for sid in SPECIAL_IDS:
    if sid < MAX_TOK_ID:
        var_x[sid] = -1.0
# Require detection in at least 0.5% of training cells
var_x[count_x < int(0.005 * N_tr)] = -1.0

hvg_tok_ids = np.argsort(-var_x)[:2000]
tok_to_hvg_col = np.full(MAX_TOK_ID, -1, dtype=np.int32)
for col_idx, tid in enumerate(hvg_tok_ids):
    tok_to_hvg_col[tid] = col_idx

hvg_symbols = [inv_name.get(inv_tok.get(int(t), ""), f"TOK_{t}") for t in hvg_tok_ids]
log(f"Selected top 2,000 HVGs from <=7d training pool. Top 15 HVGs: {hvg_symbols[:15]}")

def build_sparse_hvg_matrix(seqs, tok_map, n_cols=2000):
    """Builds a CSR matrix (n_cells, n_cols) of rank-normalized expression [0, 1]."""
    rows_list = []
    cols_list = []
    vals_list = []
    for i, seq in enumerate(seqs):
        if len(seq) <= 2:
            continue
        arr = np.asarray(seq[1:-1], dtype=np.int64)
        valid = arr < len(tok_map)
        if not np.all(valid):
            arr = arr[valid]
        cols = tok_map[arr]
        mask = cols >= 0
        if np.any(mask):
            c_sel = cols[mask]
            n_g = float(len(arr))
            ranks = np.nonzero(mask)[0].astype(np.float32)
            v_sel = 1.0 - (ranks / n_g)
            rows_list.append(np.full(len(c_sel), i, dtype=np.int32))
            cols_list.append(c_sel.astype(np.int32))
            vals_list.append(v_sel)
    r_all = np.concatenate(rows_list)
    c_all = np.concatenate(cols_list)
    v_all = np.concatenate(vals_list)
    return sp.csr_matrix((v_all, (r_all, c_all)), shape=(len(seqs), n_cols), dtype=np.float32)

log("Building sparse HVG-2000 matrices for train_le7d, Exam 1 eval, and Exam 2 human GSE209552...")
X_train = build_sparse_hvg_matrix(train_seqs, tok_to_hvg_col, 2000)
X_eval1 = build_sparse_hvg_matrix(eval_seqs, tok_to_hvg_col, 2000)
X_eval2 = build_sparse_hvg_matrix(seqs_209, tok_to_hvg_col, 2000)
log(f"Matrices built: X_train={X_train.shape} (nnz={X_train.nnz:,}), X_eval1={X_eval1.shape}, X_eval2={X_eval2.shape}")

# 6. Train Conventional Baselines on <=7d Mouse Training Pool
# B1: HVG-2000 + L2-Regularized Logistic Regression
log("Fitting Baseline 1: B1_HVG2000_LogReg ...")
b1_clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", n_jobs=-1, random_state=42)
b1_clf.fit(X_train, train_y)
p_b1_eval1 = b1_clf.predict_proba(X_eval1)[:, 1].astype(np.float32)
p_b1_eval2 = b1_clf.predict_proba(X_eval2)[:, 1].astype(np.float32)

# Fit TruncatedSVD (PCA-50) on X_train
log("Fitting 50-component TruncatedSVD (PCA-50) on X_train ...")
svd50 = TruncatedSVD(n_components=50, random_state=42)
Z_train = svd50.fit_transform(X_train)
Z_eval1 = svd50.transform(X_eval1)
Z_eval2 = svd50.transform(X_eval2)

# Also select top 300 HVGs for dense tree-ensemble (HistGradientBoosting)
X_train_300 = X_train[:, :300].toarray()
X_eval1_300 = X_eval1[:, :300].toarray()
X_eval2_300 = X_eval2[:, :300].toarray()

# B2: HVG + HistGradientBoosting (Gradient Boosted Decision Trees)
log("Fitting Baseline 2: B2_HVG_GBDT (HistGradientBoosting) ...")
b2_clf = HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.1, random_state=42)
b2_clf.fit(X_train_300, train_y)
p_b2_eval1 = b2_clf.predict_proba(X_eval1_300)[:, 1].astype(np.float32)
p_b2_eval2 = b2_clf.predict_proba(X_eval2_300)[:, 1].astype(np.float32)

# B3: PCA-50 + Shallow MLP (2-layer neural network 128-64)
log("Fitting Baseline 3: B3_PCA50_MLP ...")
scaler_pca = StandardScaler()
Z_train_std = scaler_pca.fit_transform(Z_train)
Z_eval1_std = scaler_pca.transform(Z_eval1)
Z_eval2_std = scaler_pca.transform(Z_eval2)

b3_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=40, alpha=1e-3, random_state=42, early_stopping=True)
b3_clf.fit(Z_train_std, train_y)
p_b3_eval1 = b3_clf.predict_proba(Z_eval1_std)[:, 1].astype(np.float32)
p_b3_eval2 = b3_clf.predict_proba(Z_eval2_std)[:, 1].astype(np.float32)

# B4: Harmony-Style Cohort-Standardized PCA-50 + L2 Linear Classifier
# Standardizes each dataset/cohort's PCA-50 distribution to zero mean & unit variance before classification
log("Fitting Baseline 4: B4_Harmony_ZPCA50 ...")
def domain_standardize(Z, domain_labels):
    Z_out = np.zeros_like(Z, dtype=np.float32)
    for dom in np.unique(domain_labels):
        m = (domain_labels == dom)
        if np.sum(m) > 5:
            Z_out[m] = StandardScaler().fit_transform(Z[m])
        else:
            Z_out[m] = Z[m]
    return Z_out

Z_train_harm = domain_standardize(Z_train, train_ds_name)
Z_eval1_harm = domain_standardize(Z_eval1, preds_exam1["dataset"].values)
Z_eval2_harm = StandardScaler().fit_transform(Z_eval2)

b4_clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
b4_clf.fit(Z_train_harm, train_y)
p_b4_eval1 = b4_clf.predict_proba(Z_eval1_harm)[:, 1].astype(np.float32)
p_b4_eval2 = b4_clf.predict_proba(Z_eval2_harm)[:, 1].astype(np.float32)

# B5: Knowledge-Driven Curated DAM + SenMayo Signature Score
log("Computing Baseline 5: B5_DAM_SenMayo_Sig ...")
POS_SIG_GENES = [
    "APOE", "SPP1", "TREM2", "LPL", "CST7", "ITGAX", "CD9", "CTSD", "CTSB", "CTSZ",
    "TYROBP", "B2M", "FTH1", "C1QA", "C1QB", "C1QC", "GFAP", "VIM", "SERPINA3",
    "CDKN1A", "CDKN2A", "IL1B", "CCL2", "STAT1", "IRF7", "MIF", "BAX", "LGALS3", "GPNMB"
]
NEG_SIG_GENES = [
    "P2RY12", "TMEM119", "CX3CR1", "CSF1R", "SALL1", "HEXBM", "SLC1A2", "ALDH1L1", "GJA1"
]
pos_tok_set = {tok_dict[name_dict[g]] for g in POS_SIG_GENES if g in name_dict and name_dict[g] in tok_dict}
neg_tok_set = {tok_dict[name_dict[g]] for g in NEG_SIG_GENES if g in name_dict and name_dict[g] in tok_dict}
log(f"Matched {len(pos_tok_set)} positive DAM/SenMayo tokens and {len(neg_tok_set)} homeostatic anchor tokens.")

sig_weight_by_tok = np.zeros(MAX_TOK_ID, dtype=np.float32)
for t in pos_tok_set:
    sig_weight_by_tok[t] = 1.0
for t in neg_tok_set:
    sig_weight_by_tok[t] = -1.0

def compute_signature_score(seqs):
    scores = np.zeros(len(seqs), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if len(seq) <= 2:
            continue
        arr = np.asarray(seq[1:-1], dtype=np.int64)
        valid = arr < MAX_TOK_ID
        if not np.all(valid):
            arr = arr[valid]
        w = sig_weight_by_tok[arr]
        nz = np.nonzero(w)[0]
        if len(nz) > 0:
            n_g = float(len(arr))
            ranks_val = 1.0 - (nz.astype(np.float32) / n_g)
            scores[i] = float(np.sum(w[nz] * ranks_val))
    return scores

p_b5_eval1 = compute_signature_score(eval_seqs)
p_b5_eval2 = compute_signature_score(seqs_209)

# Attach all baseline scores to preds_exam1 and preds_209
preds_exam1["B1_HVG2000_LogReg"] = p_b1_eval1
preds_exam1["B2_HVG_GBDT"] = p_b2_eval1
preds_exam1["B3_PCA50_MLP"] = p_b3_eval1
preds_exam1["B4_Harmony_ZPCA50"] = p_b4_eval1
preds_exam1["B5_DAM_SenMayo_Sig"] = p_b5_eval1

preds_209["B1_HVG2000_LogReg"] = p_b1_eval2
preds_209["B2_HVG_GBDT"] = p_b2_eval2
preds_209["B3_PCA50_MLP"] = p_b3_eval2
preds_209["B4_Harmony_ZPCA50"] = p_b4_eval2
preds_209["B5_DAM_SenMayo_Sig"] = p_b5_eval2

# 7. Define Evaluation Benchmarks & 5-Trajectory Consensus
MAJOR_GLIA = {"Microglia", "Astrocyte", "Oligo", "Oligodendrocyte", "OPC"}

# Build masks for Exam 1 cohorts
m_indom = (preds_exam1["split"] == "test_le7d").values
m_d28 = ((preds_exam1["split"] == "holdout_D28") | ((preds_exam1["split"] == "test_le7d") & (preds_exam1["dataset"] == "GSE271769") & (preds_exam1["grp"] == "sham"))).values
m_6mo = ((preds_exam1["split"] == "holdout_6mo") | ((preds_exam1["split"] == "test_le7d") & (preds_exam1["dataset"] == "GSE269748") & (preds_exam1["grp"] == "sham"))).values

assert np.sum(m_indom) == 45675, f"Expected 45,675 InDomain cells, got {np.sum(m_indom)}"
assert np.sum(m_d28) == 39001, f"Expected 39,001 D28 cells, got {np.sum(m_d28)}"
assert np.sum(m_6mo) == 48595, f"Expected 48,595 6mo cells, got {np.sum(m_6mo)}"

# Methods to compare on Exam 1 and Exam 2
METHODS_EXAM1 = [
    ("TBI_GF_Specialist", "P_injury_le7d", "emb_shift_le7d"),
    ("TBI_GF_Full_Oracle", "P_injury_oracle", "emb_shift_oracle"),
    ("Frozen_GF_V2_104M", "emb_shift_frozen", "emb_shift_frozen"),
    ("B1_HVG2000_LogReg", "B1_HVG2000_LogReg", "B1_HVG2000_LogReg"),
    ("B2_HVG_GBDT", "B2_HVG_GBDT", "B2_HVG_GBDT"),
    ("B3_PCA50_MLP", "B3_PCA50_MLP", "B3_PCA50_MLP"),
    ("B4_Harmony_ZPCA50", "B4_Harmony_ZPCA50", "B4_Harmony_ZPCA50"),
    ("B5_DAM_SenMayo_Sig", "B5_DAM_SenMayo_Sig", "B5_DAM_SenMayo_Sig"),
]

METHODS_EXAM2 = [
    ("TBI_GF_Specialist", "P_injury", "ft_injury_cos_score"),
    ("TBI_GF_Full_Oracle", "P_injury", "ft_injury_cos_score"),
    ("Frozen_GF_V2_104M", "frozen_injury_score", "frozen_injury_score"),
    ("B1_HVG2000_LogReg", "B1_HVG2000_LogReg", "B1_HVG2000_LogReg"),
    ("B2_HVG_GBDT", "B2_HVG_GBDT", "B2_HVG_GBDT"),
    ("B3_PCA50_MLP", "B3_PCA50_MLP", "B3_PCA50_MLP"),
    ("B4_Harmony_ZPCA50", "B4_Harmony_ZPCA50", "B4_Harmony_ZPCA50"),
    ("B5_DAM_SenMayo_Sig", "B5_DAM_SenMayo_Sig", "B5_DAM_SenMayo_Sig"),
]

def safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))

def safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))

benchmark_rows = []
traj_rows = []

cohorts_spec = [
    ("Exam1_InDomain_le7d", preds_exam1.loc[m_indom].copy(), METHODS_EXAM1),
    ("Exam1_Extrap_D28_GSE271769", preds_exam1.loc[m_d28].copy(), METHODS_EXAM1),
    ("Exam1_Extrap_6mo_GSE269748", preds_exam1.loc[m_6mo].copy(), METHODS_EXAM1),
    ("Exam2_Human_GSE209552", preds_209.copy(), METHODS_EXAM2),
]

for cohort_name, df_c, m_list in cohorts_spec:
    y_all = (df_c["grp"].values == "injury").astype(np.int32)
    ct_col_name = "ct" if "ct" in df_c.columns else "cell_type"
    ct_all = df_c[ct_col_name].values
    m_glia = np.isin(ct_all, list(MAJOR_GLIA))
    m_mg = (ct_all == "Microglia")
    m_as = (ct_all == "Astrocyte")

    # Sample-level grouping
    samp_col = "sample"

    # Bootstrap 80% indices
    rng_b = np.random.RandomState(42)
    boot_idx = rng_b.choice(len(df_c), size=int(0.8 * len(df_c)), replace=False)

    for m_name, col_prob, col_emb in m_list:
        s_prob = df_c[col_prob].values.astype(np.float64)
        s_emb = df_c[col_emb].values.astype(np.float64)
        # Primary score for each method: best of prob/emb for TBI-GF, direct score for baselines
        s_best = s_emb if (cohort_name in ["Exam1_Extrap_6mo_GSE269748", "Exam2_Human_GSE209552"] and "TBI_GF" in m_name) else s_prob

        auroc_all = safe_auroc(y_all, s_best)
        auprc_all = safe_auprc(y_all, s_best)
        auroc_prob_all = safe_auroc(y_all, s_prob)
        auroc_emb_all = safe_auroc(y_all, s_emb)

        auroc_glia = safe_auroc(y_all[m_glia], s_best[m_glia])
        auroc_mg = safe_auroc(y_all[m_mg], s_best[m_mg])
        auroc_as = safe_auroc(y_all[m_as], s_best[m_as])
        auroc_boot = safe_auroc(y_all[boot_idx], s_best[boot_idx])

        # Sample / Donor-level AUROC (using Microglia mean score for human GSE209552 or all-nuclei mean score)
        df_c_tmp = df_c.copy()
        df_c_tmp["_score"] = s_prob if "TBI_GF" in m_name else s_best
        if cohort_name == "Exam2_Human_GSE209552":
            df_mg_tmp = df_c_tmp[df_c_tmp[ct_col_name] == "Microglia"]
            samp_df = df_mg_tmp.groupby(samp_col).agg(
                y=("grp", lambda x: 1 if (x == "injury").mean() > 0.5 else 0),
                score=("_score", "mean")
            )
        else:
            samp_df = df_c_tmp.groupby(samp_col).agg(
                y=("grp", lambda x: 1 if (x == "injury").mean() > 0.5 else 0),
                score=("_score", "mean")
            )
        auroc_sample = safe_auroc(samp_df["y"].values, samp_df["score"].values)

        benchmark_rows.append({
            "cohort": cohort_name,
            "method": m_name,
            "n_cells": int(len(df_c)),
            "n_samples": int(len(samp_df)),
            "auroc_all_nuclei": round(auroc_all, 4),
            "auprc_all_nuclei": round(auprc_all, 4),
            "auroc_prob_head": round(auroc_prob_all, 4),
            "auroc_emb_centroid": round(auroc_emb_all, 4),
            "auroc_major_glia": round(auroc_glia, 4),
            "auroc_microglia": round(auroc_mg, 4),
            "auroc_astrocyte": round(auroc_as, 4),
            "auroc_boot80": round(auroc_boot, 4),
            "auroc_sample_donor": round(auroc_sample, 4),
        })

        for t_id, t_val in [
            ("T1_AllNuclei", auroc_all),
            ("T2_Boot80", auroc_boot),
            ("T3_MajorGlia", auroc_glia),
            ("T4_Microglia", auroc_mg),
            ("T5_SampleDonor", auroc_sample),
        ]:
            traj_rows.append({
                "cohort": cohort_name,
                "method": m_name,
                "trajectory": t_id,
                "auroc": round(float(t_val), 4) if not np.isnan(t_val) else np.nan,
            })

bench_df = pd.DataFrame(benchmark_rows)
traj_df = pd.DataFrame(traj_rows)

# Compute 5-trajectory summary per (cohort, method)
cons_rows = []
for (coh, mth), gdf in traj_df.groupby(["cohort", "method"], sort=False):
    vals = gdf["auroc"].dropna().values
    cons_rows.append({
        "cohort": coh,
        "method": mth,
        "T1_AllNuclei": float(gdf.loc[gdf["trajectory"] == "T1_AllNuclei", "auroc"].values[0]),
        "T2_Boot80": float(gdf.loc[gdf["trajectory"] == "T2_Boot80", "auroc"].values[0]),
        "T3_MajorGlia": float(gdf.loc[gdf["trajectory"] == "T3_MajorGlia", "auroc"].values[0]),
        "T4_Microglia": float(gdf.loc[gdf["trajectory"] == "T4_Microglia", "auroc"].values[0]),
        "T5_SampleDonor": float(gdf.loc[gdf["trajectory"] == "T5_SampleDonor", "auroc"].values[0]),
        "mean_5traj_auroc": round(float(np.mean(vals)), 4),
        "min_5traj_auroc": round(float(np.min(vals)), 4),
        "max_5traj_auroc": round(float(np.max(vals)), 4),
    })
cons_df = pd.DataFrame(cons_rows)

bench_csv = os.path.join(OUT_DIR, "tier1_baseline_benchmark.csv")
cons_csv = os.path.join(OUT_DIR, "tier1_baseline_5traj_consensus.csv")
bench_df.to_csv(bench_csv, index=False)
cons_df.to_csv(cons_csv, index=False)

log(f"Saved {bench_csv} and {cons_csv}")
print("\n=== TIER-1 UPGRADE #1: CONVENTIONAL BASELINES VS TBI-GF SUMMARY ===")
print(cons_df.to_string(index=False))
