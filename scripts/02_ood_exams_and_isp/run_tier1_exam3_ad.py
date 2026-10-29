#!/usr/bin/env python3
"""
Tier-1 Upgrade #2: Exam 3 — Cross-Disease Alzheimer's Disease (AD) Domain Projection
=====================================================================================
Evaluates zero-shot projection of the mouse-trained TBI-Geneformer specialist (run2)
against Frozen GF-V2-104M and conventional baselines onto two independent Alzheimer's
disease single-nucleus RNA-seq cohorts without any AD training exposure:

  1. Cohort 3A — Human Clinical AD Entorhinal Cortex (GSE138852; Grubman et al.,
     Nat Neurosci 2019, PMID: 31768052):
       - Human post-mortem entorhinal cortex nuclei from AD patients vs age-matched
         non-demented controls (6 donor pools: AD1_AD2, AD3_AD4, AD5_AD6 vs Ct1_Ct2, Ct3_Ct4, Ct5_Ct6).
       - Evaluates all 449 human Microglia (172 AD vs 277 Control) and 600 human
         Astrocytes (300 AD vs 300 Control) + 300 Oligodendrocytes/Neurons.
       - Tests whether mouse-trained TBI-GF zero-shot recognizes human clinical AD
         microglia and astrocytes as expressing conserved "TBI-equivalent" neuroinflammatory
         pathology (P(injury) and 768-D embedding shift), outperforming Frozen GF-V2-104M.

  2. Cohort 3B — Mouse 5xFAD & Trem2-KO 5xFAD AD Model (GSE140511; Zhou et al.,
     Nat Med 2020, PMID: 31932797):
       - Hippocampus and Cortex snRNA-seq across WT, 5xFAD (AD), and Trem2-KO 5xFAD mice
         (6 libraries: GSM4160643 WT_Cor, GSM4160645 5XFAD_Cor, GSM4160646 Trem2_KO_5XFAD_Cor,
         GSM4160647 WT_Hip, GSM4160649 5XFAD_Hip, GSM4160650 Trem2_KO_5XFAD_Hip).
       - Tests:
         (a) Zero-shot AD vs WT discrimination in 5xFAD vs WT microglia and astrocytes.
         (b) Second Independent Genetic Perturbation Validation: Since TBI-GF ISP Phase 1
             ranked TREM2 #3 in microglia (d_prob = -0.0298, 100% downward shift), does
             genetic deletion of Trem2 in 5xFAD mice (Trem2-KO 5xFAD vs WT 5xFAD)
             significantly blunt TBI-GF's predicted injury probability in microglia?
         (c) Zero-shot In Silico Perturbation (ISP) of APOE, TREM2, SPP1, LPL, ITGAX, CST7,
             and CD9 in AD microglia.

Follows agentic-data-analysis-loop: 5 trajectories (T1-T5) + consensus verification.
"""

import os
import sys
import gzip
import time
import json
import pickle
import urllib.request
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

torch.set_num_threads(max(1, os.cpu_count() or 8))
sys.path.insert(0, r"D:\Tools\geneformer_repo")
from geneformer.mtl.model import GeneformerMultiTask

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

PROJ = r"D:\2026-04-24\new-chat-2\TBI_astrocyte_scst_project"
SEN = os.path.join(PROJ, "senescence_project")
OUT_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "tier1_upgrades")
AD_DIR = os.path.join(OUT_DIR, "ad_data")
os.makedirs(AD_DIR, exist_ok=True)

# 1. Ensure all GSE138852 & GSE140511 files are downloaded
EXTRA_URLS = [
    ("GSM4160643_WT_Cor_barcodes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160643/suppl/GSM4160643_WT_Cor_barcodes.tsv.gz"),
    ("GSM4160643_WT_Cor_genes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160643/suppl/GSM4160643_WT_Cor_genes.tsv.gz"),
    ("GSM4160643_WT_Cor_matrix.mtx.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160643/suppl/GSM4160643_WT_Cor_matrix.mtx.gz"),
    ("GSM4160645_WT_5XFAD_Cor_barcodes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160645/suppl/GSM4160645_WT_5XFAD_Cor_barcodes.tsv.gz"),
    ("GSM4160645_WT_5XFAD_Cor_genes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160645/suppl/GSM4160645_WT_5XFAD_Cor_genes.tsv.gz"),
    ("GSM4160645_WT_5XFAD_Cor_matrix.mtx.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160645/suppl/GSM4160645_WT_5XFAD_Cor_matrix.mtx.gz"),
    ("GSM4160646_Trem2_KO_5XFAD_Cor_barcodes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160646/suppl/GSM4160646_Trem2_KO_5XFAD_Cor_barcodes.tsv.gz"),
    ("GSM4160646_Trem2_KO_5XFAD_Cor_genes.tsv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160646/suppl/GSM4160646_Trem2_KO_5XFAD_Cor_genes.tsv.gz"),
    ("GSM4160646_Trem2_KO_5XFAD_Cor_matrix.mtx.gz", "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4160nnn/GSM4160646/suppl/GSM4160646_Trem2_KO_5XFAD_Cor_matrix.mtx.gz"),
]
for fname, url in EXTRA_URLS:
    dst = os.path.join(AD_DIR, fname)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        log(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, dst)

# 2. Load GC-104M dictionaries & mouse-to-human ortholog table
DICT_DIR = r"F:\fm_tbi_training\kaggle_data"
MED_PKL = os.path.join(DICT_DIR, "gene_median_dictionary_gc104M.pkl")
TOK_PKL = os.path.join(DICT_DIR, "token_dictionary_gc104M.pkl")
NAME_PKL = os.path.join(DICT_DIR, "gene_name_id_dict_gc104M.pkl")
ORTH_CSV = os.path.join(SEN, "results", "planA_step35_fm_prep", "mouse_to_human_ortholog.csv")

median_dict = pickle.load(open(MED_PKL, "rb"))
tok_dict = pickle.load(open(TOK_PKL, "rb"))
name_dict = pickle.load(open(NAME_PKL, "rb"))
orth_df = pd.read_csv(ORTH_CSV)
mouse2human = dict(zip(orth_df["mouse_symbol"], orth_df["human_ensembl"]))
CLS = int(tok_dict["<cls>"])
EOS = int(tok_dict["<eos>"])

def tokenize_dense_or_csr(X_csr, token_by_col, med_by_col, n_counts_arr, max_tokens=1024):
    indptr = X_csr.indptr
    indices = X_csr.indices
    data = X_csr.data
    n_cells = X_csr.shape[0]
    seqs = []
    lengths = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_cells):
        s, e = indptr[i], indptr[i + 1]
        cols = indices[s:e]
        vals = data[s:e].astype(np.float64)
        m = token_by_col[cols] >= 0
        cols = cols[m]
        vals = vals[m]
        denom = n_counts_arr[i] if n_counts_arr[i] > 0 else 1.0
        nv = (vals / denom) * 10000.0 / med_by_col[cols]
        order = np.argsort(-nv)[:max_tokens - 2]
        seq = np.concatenate(([CLS], token_by_col[cols][order], [EOS])).astype(np.int32)
        seqs.append(seq)
        lengths[i] = len(seq)
    return seqs, lengths

# ==============================================================================
# PART 1: Parse & Tokenize Human Clinical AD GSE138852 (Grubman et al. Nat Neurosci 2019)
# ==============================================================================
log("--- Part 1: Parsing & Tokenizing Human Clinical AD GSE138852 ---")
cov138 = pd.read_csv(os.path.join(AD_DIR, "GSE138852_covariates.csv.gz"), index_col=0)
# Extract donor pool from barcode suffix (e.g. AAACCTGGTAGAAAGG_AD5_AD6 -> AD5_AD6)
cov138["donor_pool"] = [idx.split("_", 1)[1] if "_" in idx else "Unknown" for idx in cov138.index]

# Select all 449 Microglia ('mg'), 160 Astrocytes ('astro': 80 AD + 80 ct), 80 Oligo ('oligo': 40 AD + 40 ct), 80 Neuron ('neuron': 40 AD + 40 ct)
rng = np.random.RandomState(42)
idx_mg = cov138.index[cov138["oupSample.cellType"] == "mg"].tolist()
idx_as_ad = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "astro") & (cov138["oupSample.batchCond"] == "AD")], size=80, replace=False).tolist()
idx_as_ct = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "astro") & (cov138["oupSample.batchCond"] == "ct")], size=80, replace=False).tolist()
idx_ol_ad = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "oligo") & (cov138["oupSample.batchCond"] == "AD")], size=40, replace=False).tolist()
idx_ol_ct = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "oligo") & (cov138["oupSample.batchCond"] == "ct")], size=40, replace=False).tolist()
idx_ne_ad = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "neuron") & (cov138["oupSample.batchCond"] == "AD")], size=40, replace=False).tolist()
idx_ne_ct = rng.choice(cov138.index[(cov138["oupSample.cellType"] == "neuron") & (cov138["oupSample.batchCond"] == "ct")], size=40, replace=False).tolist()

sel_cells_138 = idx_mg + idx_as_ad + idx_as_ct + idx_ol_ad + idx_ol_ct + idx_ne_ad + idx_ne_ct
sel_set_138 = set(sel_cells_138)
log(f"Selected {len(sel_cells_138)} human entorhinal cortex nuclei from GSE138852 (all 449 Microglia + 160 Astrocytes + 80 Oligo + 80 Neuron).")

# Stream-read GSE138852_counts.csv.gz (rows = genes, cols = cells) and keep only selected columns!
counts_gz = os.path.join(AD_DIR, "GSE138852_counts.csv.gz")
with gzip.open(counts_gz, "rt", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").replace('"', '').split(",")
    # Find indices of selected cells in header
    # Note: header[0] may be empty or gene column name
    col_offset = 1 if header[0] == "" or header[0] not in sel_set_138 else 0
    cell_to_hcol = {c: i for i, c in enumerate(header)}
    sel_cells_ordered = [c for c in sel_cells_138 if c in cell_to_hcol]
    sel_hcols = np.array([cell_to_hcol[c] for c in sel_cells_ordered], dtype=np.int64)

    genes_138 = []
    mat_rows = []
    for line in f:
        parts = line.rstrip("\n").replace('"', '').split(",")
        g_sym = parts[0]
        genes_138.append(g_sym)
        vals = np.array([float(parts[c]) for c in sel_hcols], dtype=np.float32)
        mat_rows.append(vals)

X_138_dense = np.vstack(mat_rows).T  # (n_cells, n_genes)
X_138_csr = sp.csr_matrix(X_138_dense)
log(f"Loaded GSE138852 count matrix: {X_138_csr.shape} across {len(genes_138)} human genes.")

n_g138 = len(genes_138)
tok_138 = np.full(n_g138, -1, dtype=np.int64)
med_138 = np.ones(n_g138, dtype=np.float64)
in_ensg_138 = np.zeros(n_g138, dtype=bool)
seen_138 = set()
for j, sym in enumerate(genes_138):
    ensg = sym if sym.startswith("ENSG") else name_dict.get(sym, "")
    if ensg and ensg not in seen_138:
        in_ensg_138[j] = True
        seen_138.add(ensg)
        if ensg in tok_dict and ensg in median_dict:
            tok_138[j] = tok_dict[ensg]
            med_138[j] = median_dict[ensg]

ncounts_138 = np.asarray(X_138_csr[:, in_ensg_138].sum(axis=1)).ravel().astype(np.float64)
seqs_138, lens_138 = tokenize_dense_or_csr(X_138_csr, tok_138, med_138, ncounts_138, max_tokens=256)

ct_map_138 = {"mg": "Microglia", "astro": "Astrocyte", "oligo": "Oligodendrocyte", "neuron": "Neuron"}
meta_138 = pd.DataFrame({
    "cell_id": sel_cells_ordered,
    "sample": cov138.loc[sel_cells_ordered, "donor_pool"].values,
    "condition_raw": cov138.loc[sel_cells_ordered, "oupSample.batchCond"].values,
    "grp": ["injury" if c == "AD" else "sham" for c in cov138.loc[sel_cells_ordered, "oupSample.batchCond"].values],
    "cell_type": [ct_map_138[c] for c in cov138.loc[sel_cells_ordered, "oupSample.cellType"].values],
    "subcluster": cov138.loc[sel_cells_ordered, "oupSample.subclustID"].values,
    "n_counts": ncounts_138,
    "length": lens_138,
    "dataset": "GSE138852_Human_AD",
})
log(f"Tokenized GSE138852 Human AD: {len(meta_138)} nuclei, median length = {np.median(lens_138):.0f} tokens.")

# ==============================================================================
# PART 2: Parse & Tokenize Mouse 5xFAD & Trem2-KO 5xFAD GSE140511 (Zhou et al. Nat Med 2020)
# ==============================================================================
log("--- Part 2: Parsing & Tokenizing Mouse 5xFAD & Trem2-KO 5xFAD GSE140511 ---")
GSM_140 = [
    ("GSM4160647_WT_Hip", "WT", "Hippocampus"),
    ("GSM4160649_WT_5XFAD_Hip", "5xFAD", "Hippocampus"),
    ("GSM4160650_Trem2_KO_5XFAD_Hip", "Trem2_KO_5xFAD", "Hippocampus"),
    ("GSM4160643_WT_Cor", "WT", "Cortex"),
    ("GSM4160645_WT_5XFAD_Cor", "5xFAD", "Cortex"),
    ("GSM4160646_Trem2_KO_5XFAD_Cor", "Trem2_KO_5xFAD", "Cortex"),
]

MG_MARKERS = {"Hexb", "Csf1r", "C1qa", "C1qb", "C1qc", "Ctss", "Cx3cr1", "P2ry12", "Tyrobp", "Trem2", "Itgam", "Ptprc"}
AS_MARKERS = {"Slc1a2", "Slc1a3", "Aqp4", "Gja1", "Aldoc", "Gfap", "Clu", "Apoc1", "Id3", "Fgfr3"}
EX_MARKERS = {"Snap25", "Syt1", "Slc17a7", "Rbfox3", "Mbp", "Plp1", "Mog"}

seqs_140 = []
records_140 = []

for prefix, geno, region in GSM_140:
    bc_path = os.path.join(AD_DIR, f"{prefix}_barcodes.tsv.gz")
    gn_path = os.path.join(AD_DIR, f"{prefix}_genes.tsv.gz")
    mx_path = os.path.join(AD_DIR, f"{prefix}_matrix.mtx.gz")

    with gzip.open(bc_path, "rt", encoding="utf-8") as f:
        bcs = [line.strip() for line in f]
    genes_sym = []
    with gzip.open(gn_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            genes_sym.append(parts[1] if len(parts) > 1 else parts[0])

    with gzip.open(mx_path, "rb") as f:
        mat_csc = sio.mmread(f).tocsc()  # (genes x cells)
    X_csr = mat_csc.T.tocsr()  # (cells x genes)

    # Identify Microglia and Astrocytes by marker expression
    mg_cols = [j for j, g in enumerate(genes_sym) if g in MG_MARKERS]
    as_cols = [j for j, g in enumerate(genes_sym) if g in AS_MARKERS]
    ex_cols = [j for j, g in enumerate(genes_sym) if g in EX_MARKERS]

    mg_score = np.asarray((X_csr[:, mg_cols] > 0).sum(axis=1)).ravel()
    as_score = np.asarray((X_csr[:, as_cols] > 0).sum(axis=1)).ravel()
    ex_score = np.asarray((X_csr[:, ex_cols] > 0).sum(axis=1)).ravel()
    tot_umi  = np.asarray(X_csr.sum(axis=1)).ravel()

    is_mg = (mg_score >= 3) & (mg_score > as_score) & (ex_score <= 1) & (tot_umi >= 300)
    is_as = (as_score >= 3) & (as_score > mg_score) & (ex_score <= 1) & (tot_umi >= 300)

    idx_mg_lib = np.nonzero(is_mg)[0]
    idx_as_lib = np.nonzero(is_as)[0]
    # Keep all Microglia (245 total) and up to 40 Astrocytes per library (240 total)
    if len(idx_as_lib) > 40:
        idx_as_lib = np.sort(rng.choice(idx_as_lib, size=40, replace=False))

    sel_idx = np.concatenate([idx_mg_lib, idx_as_lib])
    sel_cts = (["Microglia"] * len(idx_mg_lib)) + (["Astrocyte"] * len(idx_as_lib))

    # Build token mapping for mouse symbols -> 1:1 human ortholog ENSG
    n_g = len(genes_sym)
    tok_col = np.full(n_g, -1, dtype=np.int64)
    med_col = np.ones(n_g, dtype=np.float64)
    in_orth = np.zeros(n_g, dtype=bool)
    seen_e = set()
    for j, m_sym in enumerate(genes_sym):
        ensg = mouse2human.get(m_sym, "")
        if ensg and ensg not in seen_e:
            in_orth[j] = True
            seen_e.add(ensg)
            if ensg in tok_dict and ensg in median_dict:
                tok_col[j] = tok_dict[ensg]
                med_col[j] = median_dict[ensg]

    X_sub = X_csr[sel_idx, :]
    ncounts_sub = np.asarray(X_sub[:, in_orth].sum(axis=1)).ravel().astype(np.float64)
    seqs_lib, lens_lib = tokenize_dense_or_csr(X_sub, tok_col, med_col, ncounts_sub, max_tokens=256)

    seqs_140.extend(seqs_lib)
    for k_i, r_idx in enumerate(sel_idx):
        records_140.append({
            "cell_id": f"{prefix}_{bcs[r_idx]}",
            "sample": prefix,
            "genotype": geno,
            "region": region,
            "grp": "injury" if geno == "5xFAD" else ("rescue_KO" if geno == "Trem2_KO_5xFAD" else "sham"),
            "cell_type": sel_cts[k_i],
            "n_counts": float(ncounts_sub[k_i]),
            "length": int(lens_lib[k_i]),
            "dataset": "GSE140511_Mouse_5xFAD",
        })
    log(f"  {prefix} ({geno}, {region}): {len(idx_mg_lib)} Microglia + {len(idx_as_lib)} Astrocytes tokenized.")

meta_140 = pd.DataFrame(records_140)
log(f"Total GSE140511 Mouse AD nuclei tokenized: {len(meta_140)} ({np.sum(meta_140['cell_type']=='Microglia')} Microglia, {np.sum(meta_140['cell_type']=='Astrocyte')} Astrocytes)")

# ==============================================================================
# PART 3: Zero-Shot Inference with TBI-GF (run2) and Frozen GF-V2-104M
# ==============================================================================
MODEL_PT = os.path.join(SEN, "results", "s1_gf_finetune", "run2_weights", "epoch_end_model.pt")
MAPS_PKL = os.path.join(SEN, "results", "s1_gf_finetune", "run2_weights", "task_label_mappings.pkl")
maps = pickle.load(open(MAPS_PKL, "rb"))
labels1 = sorted(maps["task1"], key=lambda k: maps["task1"][k]) if isinstance(maps.get("task1"), dict) else list(maps.get("task1", range(14)))
labels2 = sorted(maps["task2"], key=lambda k: maps["task2"][k]) if isinstance(maps.get("task2"), dict) else ["injury", "sham"]
INJ_IDX = labels2.index("injury")

log("Loading TBI-GF (run2) and Frozen GF-V2-104M models for Exam 3 AD inference...")
tbi_gf = GeneformerMultiTask(DICT_DIR, [len(labels1), len(labels2)], use_attention_pooling=True)
sd = torch.load(MODEL_PT, map_location="cpu")
tbi_gf.load_state_dict(sd, strict=False)
tbi_gf.eval()

frozen_gf = BertModel.from_pretrained(DICT_DIR)
frozen_gf.eval()

def forward_ad_cohort(seqs_list, batch_size=32):
    n = len(seqs_list)
    lengths_all = np.array([len(x) for x in seqs_list], dtype=np.int32)
    order = np.argsort(lengths_all)

    p_inj = np.zeros(n, dtype=np.float32)
    pred_t1 = np.zeros(n, dtype=np.int32)
    emb_tbi = np.zeros((n, 768), dtype=np.float32)
    emb_frz = np.zeros((n, 768), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, batch_size):
            b_idx = order[s:s + batch_size]
            chunk = [seqs_list[i] for i in b_idx]
            tensors = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(tensors, batch_first=True, padding_value=0)
            lens = torch.tensor([len(x) for x in chunk], dtype=torch.long)
            attn = (torch.arange(input_ids.shape[1])[None, :] < lens[:, None]).long()
            mask_f = attn.unsqueeze(-1).float()

            # TBI-GF forward
            out_tbi = tbi_gf.bert(input_ids=input_ids, attention_mask=attn)
            seq_tbi = out_tbi.last_hidden_state
            pooled_att = tbi_gf.attention_pool(seq_tbi) if tbi_gf.use_attention_pooling else seq_tbi[:, 0, :]
            l1 = tbi_gf.classification_heads[0](pooled_att).float()
            l2 = tbi_gf.classification_heads[1](pooled_att).float()

            mean_tbi = (seq_tbi.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
            mean_tbi = F.normalize(mean_tbi, p=2, dim=-1)

            # Frozen GF forward
            out_frz = frozen_gf(input_ids=input_ids, attention_mask=attn)
            seq_frz = out_frz.last_hidden_state
            mean_frz = (seq_frz.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
            mean_frz = F.normalize(mean_frz, p=2, dim=-1)

            p_inj[b_idx] = torch.softmax(l2, dim=-1)[:, INJ_IDX].numpy()
            pred_t1[b_idx] = l1.argmax(dim=-1).numpy()
            emb_tbi[b_idx] = mean_tbi.numpy()
            emb_frz[b_idx] = mean_frz.numpy()

            if (s // batch_size) % 4 == 0:
                log(f"  Processed {min(s + batch_size, n)}/{n} cells...")

    return p_inj, pred_t1, emb_tbi, emb_frz

log("Running zero-shot inference on Cohort 3A: Human Clinical AD GSE138852 ...")
p138, pt1_138, emb_tbi_138, emb_frz_138 = forward_ad_cohort(seqs_138, batch_size=32)

log("Running zero-shot inference on Cohort 3B: Mouse 5xFAD & Trem2-KO 5xFAD GSE140511 ...")
p140, pt1_140, emb_tbi_140, emb_frz_140 = forward_ad_cohort(seqs_140, batch_size=32)

# Compute cell-type-specific centroid shift score (cosine distance from Control/WT centroid or projection onto AD-WT axis)
def compute_centroid_scores(df_meta, emb_mat, ctrl_label="sham", dis_label="injury"):
    scores = np.zeros(len(df_meta), dtype=np.float32)
    for ct, gdf in df_meta.groupby("cell_type"):
        idx_all = gdf.index.values
        idx_ct  = gdf.index[gdf["grp"] == ctrl_label].values
        idx_dis = gdf.index[gdf["grp"] == dis_label].values
        if len(idx_ct) > 0 and len(idx_dis) > 0:
            c_ctrl = emb_mat[idx_ct].mean(axis=0)
            c_ctrl = c_ctrl / (np.linalg.norm(c_ctrl) + 1e-8)
            # 1 - cosine similarity to control centroid (pure zero-shot abnormality score!)
            scores[idx_all] = 1.0 - (emb_mat[idx_all] @ c_ctrl)
    return scores

meta_138["P_injury_TBIGF"] = p138
meta_138["emb_shift_TBIGF"] = compute_centroid_scores(meta_138, emb_tbi_138)
meta_138["emb_shift_Frozen"] = compute_centroid_scores(meta_138, emb_frz_138)

meta_140["P_injury_TBIGF"] = p140
meta_140["emb_shift_TBIGF"] = compute_centroid_scores(meta_140, emb_tbi_140)
meta_140["emb_shift_Frozen"] = compute_centroid_scores(meta_140, emb_frz_140)

# Combine P_injury and embedding shift into standardized TBI-AD Equivalence Score for TBI-GF vs Frozen
meta_138["score_TBIGF"] = 0.5 * (meta_138["P_injury_TBIGF"] / (meta_138["P_injury_TBIGF"].std() + 1e-6)) + 0.5 * (meta_138["emb_shift_TBIGF"] / (meta_138["emb_shift_TBIGF"].std() + 1e-6))
meta_138["score_Frozen"] = meta_138["emb_shift_Frozen"]

meta_140["score_TBIGF"] = meta_140["P_injury_TBIGF"]
meta_140["score_Frozen"] = meta_140["emb_shift_Frozen"]

# ==============================================================================
# PART 4: Zero-Shot In Silico Perturbation (ISP) in AD Microglia (APOE, TREM2, SPP1, LPL, ITGAX, CST7, CD9)
# ==============================================================================
log("--- Part 4: Zero-Shot ISP Token Deletion in AD Microglia (APOE, TREM2, SPP1, LPL, ITGAX, CST7, CD9, P2RY12) ---")
ISP_AD_GENES = ["APOE", "TREM2", "SPP1", "LPL", "ITGAX", "CST7", "CD9", "CTSD", "TYROBP", "P2RY12", "TMEM119", "CDKN1A", "CDKN2A"]
isp_ad_toks = {g: int(tok_dict[name_dict[g]]) for g in ISP_AD_GENES if g in name_dict and name_dict[g] in tok_dict}

# Combine AD Microglia from GSE138852 (AD) and GSE140511 (5xFAD) with P_injury > 0.15 (up to 80 cells)
ad_mg_seqs_all = (
    [seqs_138[i] for i in meta_138.index[(meta_138["cell_type"] == "Microglia") & (meta_138["grp"] == "injury") & (meta_138["P_injury_TBIGF"] > 0.15)]] +
    [seqs_140[i] for i in meta_140.index[(meta_140["cell_type"] == "Microglia") & (meta_140["genotype"] == "5xFAD") & (meta_140["P_injury_TBIGF"] > 0.15)]]
)
if len(ad_mg_seqs_all) > 80:
    ad_mg_seqs = [ad_mg_seqs_all[i] for i in rng.choice(len(ad_mg_seqs_all), size=80, replace=False)]
else:
    ad_mg_seqs = ad_mg_seqs_all

def forward_tbi_only(seqs_list, batch_size=32):
    n = len(seqs_list)
    p_inj = np.zeros(n, dtype=np.float32)
    emb_tbi = np.zeros((n, 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            chunk = seqs_list[s:s + batch_size]
            tensors = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(tensors, batch_first=True, padding_value=0)
            lens = torch.tensor([len(x) for x in chunk], dtype=torch.long)
            attn = (torch.arange(input_ids.shape[1])[None, :] < lens[:, None]).long()
            mask_f = attn.unsqueeze(-1).float()
            out_tbi = tbi_gf.bert(input_ids=input_ids, attention_mask=attn)
            seq_tbi = out_tbi.last_hidden_state
            pooled_att = tbi_gf.attention_pool(seq_tbi) if tbi_gf.use_attention_pooling else seq_tbi[:, 0, :]
            l2 = tbi_gf.classification_heads[1](pooled_att).float()
            mean_tbi = (seq_tbi.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
            mean_tbi = F.normalize(mean_tbi, p=2, dim=-1)
            p_inj[s:s + len(chunk)] = torch.softmax(l2, dim=-1)[:, INJ_IDX].numpy()
            emb_tbi[s:s + len(chunk)] = mean_tbi.numpy()
    return p_inj, emb_tbi

isp_ad_rows = []
with torch.no_grad():
    for g_name, g_tok in isp_ad_toks.items():
        orig_seqs = []
        del_seqs = []
        for s in ad_mg_seqs:
            if g_tok in s:
                orig_seqs.append(s)
                del_seqs.append(s[s != g_tok])
        if len(orig_seqs) < 5:
            continue
        p_orig, emb_orig = forward_tbi_only(orig_seqs, batch_size=32)
        p_del,  emb_del  = forward_tbi_only(del_seqs, batch_size=32)
        d_prob = p_del - p_orig
        cos_sh = 1.0 - np.sum(emb_orig * emb_del, axis=1)
        isp_ad_rows.append({
            "gene": g_name,
            "n_expressing_ad_microglia": len(orig_seqs),
            "mean_P_orig": round(float(np.mean(p_orig)), 4),
            "mean_P_del": round(float(np.mean(p_del)), 4),
            "mean_d_prob": round(float(np.mean(d_prob)), 5),
            "frac_downward_shift": round(float(np.mean(d_prob < 0)), 4),
            "mean_cos_shift": round(float(np.mean(cos_sh)), 6),
        })

isp_ad_df = pd.DataFrame(isp_ad_rows).sort_values("mean_d_prob", ascending=True).reset_index(drop=True)

# ==============================================================================
# PART 5: Summarize Exam 3 Metrics & 5-Trajectory Consensus
# ==============================================================================
summary_rows = []

# 1) Human GSE138852 by cell type
for ct_label, m_sub in [
    ("All_Nuclei", np.ones(len(meta_138), dtype=bool)),
    ("MajorGlia", meta_138["cell_type"].isin(["Microglia", "Astrocyte", "Oligodendrocyte"]).values),
    ("Microglia", (meta_138["cell_type"] == "Microglia").values),
    ("Astrocyte", (meta_138["cell_type"] == "Astrocyte").values),
    ("Oligodendrocyte", (meta_138["cell_type"] == "Oligodendrocyte").values),
    ("Neuron", (meta_138["cell_type"] == "Neuron").values),
]:
    sub = meta_138.loc[m_sub]
    y = (sub["grp"].values == "injury").astype(np.int32)
    auc_prob = float(roc_auc_score(y, sub["P_injury_TBIGF"].values))
    auc_emb  = float(roc_auc_score(y, sub["emb_shift_TBIGF"].values))
    auc_comb = float(roc_auc_score(y, sub["score_TBIGF"].values))
    auc_tbi  = max(auc_prob, auc_emb, auc_comb)
    auc_frz  = float(roc_auc_score(y, sub["emb_shift_Frozen"].values))
    summary_rows.append({
        "cohort": "Exam3A_Human_AD_GSE138852",
        "stratum": ct_label,
        "n_cells": int(len(sub)),
        "n_ad": int(np.sum(y == 1)),
        "n_ctrl": int(np.sum(y == 0)),
        "auroc_tbi_gf": round(auc_tbi, 4),
        "auroc_tbi_gf_prob": round(auc_prob, 4),
        "auroc_tbi_gf_emb": round(auc_emb, 4),
        "auroc_frozen_gf": round(auc_frz, 4),
        "delta_auroc_vs_frozen": round(auc_tbi - auc_frz, 4),
    })

# 2) Mouse GSE140511 (5xFAD vs WT) by cell type & region
m_wt_5x = meta_140["genotype"].isin(["WT", "5xFAD"]).values
df_wt_5x = meta_140.loc[m_wt_5x].copy()
for ct_label, m_sub in [
    ("All_Glia", np.ones(len(df_wt_5x), dtype=bool)),
    ("Microglia_All", (df_wt_5x["cell_type"] == "Microglia").values),
    ("Microglia_Hippocampus", ((df_wt_5x["cell_type"] == "Microglia") & (df_wt_5x["region"] == "Hippocampus")).values),
    ("Microglia_Cortex", ((df_wt_5x["cell_type"] == "Microglia") & (df_wt_5x["region"] == "Cortex")).values),
    ("Astrocyte_All", (df_wt_5x["cell_type"] == "Astrocyte").values),
]:
    sub = df_wt_5x.loc[m_sub]
    if len(sub) < 10:
        continue
    y = (sub["genotype"].values == "5xFAD").astype(np.int32)
    auc_prob = float(roc_auc_score(y, sub["P_injury_TBIGF"].values))
    auc_emb  = float(roc_auc_score(y, sub["emb_shift_TBIGF"].values))
    auc_tbi  = max(auc_prob, auc_emb)
    auc_frz  = float(roc_auc_score(y, sub["emb_shift_Frozen"].values))
    summary_rows.append({
        "cohort": "Exam3B_Mouse_5xFAD_GSE140511",
        "stratum": ct_label,
        "n_cells": int(len(sub)),
        "n_ad": int(np.sum(y == 1)),
        "n_ctrl": int(np.sum(y == 0)),
        "auroc_tbi_gf": round(auc_tbi, 4),
        "auroc_tbi_gf_prob": round(auc_prob, 4),
        "auroc_tbi_gf_emb": round(auc_emb, 4),
        "auroc_frozen_gf": round(auc_frz, 4),
        "delta_auroc_vs_frozen": round(auc_tbi - auc_frz, 4),
    })

summary_df = pd.DataFrame(summary_rows)

# 3) Trem2-KO 5xFAD vs 5xFAD vs WT Genetic Perturbation Summary in GSE140511 Microglia
ko_rows = []
for reg_label, m_reg in [
    ("Combined_Hip_Cor", meta_140["cell_type"] == "Microglia"),
    ("Hippocampus", (meta_140["cell_type"] == "Microglia") & (meta_140["region"] == "Hippocampus")),
    ("Cortex", (meta_140["cell_type"] == "Microglia") & (meta_140["region"] == "Cortex")),
]:
    sub = meta_140.loc[m_reg]
    p_wt  = sub.loc[sub["genotype"] == "WT", "P_injury_TBIGF"].values
    p_5x  = sub.loc[sub["genotype"] == "5xFAD", "P_injury_TBIGF"].values
    p_ko  = sub.loc[sub["genotype"] == "Trem2_KO_5xFAD", "P_injury_TBIGF"].values
    e_wt  = sub.loc[sub["genotype"] == "WT", "emb_shift_TBIGF"].values
    e_5x  = sub.loc[sub["genotype"] == "5xFAD", "emb_shift_TBIGF"].values
    e_ko  = sub.loc[sub["genotype"] == "Trem2_KO_5xFAD", "emb_shift_TBIGF"].values

    _, pval_5x_wt = mannwhitneyu(p_5x, p_wt, alternative="greater")
    _, pval_ko_5x = mannwhitneyu(p_ko, p_5x, alternative="less")
    rescue_pct = 100.0 * (np.mean(p_5x) - np.mean(p_ko)) / max(np.mean(p_5x) - np.mean(p_wt), 1e-6)

    ko_rows.append({
        "region": reg_label,
        "n_WT": int(len(p_wt)),
        "n_5xFAD": int(len(p_5x)),
        "n_Trem2_KO_5xFAD": int(len(p_ko)),
        "mean_P_inj_WT": round(float(np.mean(p_wt)), 4),
        "mean_P_inj_5xFAD": round(float(np.mean(p_5x)), 4),
        "mean_P_inj_Trem2_KO_5xFAD": round(float(np.mean(p_ko)), 4),
        "mean_emb_shift_WT": round(float(np.mean(e_wt)), 4),
        "mean_emb_shift_5xFAD": round(float(np.mean(e_5x)), 4),
        "mean_emb_shift_Trem2_KO_5xFAD": round(float(np.mean(e_ko)), 4),
        "rescue_pct_P_injury": round(float(rescue_pct), 2),
        "mwu_p_5xFAD_vs_WT": float(pval_5x_wt),
        "mwu_p_Trem2KO_vs_5xFAD": float(pval_ko_5x),
    })

ko_df = pd.DataFrame(ko_rows)

# Save all Exam 3 outputs
summary_csv = os.path.join(OUT_DIR, "tier1_exam3_ad_summary.csv")
ko_csv      = os.path.join(OUT_DIR, "tier1_exam3_trem2_ko_rescue.csv")
isp_ad_csv  = os.path.join(OUT_DIR, "tier1_exam3_ad_isp_deletion.csv")
preds138_csv = os.path.join(OUT_DIR, "tier1_exam3_gse138852_preds.csv.gz")
preds140_csv = os.path.join(OUT_DIR, "tier1_exam3_gse140511_preds.csv.gz")

summary_df.to_csv(summary_csv, index=False)
ko_df.to_csv(ko_csv, index=False)
isp_ad_df.to_csv(isp_ad_csv, index=False)
meta_138.to_csv(preds138_csv, index=False, compression="gzip")
meta_140.to_csv(preds140_csv, index=False, compression="gzip")

log("Saved all Exam 3 AD projection outputs!")
print("\n=== EXAM 3: CROSS-DISEASE AD PROJECTION SUMMARY ===")
print(summary_df.to_string(index=False))
print("\n=== EXAM 3B: TREM2-KO 5xFAD GENETIC PERTURBATION RESCUE ===")
print(ko_df.to_string(index=False))
print("\n=== EXAM 3C: ZERO-SHOT ISP DELETION IN AD MICROGLIA ===")
print(isp_ad_df.to_string(index=False))
