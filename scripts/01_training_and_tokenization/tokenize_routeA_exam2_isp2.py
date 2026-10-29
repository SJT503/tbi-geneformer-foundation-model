#!/usr/bin/env python3
"""
Route A Local Tokenization Script (Exam 2 GSE209552 Human + ISP Phase 2 GSE333879 Mouse D30)
=============================================================================================
Strictly mirrors the verified GC-104M rank-value tokenization from F:/fm_tbi_training/kaggle_tokenize/kernel.py:
  - Normalizes per-cell counts by total mapped counts (or total counts, matching planA_35c2):
    Wait: in planA_35c2_merge_h5ad.py line 67:
      n_counts = X_full.sum(axis=1) (sum of counts over mapped human Ensembl genes!)
      nv = (vals / n_counts[i]) * 10000.0 / med_by_col[cols]
      order = np.argsort(-nv)[:4094]
      seq = [CLS] + token_by_col[cols][order] + [EOS]
  - We verify and record both total-mapped-count normalization (exact match to run2 training)
    and token statistics for:
    1) Exam 2: Human acute TBI GSE209552 (17 samples: 5 Control + 12 TBI, 24,241 QC-passed nuclei)
    2) ISP Phase 2: In-house mouse D30 snRNA-seq GSE333879 (3 libraries: Sham 11,378 + CCI_Scr 6,845 + CCI_TGFbetaR1_KO 16,824 = 35,047 nuclei)
"""

import os
import sys
import time
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

PROJ = r"D:\2026-04-24\new-chat-2\TBI_astrocyte_scst_project"
SEN = os.path.join(PROJ, "senescence_project")
OUT_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "routeA_tokenized")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load GC-104M dictionaries & mouse-to-human ortholog table
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

CLS = tok_dict["<cls>"]
EOS = tok_dict["<eos>"]
log(f"Loaded GC-104M dicts: tok_dict={len(tok_dict)}, median_dict={len(median_dict)}, mouse2human={len(mouse2human)}, CLS={CLS}, EOS={EOS}")

def tokenize_csr_cell_by_gene(X_csr, token_by_col, med_by_col, n_counts_arr):
    """
    Exact match to F:/fm_tbi_training/kaggle_tokenize/kernel.py tok_range():
      X_csr: (n_cells, n_genes) CSR matrix
      token_by_col: (n_genes,) int64 (-1 if not in GC-104M tok_dict/median_dict)
      med_by_col: (n_genes,) float64
      n_counts_arr: (n_cells,) float64 (sum of X_csr row counts, matching planA_35c2_merge_h5ad.py line 67)
    """
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
        order = np.argsort(-nv)[:4094]
        seq = np.concatenate(([CLS], token_by_col[cols][order], [EOS])).astype(np.int32)
        seqs.append(seq)
        lengths[i] = len(seq)
    return seqs, lengths

# ==============================================================================
# PART 1: Tokenize Human Acute TBI GSE209552 (17 samples, 24,241 QC-passed cells)
# ==============================================================================
log("--- Part 1: Tokenizing Human GSE209552 (Exam 2 Cross-Species Migration) ---")
meta209_path = os.path.join(PROJ, "results", "tables", "gse209552_celltype_transfer_metadata.csv")
meta209 = pd.read_csv(meta209_path)
root209 = os.path.join(PROJ, "raw_data", "GSE209552", "extracted_unzipped")

# Map predicted_celltype to run2 cell_type namespace
ct_map_209 = {
    "microglia_macrophage": "Microglia",
    "astrocyte": "Astrocyte",
    "oligodendrocyte": "Oligodendrocyte",
    "opc": "OPC",
    "neuron": "Neuron",
    "endothelial": "Endothelial",
}
grp_map_209 = {
    "tbi": "injury",
    "control": "sham",
}

gse209_records = []
gse209_seqs = []
sample_stats_209 = []

for sid, sdf in meta209.groupby("sample_id", sort=False):
    sdir = os.path.join(root209, sid)
    assert os.path.isdir(sdir), f"Missing sample dir: {sdir}"
    # Read features.tsv (col 0 = ENSG, col 1 = symbol)
    feats = []
    with open(os.path.join(sdir, "features.tsv"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ensg = parts[0].split(".")[0]
            sym = parts[1] if len(parts) > 1 else ""
            # Direct ENSG match first; fallback to gene symbol -> ENSG via name_dict
            if ensg not in tok_dict and sym in name_dict:
                ensg = name_dict[sym]
            feats.append(ensg)
    with open(os.path.join(sdir, "barcodes.tsv"), "r", encoding="utf-8") as f:
        bcs = [line.rstrip("\n") for line in f]
    bc_to_col = {bc: idx for idx, bc in enumerate(bcs)}

    # Build token_by_col and med_by_col for this sample's gene list
    # Wait: to match planA_35c2 exactly, let's check if duplicate ENSG IDs exist and deduplicate or map directly
    n_genes = len(feats)
    token_by_col = np.full(n_genes, -1, dtype=np.int64)
    med_by_col = np.ones(n_genes, dtype=np.float64)
    in_ensg = np.zeros(n_genes, dtype=bool)
    seen_ensg = set()
    for j, e in enumerate(feats):
        if e.startswith("ENSG") and e not in seen_ensg:
            in_ensg[j] = True
            seen_ensg.add(e)
            if e in tok_dict and e in median_dict:
                token_by_col[j] = tok_dict[e]
                med_by_col[j] = median_dict[e]

    # Load matrix.mtx (genes x barcodes) -> transpose to (barcodes x genes) CSR
    mat_coo = sio.mmread(os.path.join(sdir, "matrix.mtx")).tocsc()  # genes x cells
    # Select QC-passed barcodes in the exact order of sdf
    gsm = sdf["gsm"].iloc[0]
    prefix = f"{gsm}_{sid}_"
    sel_cols = []
    for cell_str in sdf["cell"]:
        assert cell_str.startswith(prefix), f"Unexpected cell prefix: {cell_str} vs {prefix}"
        bc = cell_str[len(prefix):]
        assert bc in bc_to_col, f"Barcode {bc} not found in {sid}"
        sel_cols.append(bc_to_col[bc])

    X_sub = mat_coo[:, sel_cols].T.tocsr()  # (n_qc_cells, n_genes)
    # n_counts = sum of mapped/unique ENSG counts per cell
    n_counts_arr = np.asarray(X_sub[:, in_ensg].sum(axis=1)).ravel().astype(np.float64)
    seqs, lengths = tokenize_csr_cell_by_gene(X_sub, token_by_col, med_by_col, n_counts_arr)

    gse209_seqs.extend(seqs)
    for idx_row, (_, row) in enumerate(sdf.iterrows()):
        gse209_records.append({
            "cell_id": row["cell"],
            "sample": sid,
            "gsm": row["gsm"],
            "condition_raw": row["condition"],
            "grp": grp_map_209[row["condition"]],
            "predicted_celltype_raw": row["predicted_celltype"],
            "cell_type": ct_map_209[row["predicted_celltype"]],
            "prediction_score_max": float(row["prediction.score.max"]),
            "brain_region": row["brain_region"],
            "n_counts": float(n_counts_arr[idx_row]),
            "length": int(lengths[idx_row]),
            "dataset": "GSE209552",
            "time_point": "acute",
        })
    sample_stats_209.append({
        "sample_id": sid,
        "gsm": gsm,
        "condition": sdf["condition"].iloc[0],
        "n_cells": len(sdf),
        "n_mappable_genes": int((token_by_col >= 0).sum()),
        "mean_token_len": round(float(lengths.mean()), 1),
        "min_token_len": int(lengths.min()),
        "max_token_len": int(lengths.max()),
    })
    log(f"  GSE209552 {sid:28s} ({sdf['condition'].iloc[0]:7s}): n={len(sdf):4d} | mappable_genes={(token_by_col>=0).sum()} | mean_len={lengths.mean():.1f}")

df209 = pd.DataFrame(gse209_records)
assert len(df209) == len(gse209_seqs) == len(meta209), f"Count mismatch: {len(df209)} vs {len(meta209)}"
log(f"GSE209552 tokenized total: {len(df209):,} cells | grp={df209['grp'].value_counts().to_dict()} | ct={df209['cell_type'].value_counts().to_dict()}")

# Save GSE209552 compact pickle + metadata CSV
gse209_pkl_path = os.path.join(OUT_DIR, "gse209552_tokenized.pkl")
with open(gse209_pkl_path, "wb") as f:
    pickle.dump({"meta": df209, "input_ids": gse209_seqs, "sample_stats": sample_stats_209}, f, protocol=pickle.HIGHEST_PROTOCOL)
df209.to_csv(os.path.join(OUT_DIR, "gse209552_tokenized_meta.csv.gz"), index=False, compression="gzip")
pd.DataFrame(sample_stats_209).to_csv(os.path.join(OUT_DIR, "gse209552_sample_tokenize_stats.csv"), index=False)
log(f"Saved {gse209_pkl_path} ({os.path.getsize(gse209_pkl_path)/1e6:.2f} MB)")

# ==============================================================================
# PART 2: Tokenize In-House Mouse D30 GSE333879 (35,047 nuclei: Sham / CCI_Scr / KO)
# ==============================================================================
log("--- Part 2: Tokenizing In-House Mouse D30 GSE333879 (ISP Phase 2 Anti-Circularity & KO Arm) ---")
gse333_dir = os.path.join(PROJ, "SciData_descriptor", "geo_submission", "geo_upload_ready")
meta333 = pd.read_csv(os.path.join(gse333_dir, "per_nucleus_metadata.csv"))

ct_map_333 = {
    "Microglia": "Microglia",
    "Astrocytes": "Astrocyte",
    "Oligodendrocytes": "Oligodendrocyte",
    "OPCs": "OPC",
    "Neuron": "Neuron",
    "Pericyte": "Pericyte",
}
grp_map_333 = {
    "Sham": "sham",
    "CCI_Scr": "injury",
    "CCI_TGFbetaR1_KO": "injury_KO",
}

gse333_records = []
gse333_seqs = []
lib_stats_333 = []

for lib in ["Sham", "CCI_Scr", "CCI_TGFbetaR1_KO"]:
    sdf = meta333[meta333["library"] == lib].copy()
    with gzip.open(os.path.join(gse333_dir, f"{lib}_features.tsv.gz"), "rt", encoding="utf-8") as f:
        mouse_genes = [line.rstrip("\n").split("\t")[1] for line in f]
    with gzip.open(os.path.join(gse333_dir, f"{lib}_barcodes.tsv.gz"), "rt", encoding="utf-8") as f:
        bcs = [line.rstrip("\n") for line in f]
    bc_to_col = {bc: idx for idx, bc in enumerate(bcs)}

    # Exact planA_35c2_merge_h5ad.py ortholog projection:
    # Map mouse gene symbols -> human Ensembl IDs via mouse2human, sum if any collide onto sorted unique human Ensembl IDs
    mapped_pairs = [(gi, mouse2human[g]) for gi, g in enumerate(mouse_genes) if g in mouse2human]
    unique_hens = sorted(set(h for _, h in mapped_pairs))
    hens_idx = {h: i for i, h in enumerate(unique_hens)}
    row_map_arr = np.full(len(mouse_genes), -1, dtype=np.int64)
    for gi, h in mapped_pairs:
        row_map_arr[gi] = hens_idx[h]

    # Token and median arrays on unique_hens
    token_by_col = np.full(len(unique_hens), -1, dtype=np.int64)
    med_by_col = np.ones(len(unique_hens), dtype=np.float64)
    for j, e in enumerate(unique_hens):
        if e in tok_dict and e in median_dict:
            token_by_col[j] = tok_dict[e]
            med_by_col[j] = median_dict[e]

    # Read matrix.mtx.gz (mouse_genes x all_barcodes), subset to QC-passed barcodes first
    # Note: In merged Seurat metadata, barcodes have library suffix (-1, -2, -3), whereas raw 10x barcodes end in -1
    E_csc = sio.mmread(os.path.join(gse333_dir, f"{lib}_matrix.mtx.gz")).tocsc()
    sel_cols = [bc_to_col[bc.split("-")[0] + "-1"] for bc in sdf["barcode"]]
    E_sub_csc = E_csc[:, sel_cols]  # (mouse_genes, n_qc_cells)

    # Also compute exact planA_35a is_dbl flag: my_m = mean(Lyz2,Hexb,Aif1,Itgam) > 0.5 & ol_m = mean(Plp1,Mbp,Mog) > 1
    mg_idx = [i for i, g in enumerate(mouse_genes) if g in ("Lyz2", "Hexb", "Aif1", "Itgam")]
    ol_idx = [i for i, g in enumerate(mouse_genes) if g in ("Plp1", "Mbp", "Mog")]
    my_m = np.asarray(E_sub_csc[mg_idx, :].mean(axis=0)).ravel()
    ol_m = np.asarray(E_sub_csc[ol_idx, :].mean(axis=0)).ravel()
    is_dbl_arr = ((my_m > 0.5) & (ol_m > 1.0)).astype(np.int32)

    # Project to human Ensembl space (exact match to planA_35c2_merge_h5ad.py lines 43-49)
    E_coo = E_sub_csc.tocoo()
    mask = row_map_arr[E_coo.row] >= 0
    new_row = row_map_arr[E_coo.row[mask]]
    new_col = E_coo.col[mask]
    new_data = E_coo.data[mask]
    M_csr_cell_by_gene = sp.csr_matrix(
        (new_data, (new_row, new_col)),
        shape=(len(unique_hens), len(sel_cols))
    ).T.tocsr()  # (n_qc_cells, n_unique_hens)

    n_counts_arr = np.asarray(M_csr_cell_by_gene.sum(axis=1)).ravel().astype(np.float64)
    seqs, lengths = tokenize_csr_cell_by_gene(M_csr_cell_by_gene, token_by_col, med_by_col, n_counts_arr)

    gse333_seqs.extend(seqs)
    for idx_row, (_, row) in enumerate(sdf.iterrows()):
        gse333_records.append({
            "cell_id": f"{lib}_{row['barcode']}",
            "barcode": row["barcode"],
            "sample": lib,
            "library": lib,
            "grp": "sham" if lib == "Sham" else "injury",
            "condition_arm": grp_map_333[lib],
            "novo_broad": row["novo_broad"],
            "novogene_celltype": row["novogene_celltype"],
            "cell_type": ct_map_333[row["novo_broad"]],
            "scDblFinder_class": row["scDblFinder_class"],
            "is_dbl": int(is_dbl_arr[idx_row]),
            "n_counts": float(n_counts_arr[idx_row]),
            "length": int(lengths[idx_row]),
            "dataset": "GSE333879",
            "time_point": "30d",
        })
    lib_stats_333.append({
        "library": lib,
        "n_cells": len(sdf),
        "n_singlets_scDbl": int((sdf["scDblFinder_class"] == "singlet").sum()),
        "n_clean_is_dbl": int((is_dbl_arr == 0).sum()),
        "n_mapped_mouse_genes": len(mapped_pairs),
        "n_unique_human_ensg": len(unique_hens),
        "n_in_gc104m": int((token_by_col >= 0).sum()),
        "mean_token_len": round(float(lengths.mean()), 1),
        "min_token_len": int(lengths.min()),
        "max_token_len": int(lengths.max()),
    })
    log(f"  GSE333879 {lib:18s}: n={len(sdf):5d} | mapped_ensg={len(unique_hens)} (in_gc104m={(token_by_col>=0).sum()}) | mean_len={lengths.mean():.1f}")

df333 = pd.DataFrame(gse333_records)
assert len(df333) == len(gse333_seqs) == len(meta333), f"Count mismatch: {len(df333)} vs {len(meta333)}"
log(f"GSE333879 tokenized total: {len(df333):,} cells | arm={df333['condition_arm'].value_counts().to_dict()} | ct={df333['cell_type'].value_counts().to_dict()}")

gse333_pkl_path = os.path.join(OUT_DIR, "gse333879_tokenized.pkl")
with open(gse333_pkl_path, "wb") as f:
    pickle.dump({"meta": df333, "input_ids": gse333_seqs, "lib_stats": lib_stats_333}, f, protocol=pickle.HIGHEST_PROTOCOL)
df333.to_csv(os.path.join(OUT_DIR, "gse333879_tokenized_meta.csv.gz"), index=False, compression="gzip")
pd.DataFrame(lib_stats_333).to_csv(os.path.join(OUT_DIR, "gse333879_lib_tokenize_stats.csv"), index=False)
log(f"Saved {gse333_pkl_path} ({os.path.getsize(gse333_pkl_path)/1e6:.2f} MB)")

summary = {
    "GSE209552_human": {
        "n_samples": len(sample_stats_209),
        "n_cells": len(df209),
        "grp_counts": df209["grp"].value_counts().to_dict(),
        "cell_type_counts": df209["cell_type"].value_counts().to_dict(),
        "mean_token_length": round(float(df209["length"].mean()), 1),
        "pkl_size_mb": round(os.path.getsize(gse209_pkl_path) / 1e6, 2),
    },
    "GSE333879_mouse_D30": {
        "n_libraries": len(lib_stats_333),
        "n_cells": len(df333),
        "condition_arm_counts": df333["condition_arm"].value_counts().to_dict(),
        "cell_type_counts": df333["cell_type"].value_counts().to_dict(),
        "microglia_by_arm": df333[df333["cell_type"] == "Microglia"]["condition_arm"].value_counts().to_dict(),
        "astrocyte_by_arm": df333[df333["cell_type"] == "Astrocyte"]["condition_arm"].value_counts().to_dict(),
        "mean_token_length": round(float(df333["length"].mean()), 1),
        "pkl_size_mb": round(os.path.getsize(gse333_pkl_path) / 1e6, 2),
    },
    "runtime_s": round(time.time() - t0, 1),
}
with open(os.path.join(OUT_DIR, "routeA_tokenize_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
log("SUMMARY: " + json.dumps(summary, ensure_ascii=False))
log("ROUTE A LOCAL TOKENIZATION DONE")
