#!/usr/bin/env python3
"""
Tier-1 Upgrade #3: White-Box Attention-Rollout Gene Regulatory Network (GRN)
============================================================================
Extracts Layer-11 (final 12th Transformer block, 12 attention heads) and
Top-4 Layer Rollout (Layers 8-11) self-attention coupling matrices from:
  1. TBI-GF Specialist (run2, epoch_end_model.pt)
  2. Frozen GF-V2-104M Pretrained Baseline

Across 5 independent biological trajectories (T1-T5):
  - T1_Mouse_Acute_le7d:        Acute <=7d TBI Microglia (discovery test split)
  - T2_Mouse_Subacute_D28:      Subacute D28 TBI Microglia (GSE271769 holdout)
  - T3_Mouse_Chronic_6mo:       Chronic 6mo TBI Microglia (GSE269748 holdout)
  - T4_Mouse_D30_GSE333879:     Unseen D30 CCI_Scr (vs CCI_TGFbetaR1_KO) Microglia (GSE333879)
  - T5_Human_Clinical_GSE209552: Unseen Human Clinical TBI Microglia (GSE209552)

Quantifies:
  - Pairwise gene-to-gene attention coupling A(i <-> j) and Disease-Adapted Rewiring
    Delta_A(i <-> j) = A_TBIGF(i <-> j) - A_Frozen(i <-> j)
  - Node attention hub centrality (weighted degree / PageRank centrality) across the
    candidate & reference genes, testing whether APOE-TREM2-SPP1-LPL-ITGAX-CST7-CD9
    forms a conserved cross-timepoint, cross-species attention hub while CDKN2A/CDKN1A
    remain peripheral effectors and P2RY12/TMEM119/CSF1R decouple.
  - Effect of microglial Tgfbr1 cKO (GSE333879 CCI_TGFbetaR1_KO vs CCI_Scr) on
    DAM hub attention coupling.
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
import torch
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
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load GC-104M dictionaries
DICT_DIR = r"F:\fm_tbi_training\kaggle_data"
TOK_PKL = os.path.join(DICT_DIR, "token_dictionary_gc104M.pkl")
NAME_PKL = os.path.join(DICT_DIR, "gene_name_id_dict_gc104M.pkl")
tok_dict = pickle.load(open(TOK_PKL, "rb"))
name_dict = pickle.load(open(NAME_PKL, "rb"))

# Target gene panel (19 core + 4 astroglial/DAM context = 23 genes)
PANEL_GENES = [
    ("APOE", "DAM_Core"),
    ("SPP1", "DAM_Core"),
    ("TREM2", "DAM_Core"),
    ("LPL", "DAM_Core"),
    ("ITGAX", "DAM_Core"),
    ("CST7", "DAM_Core"),
    ("CD9", "DAM_Core"),
    ("CTSD", "DAM_Core"),
    ("TYROBP", "DAM_Core"),
    ("B2M", "DAM_Core"),
    ("IL1B", "SASP_Inflam"),
    ("MIF", "SASP_Inflam"),
    ("BAX", "SASP_Inflam"),
    ("STAT1", "IFN_Axis"),
    ("STAT2", "IFN_Axis"),
    ("IRF7", "IFN_Axis"),
    ("HEXB", "Myeloid_Core"),
    ("P2RY12", "Homeostatic"),
    ("TMEM119", "Homeostatic"),
    ("CSF1R", "Homeostatic"),
    ("CX3CR1", "Homeostatic"),
    ("CDKN1A", "CellCycle_Sen"),
    ("CDKN2A", "CellCycle_Sen"),
]

gene_to_tok = {}
tok_to_gene = {}
gene_to_cat = {}
for g, cat in PANEL_GENES:
    if g in name_dict and name_dict[g] in tok_dict:
        tid = int(tok_dict[name_dict[g]])
        gene_to_tok[g] = tid
        tok_to_gene[tid] = g
        gene_to_cat[g] = cat

panel_gene_list = [g for g, _ in PANEL_GENES if g in gene_to_tok]
panel_tok_set = set(gene_to_tok.values())
log(f"Mapped {len(panel_gene_list)}/{len(PANEL_GENES)} panel genes to GC-104M tokens: {panel_gene_list}")

# 2. Load Tokenized Cohorts & Select Representative Microglia Enriched for Panel Genes
EXAM1_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "exam1_out")
preds_exam1 = pd.read_csv(os.path.join(EXAM1_DIR, "exam1_all_eval_preds.csv.gz"))

ARROW_DIR = r"F:\fm_tbi_training\tokenized\kernel_out5\tbi_562k.dataset"
arrow_files = sorted(glob.glob(os.path.join(ARROW_DIR, "data-*.arrow")))
tables = []
for af in arrow_files:
    with pa.memory_map(af, "r") as source:
        tables.append(ipc.open_stream(source).read_all())
full_tbl = pa.concat_tables(tables)
is_dbl_arr = full_tbl["is_dbl"].to_numpy()
clean_orig_idx = np.nonzero(is_dbl_arr == 0)[0]

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

ROUTEA_TOK_DIR = os.path.join(SEN, "results", "s1_gf_finetune", "routeA_tokenized")
d209 = pickle.load(open(os.path.join(ROUTEA_TOK_DIR, "gse209552_tokenized.pkl"), "rb"))
d333 = pickle.load(open(os.path.join(ROUTEA_TOK_DIR, "gse333879_tokenized.pkl"), "rb"))

MAX_SEQ_LEN = 512  # Top 512 rank-value tokens per cell captures active transcriptional program

def select_panel_rich_cells(candidate_seqs, max_cells=80, seed=42):
    """Selects up to max_cells sequences that contain the most panel genes within top MAX_SEQ_LEN tokens."""
    counts = []
    for idx, seq in enumerate(candidate_seqs):
        trunc = seq[:MAX_SEQ_LEN]
        c = sum(1 for t in trunc if int(t) in panel_tok_set)
        counts.append((c, len(trunc), idx))
    # Sort descending by panel gene count, then sequence length
    counts.sort(key=lambda x: (-x[0], -x[1], x[2]))
    chosen = [ candidate_seqs[idx][:MAX_SEQ_LEN] for c, _, idx in counts[:max_cells] if c >= 2 ]
    return chosen

# Build candidate pools for T1, T2, T3 from Exam 1 evaluation cells
mg_le7d_rows = preds_exam1.loc[(preds_exam1["split"] == "test_le7d") & (preds_exam1["ct"] == "Microglia") & (preds_exam1["grp"] == "injury"), "orig_row"].values
mg_d28_rows  = preds_exam1.loc[(preds_exam1["split"] == "holdout_D28") & (preds_exam1["ct"] == "Microglia") & (preds_exam1["grp"] == "injury"), "orig_row"].values
mg_6mo_rows  = preds_exam1.loc[(preds_exam1["split"] == "holdout_6mo") & (preds_exam1["ct"] == "Microglia") & (preds_exam1["grp"] == "injury"), "orig_row"].values

rng = np.random.RandomState(42)
mg_le7d_cand = [get_seq_numpy(r) for r in rng.choice(mg_le7d_rows, size=min(400, len(mg_le7d_rows)), replace=False)]
mg_d28_cand  = [get_seq_numpy(r) for r in rng.choice(mg_d28_rows, size=min(400, len(mg_d28_rows)), replace=False)]
mg_6mo_cand  = [get_seq_numpy(r) for r in rng.choice(mg_6mo_rows, size=min(400, len(mg_6mo_rows)), replace=False)]

# T4: GSE333879 D30 CCI_Scr and CCI_TGFbetaR1_KO Microglia
meta333 = pd.DataFrame(d333["meta"])
seqs333 = d333["input_ids"]
idx_333_scr = meta333.index[(meta333["cell_type"] == "Microglia") & (meta333["library"] == "CCI_Scr")].values
idx_333_ko  = meta333.index[(meta333["cell_type"] == "Microglia") & (meta333["library"] == "CCI_TGFbetaR1_KO")].values
idx_333_shm = meta333.index[(meta333["cell_type"] == "Microglia") & (meta333["library"] == "Sham")].values

mg_333_scr_cand = [seqs333[int(i)] for i in idx_333_scr]
mg_333_ko_cand  = [seqs333[int(i)] for i in idx_333_ko]
mg_333_shm_cand = [seqs333[int(i)] for i in idx_333_shm]

# T5: Human GSE209552 TBI Microglia
meta209 = pd.DataFrame(d209["meta"])
seqs209 = d209["input_ids"]
idx_209_tbi = meta209.index[(meta209["cell_type"] == "Microglia") & (meta209["grp"] == "injury")].values
mg_209_tbi_cand = [seqs209[int(i)] for i in idx_209_tbi]

cohorts_seqs = {
    "T1_Mouse_Acute_le7d": select_panel_rich_cells(mg_le7d_cand, max_cells=64),
    "T2_Mouse_Subacute_D28": select_panel_rich_cells(mg_d28_cand, max_cells=64),
    "T3_Mouse_Chronic_6mo": select_panel_rich_cells(mg_6mo_cand, max_cells=64),
    "T4_Mouse_D30_CCI_Scr": select_panel_rich_cells(mg_333_scr_cand, max_cells=64),
    "T5_Human_TBI_GSE209552": select_panel_rich_cells(mg_209_tbi_cand, max_cells=64),
    "Perturb_Mouse_D30_TGFbR1_KO": select_panel_rich_cells(mg_333_ko_cand, max_cells=64),
    "Ref_Mouse_D30_Sham": select_panel_rich_cells(mg_333_shm_cand, max_cells=64),
}
for k, v in cohorts_seqs.items():
    log(f"Selected {len(v)} panel-enriched microglia sequences for {k}")

# 3. Load TBI-GF (run2) and Frozen GF-V2-104M with Eager Attention for exact attention weight extraction
MODEL_PT = os.path.join(SEN, "results", "s1_gf_finetune", "run2_weights", "epoch_end_model.pt")
log("Loading TBI-GF (run2) and Frozen GF-V2-104M models...")
tbi_gf = GeneformerMultiTask(DICT_DIR, [14, 2], use_attention_pooling=True)
sd = torch.load(MODEL_PT, map_location="cpu")
miss, unexp = tbi_gf.load_state_dict(sd, strict=False)
assert len(miss) == 0 and len(unexp) == 0
tbi_gf.eval()

frozen_gf = BertModel.from_pretrained(DICT_DIR, attn_implementation="eager")
frozen_gf.eval()
log("Both models loaded on CPU (12 layers x 12 heads x 768 hidden).")

G = len(panel_gene_list)
gene_to_idx = {g: i for i, g in enumerate(panel_gene_list)}

def extract_attention_coupling(bert_model, seqs_list, batch_size=16):
    """
    Extracts symmetric gene-to-gene attention coupling across:
      - Layer 11 (12th / final encoder layer, averaged over 12 heads)
      - Top-4 Layer Mean Rollout (Layers 8, 9, 10, 11 averaged over 12 heads)
    Normalized by expected uniform attention (L * A_{u,v}) so scores are length-invariant!
    """
    sum_l11 = np.zeros((G, G), dtype=np.float64)
    sum_top4 = np.zeros((G, G), dtype=np.float64)
    pair_counts = np.zeros((G, G), dtype=np.int64)

    with torch.no_grad():
        for s in range(0, len(seqs_list), batch_size):
            chunk = seqs_list[s:s + batch_size]
            tensors = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(tensors, batch_first=True, padding_value=0)
            lens = torch.tensor([len(x) for x in chunk], dtype=torch.long)
            attn_mask = (torch.arange(input_ids.shape[1])[None, :] < lens[:, None]).long()

            out = bert_model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
            # out.attentions is a tuple of 12 tensors, each (B, 12, S, S)
            att_l11 = out.attentions[11].mean(dim=1).numpy()  # (B, S, S)
            att_top4 = torch.stack(out.attentions[8:12], dim=0).mean(dim=(0, 2)).numpy()  # (B, S, S)

            for b_i, seq in enumerate(chunk):
                L = float(len(seq))
                # Find positions of panel genes in this cell
                pos_by_g = []
                for pos, tok in enumerate(seq):
                    t_int = int(tok)
                    if t_int in tok_to_gene:
                        g_name = tok_to_gene[t_int]
                        pos_by_g.append((gene_to_idx[g_name], pos))
                if len(pos_by_g) < 2:
                    continue
                # Extract pairwise normalized attention (L * 0.5 * (A[p1, p2] + A[p2, p1]))
                for idx1 in range(len(pos_by_g)):
                    g1, p1 = pos_by_g[idx1]
                    for idx2 in range(idx1 + 1, len(pos_by_g)):
                        g2, p2 = pos_by_g[idx2]
                        val11 = L * 0.5 * (att_l11[b_i, p1, p2] + att_l11[b_i, p2, p1])
                        val4  = L * 0.5 * (att_top4[b_i, p1, p2] + att_top4[b_i, p2, p1])
                        sum_l11[g1, g2] += val11
                        sum_l11[g2, g1] += val11
                        sum_top4[g1, g2] += val4
                        sum_top4[g2, g1] += val4
                        pair_counts[g1, g2] += 1
                        pair_counts[g2, g1] += 1

    denom = np.maximum(pair_counts, 1).astype(np.float64)
    mean_l11 = np.where(pair_counts >= 3, sum_l11 / denom, np.nan)
    mean_top4 = np.where(pair_counts >= 3, sum_top4 / denom, np.nan)
    return mean_l11, mean_top4, pair_counts

results_by_cohort = {}
for coh_name, seqs_c in cohorts_seqs.items():
    log(f"Extracting Layer-11 & Top-4 attention matrices for {coh_name} ({len(seqs_c)} cells)...")
    tbi_l11, tbi_top4, cnts = extract_attention_coupling(tbi_gf.bert, seqs_c, batch_size=16)
    frz_l11, frz_top4, _    = extract_attention_coupling(frozen_gf, seqs_c, batch_size=16)
    results_by_cohort[coh_name] = {
        "tbi_l11": tbi_l11,
        "tbi_top4": tbi_top4,
        "frz_l11": frz_l11,
        "frz_top4": frz_top4,
        "counts": cnts,
    }

# 4. Build Edge Table & 5-Trajectory Consensus across T1-T5
TRAJ_NAMES = [
    "T1_Mouse_Acute_le7d",
    "T2_Mouse_Subacute_D28",
    "T3_Mouse_Chronic_6mo",
    "T4_Mouse_D30_CCI_Scr",
    "T5_Human_TBI_GSE209552",
]

edge_rows = []
for i in range(G):
    for j in range(i + 1, G):
        g1, g2 = panel_gene_list[i], panel_gene_list[j]
        c1, c2 = gene_to_cat[g1], gene_to_cat[g2]
        tbi_vals_l11 = []
        frz_vals_l11 = []
        tbi_vals_top4 = []
        frz_vals_top4 = []
        tot_pairs = 0
        per_traj = {}

        for tn in TRAJ_NAMES:
            r = results_by_cohort[tn]
            v_tbi11 = r["tbi_l11"][i, j]
            v_frz11 = r["frz_l11"][i, j]
            v_tbi4  = r["tbi_top4"][i, j]
            v_frz4  = r["frz_top4"][i, j]
            n_ij    = int(r["counts"][i, j])
            tot_pairs += n_ij
            per_traj[tn] = round(float(v_tbi4), 4) if not np.isnan(v_tbi4) else np.nan
            if not np.isnan(v_tbi4):
                tbi_vals_l11.append(float(v_tbi11))
                frz_vals_l11.append(float(v_frz11))
                tbi_vals_top4.append(float(v_tbi4))
                frz_vals_top4.append(float(v_frz4))

        if len(tbi_vals_top4) >= 3:
            m_tbi11 = float(np.mean(tbi_vals_l11))
            m_frz11 = float(np.mean(frz_vals_l11))
            m_tbi4  = float(np.mean(tbi_vals_top4))
            m_frz4  = float(np.mean(frz_vals_top4))
            # Check D30 perturbation effect (CCI_TGFbetaR1_KO vs CCI_Scr)
            v_scr = results_by_cohort["T4_Mouse_D30_CCI_Scr"]["tbi_top4"][i, j]
            v_ko  = results_by_cohort["Perturb_Mouse_D30_TGFbR1_KO"]["tbi_top4"][i, j]
            v_shm = results_by_cohort["Ref_Mouse_D30_Sham"]["tbi_top4"][i, j]

            edge_rows.append({
                "gene_a": g1,
                "gene_b": g2,
                "category_a": c1,
                "category_b": c2,
                "edge_type": f"{c1}<->{c2}" if c1 <= c2 else f"{c2}<->{c1}",
                "n_trajectories_detected": len(tbi_vals_top4),
                "total_cooccurring_cells": tot_pairs,
                "tbi_gf_layer11_attn": round(m_tbi11, 4),
                "frozen_layer11_attn": round(m_frz11, 4),
                "delta_layer11_attn": round(m_tbi11 - m_frz11, 4),
                "fold_layer11_attn": round(m_tbi11 / max(m_frz11, 1e-4), 4),
                "tbi_gf_top4_rollout": round(m_tbi4, 4),
                "frozen_top4_rollout": round(m_frz4, 4),
                "delta_top4_rollout": round(m_tbi4 - m_frz4, 4),
                "fold_top4_rollout": round(m_tbi4 / max(m_frz4, 1e-4), 4),
                "T1_Mouse_Acute_le7d": per_traj["T1_Mouse_Acute_le7d"],
                "T2_Mouse_Subacute_D28": per_traj["T2_Mouse_Subacute_D28"],
                "T3_Mouse_Chronic_6mo": per_traj["T3_Mouse_Chronic_6mo"],
                "T4_Mouse_D30_CCI_Scr": per_traj["T4_Mouse_D30_CCI_Scr"],
                "T5_Human_TBI_GSE209552": per_traj["T5_Human_TBI_GSE209552"],
                "D30_Sham_attn": round(float(v_shm), 4) if not np.isnan(v_shm) else np.nan,
                "D30_TGFbR1_KO_attn": round(float(v_ko), 4) if not np.isnan(v_ko) else np.nan,
                "D30_KO_vs_Scr_delta": round(float(v_ko - v_scr), 4) if (not np.isnan(v_ko) and not np.isnan(v_scr)) else np.nan,
            })

edge_df = pd.DataFrame(edge_rows).sort_values("tbi_gf_top4_rollout", ascending=False).reset_index(drop=True)

# 5. Compute Node Attention Hub Centrality Summary
node_rows = []
for g in panel_gene_list:
    sub = edge_df[(edge_df["gene_a"] == g) | (edge_df["gene_b"] == g)]
    if len(sub) == 0:
        continue
    # Also compute mean attention specifically to DAM_Core partners
    dam_partners = sub[
        ((sub["gene_a"] == g) & (sub["category_b"] == "DAM_Core")) |
        ((sub["gene_b"] == g) & (sub["category_a"] == "DAM_Core"))
    ]
    node_rows.append({
        "gene": g,
        "category": gene_to_cat[g],
        "n_edges": int(len(sub)),
        "mean_tbi_gf_top4_attn": round(float(sub["tbi_gf_top4_rollout"].mean()), 4),
        "mean_frozen_top4_attn": round(float(sub["frozen_top4_rollout"].mean()), 4),
        "mean_delta_top4_attn": round(float(sub["delta_top4_rollout"].mean()), 4),
        "mean_fold_vs_frozen": round(float(sub["tbi_gf_top4_rollout"].mean() / max(sub["frozen_top4_rollout"].mean(), 1e-4)), 4),
        "mean_tbi_gf_layer11_attn": round(float(sub["tbi_gf_layer11_attn"].mean()), 4),
        "mean_frozen_layer11_attn": round(float(sub["frozen_layer11_attn"].mean()), 4),
        "dam_hub_coupling_tbi_gf": round(float(dam_partners["tbi_gf_top4_rollout"].mean()), 4) if len(dam_partners) > 0 else np.nan,
        "dam_hub_coupling_frozen": round(float(dam_partners["frozen_top4_rollout"].mean()), 4) if len(dam_partners) > 0 else np.nan,
        "T1_Mouse_Acute": round(float(sub["T1_Mouse_Acute_le7d"].dropna().mean()), 4) if len(sub["T1_Mouse_Acute_le7d"].dropna()) > 0 else np.nan,
        "T2_Mouse_D28": round(float(sub["T2_Mouse_Subacute_D28"].dropna().mean()), 4) if len(sub["T2_Mouse_Subacute_D28"].dropna()) > 0 else np.nan,
        "T3_Mouse_6mo": round(float(sub["T3_Mouse_Chronic_6mo"].dropna().mean()), 4) if len(sub["T3_Mouse_Chronic_6mo"].dropna()) > 0 else np.nan,
        "T4_Mouse_D30_Scr": round(float(sub["T4_Mouse_D30_CCI_Scr"].dropna().mean()), 4) if len(sub["T4_Mouse_D30_CCI_Scr"].dropna()) > 0 else np.nan,
        "T5_Human_GSE209552": round(float(sub["T5_Human_TBI_GSE209552"].dropna().mean()), 4) if len(sub["T5_Human_TBI_GSE209552"].dropna()) > 0 else np.nan,
    })

node_df = pd.DataFrame(node_rows).sort_values("mean_tbi_gf_top4_attn", ascending=False).reset_index(drop=True)
node_df["hub_rank"] = np.arange(1, len(node_df) + 1)

edges_csv = os.path.join(OUT_DIR, "tier1_attention_grn_edges.csv")
nodes_csv = os.path.join(OUT_DIR, "tier1_attention_grn_nodes.csv")
edge_df.to_csv(edges_csv, index=False)
node_df.to_csv(nodes_csv, index=False)

log(f"Saved {edges_csv} ({len(edge_df)} edges) and {nodes_csv} ({len(node_df)} nodes)")
print("\n=== TOP 15 ATTENTION-ROLLOUT GRN HUB NODES ===")
print(node_df.head(15).to_string(index=False))
print("\n=== TOP 20 ATTENTION-ROLLOUT GRN EDGES ===")
print(edge_df[["gene_a", "gene_b", "edge_type", "tbi_gf_top4_rollout", "frozen_top4_rollout", "delta_top4_rollout", "fold_top4_rollout", "T1_Mouse_Acute_le7d", "T3_Mouse_Chronic_6mo", "T5_Human_TBI_GSE209552"]].head(20).to_string(index=False))
