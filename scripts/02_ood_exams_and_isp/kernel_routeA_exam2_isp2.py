#!/usr/bin/env python3
"""
Route A Kaggle T4 GPU Kernel (v2 - FP16 Turing TensorCore + Length-Sorted Single-Pass Forward)
==============================================================================================
Fixes v1 OOM on Tesla T4 (SM 7.5):
  1. Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing torch
  2. Uses torch.float16 (native Turing Tensor Core & mem-efficient SDPA dtype on T4; bf16 falls back to unfused fp32 math SDPA on SM 7.5)
  3. Sorts sequences by token length before batching (eliminates padding waste) and uses batch_size=12
  4. Computes logits_t1, logits_t2, and L2-normalized embeddings from a SINGLE model.bert() pass (2x faster, 50% less memory)
  5. Runs frozen_bert in a separate isolated pass and frees VRAM immediately via del frozen_bert + empty_cache()
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import time
import json
import pickle
import hashlib
import shutil
import tarfile
import urllib.request
import subprocess
import numpy as np
import pandas as pd

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

# ===== G1: GPU Gate =====
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

assert torch.cuda.is_available(), "G1 FAIL: No CUDA GPU detected!"
log(f"G1 PASS: GPU = {torch.cuda.get_device_name(0)} (VRAM = {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

# ===== Install vendored Geneformer =====
log("Installing vendored Geneformer from SJT503/geneformer-src...")
tb = "/kaggle/working/gf.tar.gz"
urllib.request.urlretrieve("https://codeload.github.com/SJT503/geneformer-src/tar.gz/refs/heads/main", tb)
with tarfile.open(tb) as tf:
    tf.extractall("/kaggle/working/gfsrc")
srcdir = glob.glob("/kaggle/working/gfsrc/geneformer-src-*")[0]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-warn-script-location", srcdir], check=True)
for dep in ["tdigest", "optuna", "optuna-integration", "peft", "tensorboard", "transformers<5"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location", dep], check=True)

pkgdirs = glob.glob("/usr/local/lib/python3.*/dist-packages/geneformer")
assert pkgdirs, "geneformer dist-packages dir not found"
pkgdir = pkgdirs[0]
for pkl_f in glob.glob(f"{srcdir}/geneformer/*.pkl"):
    shutil.copy(pkl_f, pkgdir)

# Clean upgfsrc right away so /kaggle/working stays clean
for tmp_p in ["/kaggle/working/gf.tar.gz", "/kaggle/working/gfsrc"]:
    if os.path.exists(tmp_p):
        shutil.rmtree(tmp_p, ignore_errors=True) if os.path.isdir(tmp_p) else os.remove(tmp_p)

import transformers as _tf
from transformers import BertModel
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, average_precision_score
from geneformer.mtl.model import GeneformerMultiTask

log(f"Transformers version: {_tf.__version__}; GeneformerMultiTask imported successfully.")

# ===== G2: Locate Mounted Datasets =====
def find_file(pattern, desc):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"G2 FAIL: Could not find {desc} ({pattern})"
    log(f"Found {desc}: {hits[0]}")
    return hits[0]

PRE_CFG = find_file("config.json", "GF-V2-104M config.json")
PRE_DIR = os.path.dirname(PRE_CFG)
MODEL_PT = find_file("epoch_end_model.pt", "run2 epoch_end_model.pt")
MAPS_PKL = find_file("task_label_mappings.pkl", "task_label_mappings.pkl")
GSE209_PKL = find_file("gse209552_tokenized.pkl", "GSE209552 tokenized pkl")
GSE333_PKL = find_file("gse333879_tokenized.pkl", "GSE333879 tokenized pkl")
NAME_PKL = find_file("gene_name_id_dict_gc104M.pkl", "gene_name_id_dict_gc104M.pkl")
TOK_PKL = find_file("token_dictionary_gc104M.pkl", "token_dictionary_gc104M.pkl")

pt_md5 = hashlib.md5(open(MODEL_PT, "rb").read()).hexdigest()
log(f"epoch_end_model.pt MD5 = {pt_md5}")
assert pt_md5 == "8f5cf1c4f0343132a8b04281504679f2", f"MD5 mismatch: {pt_md5}"

OUT = "/kaggle/working/routeA_out"
os.makedirs(OUT, exist_ok=True)

# ===== Load Label Mappings =====
maps = pickle.load(open(MAPS_PKL, "rb"))
def as_list(x):
    if isinstance(x, dict):
        return sorted(x, key=lambda k: x[k]) if all(isinstance(v, int) for v in x.values()) else list(x)
    return list(x)
try:
    labels1, labels2 = as_list(maps["task1"]), as_list(maps["task2"])
except Exception:
    labels1, labels2 = as_list(maps["cell_type"]), as_list(maps["grp"])

INJ_IDX = labels2.index("injury")
log(f"task1 labels ({len(labels1)}): {labels1}")
log(f"task2 labels ({len(labels2)}): {labels2} (INJ_IDX={INJ_IDX})")

# Map between run2's exact 14 mouse labels and our standardized 6 broad labels
# Note: in run2, labels1 = ['Astrocyte', 'ChoroidPlx', 'Endothel', 'Ependymal', 'ExN', 'InN', 'Macrophage', 'Microglia', 'Neutrophil', 'OPC', 'Oligo', 'Pericyte', 'Proliferat', 'Tcell']
RUN2_TO_BROAD = {
    "Astrocyte": "Astrocyte",
    "Microglia": "Microglia",
    "Macrophage": "Microglia",
    "Oligo": "Oligodendrocyte",
    "OPC": "OPC",
    "ExN": "Neuron",
    "InN": "Neuron",
    "Endothel": "Endothelial",
    "Pericyte": "Pericyte",
    "ChoroidPlx": "Other",
    "Ependymal": "Other",
    "Neutrophil": "Other",
    "Proliferat": "Other",
    "Tcell": "Other",
}

def load_run2_model():
    m = GeneformerMultiTask(PRE_DIR, [len(labels1), len(labels2)], use_attention_pooling=True)
    sd = torch.load(MODEL_PT, map_location="cpu")
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, f"State dict mismatch: miss={miss}, unexp={unexp}"
    return m.cuda().eval()

def forward_single_pass_sorted(model_obj, seqs, batch_size=12, is_mtl=True):
    """
    Sorts sequences by length to minimize padding, runs a SINGLE bert forward pass in FP16,
    and unsorts back to original order.
    """
    n = len(seqs)
    lengths_all = np.array([len(x) for x in seqs], dtype=np.int32)
    order = np.argsort(lengths_all)

    p_inj = np.zeros(n, dtype=np.float32) if is_mtl else None
    pred_t1 = np.zeros(n, dtype=np.int32) if is_mtl else None
    logits_t1_all = np.zeros((n, len(labels1)), dtype=np.float32) if is_mtl else None
    emb_all = np.zeros((n, 768), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, n, batch_size):
            batch_idx = order[s:s + batch_size]
            chunk = [seqs[i] for i in batch_idx]
            tensors = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(tensors, batch_first=True, padding_value=0).cuda()
            lens = torch.tensor([len(x) for x in chunk], dtype=torch.long, device="cuda")
            attn = (torch.arange(input_ids.shape[1], device="cuda")[None, :] < lens[:, None]).long()
            mask_f = attn.unsqueeze(-1).float()

            with torch.autocast("cuda", dtype=torch.float16):
                if is_mtl:
                    bert_out = model_obj.bert(input_ids=input_ids, attention_mask=attn)
                    seq_out = bert_out.last_hidden_state
                    pooled_att = model_obj.attention_pool(seq_out) if model_obj.use_attention_pooling else seq_out[:, 0, :]
                    l1 = model_obj.classification_heads[0](pooled_att).float()
                    l2 = model_obj.classification_heads[1](pooled_att).float()
                else:
                    bert_out = model_obj(input_ids=input_ids, attention_mask=attn)
                    seq_out = bert_out.last_hidden_state

                hidden_f = seq_out.float()
                pooled_mean = (hidden_f * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
                pooled_mean = F.normalize(pooled_mean, p=2, dim=-1)

            emb_all[batch_idx] = pooled_mean.cpu().numpy()
            if is_mtl:
                logits_t1_all[batch_idx] = l1.cpu().numpy()
                pred_t1[batch_idx] = l1.argmax(dim=-1).cpu().numpy()
                p_inj[batch_idx] = torch.softmax(l2, dim=-1)[:, INJ_IDX].cpu().numpy()

            if (s // batch_size) % 400 == 0:
                log(f"  forward {s:,}/{n:,} (max_len={input_ids.shape[1]}, VRAM_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB)")

    return p_inj, pred_t1, logits_t1_all, emb_all

# ==============================================================================
# EXAM 2: Cross-Species Zero-Shot Migration on Human Acute TBI GSE209552
# ==============================================================================
log("=== EXAM 2: Human Acute TBI GSE209552 (17 Samples, 24,241 Nuclei) ===")
d209 = pickle.load(open(GSE209_PKL, "rb"))
df209 = d209["meta"].copy()
seqs209 = d209["input_ids"]

# Step 2.1: Frozen pretrained GF-V2-104M pass first, then free GPU memory
log("Running Frozen Pretrained GF-V2-104M on GSE209552...")
frozen_bert = BertModel.from_pretrained(PRE_DIR, attn_implementation="sdpa").cuda().eval()
_, _, _, emb_fr_209 = forward_single_pass_sorted(frozen_bert, seqs209, batch_size=12, is_mtl=False)
del frozen_bert
torch.cuda.empty_cache()
log("Frozen Pretrained GF-V2-104M pass complete and freed from VRAM.")

# Step 2.2: Load Fine-Tuned Run2 model and run on GSE209552
log("Loading Run2 Fine-Tuned GeneformerMultiTask on GPU...")
model = load_run2_model()
p_inj_209, pred_t1_209, logits_t1_209, emb_ft_209 = forward_single_pass_sorted(model, seqs209, batch_size=12, is_mtl=True)

df209["pred_cell_type_raw14"] = [labels1[i] for i in pred_t1_209]
df209["pred_broad_ct"] = [RUN2_TO_BROAD.get(c, "Other") for c in df209["pred_cell_type_raw14"]]
df209["P_injury"] = np.round(p_inj_209, 5)
df209["pred_grp"] = np.where(p_inj_209 >= 0.5, "injury", "sham")
df209["true_injury_bin"] = (df209["grp"] == "injury").astype(int)

# Compute 6-class restricted argmax across the 6 canonical classes in GSE209552
# Note: Neuron in run2 is max(ExN, InN); Oligodendrocyte is Oligo; Endothelial is max(Endothel, Pericyte)
idx_astro = labels1.index("Astrocyte")
idx_mg = labels1.index("Microglia")
idx_macro = labels1.index("Macrophage")
idx_oligo = labels1.index("Oligo")
idx_opc = labels1.index("OPC")
idx_exn = labels1.index("ExN")
idx_inn = labels1.index("InN")
idx_endo = labels1.index("Endothel")
idx_peri = labels1.index("Pericyte")

score_6 = np.column_stack([
    np.maximum(logits_t1_209[:, idx_mg], logits_t1_209[:, idx_macro]),   # Microglia
    logits_t1_209[:, idx_astro],                                        # Astrocyte
    logits_t1_209[:, idx_oligo],                                        # Oligodendrocyte
    logits_t1_209[:, idx_opc],                                          # OPC
    np.maximum(logits_t1_209[:, idx_exn], logits_t1_209[:, idx_inn]),   # Neuron
    np.maximum(logits_t1_209[:, idx_endo], logits_t1_209[:, idx_peri]), # Endothelial
])
canon6 = ["Microglia", "Astrocyte", "Oligodendrocyte", "OPC", "Neuron", "Endothelial"]
df209["pred_canon6_ct"] = [canon6[i] for i in score_6.argmax(axis=-1)]

# Compute embedding cosine distance to control centroid per cell type (Fine-Tuned vs Frozen)
df209["frozen_injury_score"] = 0.0
df209["ft_injury_cos_score"] = 0.0
for ct in canon6:
    m_ct = (df209["cell_type"] == ct).values
    m_ctl = m_ct & (df209["grp"] == "sham").values
    if m_ctl.sum() > 5:
        c_ctl_fr = emb_fr_209[m_ctl].mean(axis=0)
        c_ctl_fr /= np.linalg.norm(c_ctl_fr) + 1e-12
        df209.loc[m_ct, "frozen_injury_score"] = 1.0 - (emb_fr_209[m_ct] @ c_ctl_fr)

        c_ctl_ft = emb_ft_209[m_ctl].mean(axis=0)
        c_ctl_ft /= np.linalg.norm(c_ctl_ft) + 1e-12
        df209.loc[m_ct, "ft_injury_cos_score"] = 1.0 - (emb_ft_209[m_ct] @ c_ctl_ft)

# 1. Cell-level & Per-Cell-Type Summary
ct_rows_209 = []
for ct in ["ALL"] + canon6:
    sub = df209 if ct == "ALL" else df209[df209["cell_type"] == ct]
    y = sub["true_injury_bin"].values
    s_ft = sub["P_injury"].values
    s_cos_ft = sub["ft_injury_cos_score"].values
    s_cos_fr = sub["frozen_injury_score"].values
    if len(np.unique(y)) < 2:
        continue
    auroc_ft = float(roc_auc_score(y, s_ft))
    auprc_ft = float(average_precision_score(y, s_ft))
    auroc_cos_ft = float(roc_auc_score(y, s_cos_ft))
    auroc_cos_fr = float(roc_auc_score(y, s_cos_fr))
    m_tbi = y == 1
    m_ctl = y == 0
    _, p_mwu = sp_stats.mannwhitneyu(s_ft[m_tbi], s_ft[m_ctl], alternative="two-sided")
    ct_rows_209.append({
        "cell_type": ct,
        "n_total": len(sub),
        "n_tbi": int(m_tbi.sum()),
        "n_control": int(m_ctl.sum()),
        "mean_P_injury_TBI": round(float(s_ft[m_tbi].mean()), 4),
        "mean_P_injury_Control": round(float(s_ft[m_ctl].mean()), 4),
        "delta_P_injury": round(float(s_ft[m_tbi].mean() - s_ft[m_ctl].mean()), 4),
        "cell_AUROC_run2_prob": round(auroc_ft, 4),
        "cell_AUPRC_run2_prob": round(auprc_ft, 4),
        "cell_AUROC_run2_emb_shift": round(auroc_cos_ft, 4),
        "cell_AUROC_frozen_emb_shift": round(auroc_cos_fr, 4),
        "mwu_p": float(p_mwu),
        "ct_acc_broad14": round(float((sub["pred_broad_ct"] == sub["cell_type"]).mean()), 4),
        "ct_acc_canon6": round(float((sub["pred_canon6_ct"] == sub["cell_type"]).mean()), 4),
    })

ct_df_209 = pd.DataFrame(ct_rows_209)
ct_df_209.to_csv(f"{OUT}/exam2_gse209552_celltype_summary.csv", index=False)
log("Exam 2 per-cell-type summary:\n" + ct_df_209.to_string())

# 2. Sample/Donor-Level Summary (17 Samples: 5 Control vs 12 TBI)
samp_rows_209 = []
for (sid, gsm, cond), sdf in df209.groupby(["sample", "gsm", "condition_raw"], sort=False):
    row_d = {
        "sample_id": sid,
        "gsm": gsm,
        "condition": cond,
        "true_injury_bin": 1 if cond == "tbi" else 0,
        "n_cells": len(sdf),
        "mean_P_injury_all": round(float(sdf["P_injury"].mean()), 4),
        "median_P_injury_all": round(float(sdf["P_injury"].median()), 4),
        "frac_pred_injury_all": round(float((sdf["P_injury"] >= 0.5).mean()), 4),
        "ct_acc_broad14": round(float((sdf["pred_broad_ct"] == sdf["cell_type"]).mean()), 4),
        "ct_acc_canon6": round(float((sdf["pred_canon6_ct"] == sdf["cell_type"]).mean()), 4),
    }
    for ct in ["Microglia", "Astrocyte", "Oligodendrocyte", "OPC", "Neuron"]:
        cs = sdf[sdf["cell_type"] == ct]
        row_d[f"n_{ct}"] = len(cs)
        row_d[f"mean_P_injury_{ct}"] = round(float(cs["P_injury"].mean()), 4) if len(cs) > 0 else np.nan
        row_d[f"mean_ft_cos_{ct}"] = round(float(cs["ft_injury_cos_score"].mean()), 5) if len(cs) > 0 else np.nan
        row_d[f"mean_fr_cos_{ct}"] = round(float(cs["frozen_injury_score"].mean()), 5) if len(cs) > 0 else np.nan
    samp_rows_209.append(row_d)

samp_df_209 = pd.DataFrame(samp_rows_209)
samp_df_209.to_csv(f"{OUT}/exam2_gse209552_sample_summary.csv", index=False)

donor_metrics_209 = {}
for col in ["mean_P_injury_all", "mean_P_injury_Microglia", "mean_P_injury_Astrocyte", "mean_P_injury_Oligodendrocyte", "mean_P_injury_OPC", "mean_P_injury_Neuron",
            "mean_ft_cos_Microglia", "mean_fr_cos_Microglia", "mean_ft_cos_Astrocyte", "mean_fr_cos_Astrocyte",
            "mean_ft_cos_Oligodendrocyte", "mean_fr_cos_Oligodendrocyte"]:
    valid = samp_df_209.dropna(subset=[col])
    y_s = valid["true_injury_bin"].values
    v_s = valid[col].values
    if len(np.unique(y_s)) == 2:
        auc_s = float(roc_auc_score(y_s, v_s))
        _, p_s = sp_stats.mannwhitneyu(v_s[y_s == 1], v_s[y_s == 0], alternative="two-sided")
        donor_metrics_209[col] = {
            "n_samples": len(valid),
            "donor_AUROC": round(auc_s, 4),
            "mean_TBI": round(float(v_s[y_s == 1].mean()), 4),
            "mean_Control": round(float(v_s[y_s == 0].mean()), 4),
            "mwu_p": round(float(p_s), 5),
        }

df209.to_csv(f"{OUT}/exam2_gse209552_cell_preds.csv.gz", index=False, compression="gzip")
with open(f"{OUT}/exam2_gse209552_metrics.json", "w") as f:
    json.dump({"cell_level": ct_rows_209, "donor_level": donor_metrics_209}, f, indent=2)
log("Exam 2 donor-level metrics:\n" + json.dumps(donor_metrics_209, indent=2))

del emb_fr_209, emb_ft_209, d209, seqs209
torch.cuda.empty_cache()

# ==============================================================================
# ISP PHASE 2 PART A: Zero-Shot & In Vivo Tgfbr1-KO Evaluation on GSE333879 (D30)
# ==============================================================================
log("=== ISP PHASE 2 PART A: In-House Mouse D30 GSE333879 (35,047 Nuclei: Sham / CCI_Scr / KO) ===")
d333 = pickle.load(open(GSE333_PKL, "rb"))
df333 = d333["meta"].copy()
seqs333 = d333["input_ids"]

p_inj_333, pred_t1_333, logits_t1_333, emb_ft_333 = forward_single_pass_sorted(
    model, seqs333, batch_size=12, is_mtl=True
)

df333["pred_cell_type_raw14"] = [labels1[i] for i in pred_t1_333]
df333["pred_broad_ct"] = [RUN2_TO_BROAD.get(c, "Other") for c in df333["pred_cell_type_raw14"]]
df333["P_injury"] = np.round(p_inj_333, 5)
df333["pred_grp"] = np.where(p_inj_333 >= 0.5, "injury", "sham")

# Cosine similarity to Sham centroid per cell_type
df333["cos_to_sham_centroid"] = np.nan
for ct in df333["cell_type"].unique():
    m_ct = (df333["cell_type"] == ct).values
    m_shm = m_ct & (df333["condition_arm"] == "sham").values & (df333["is_dbl"] == 0).values
    if m_shm.sum() > 5:
        c_shm = emb_ft_333[m_shm].mean(axis=0)
        c_shm /= np.linalg.norm(c_shm) + 1e-12
        df333.loc[m_ct, "cos_to_sham_centroid"] = emb_ft_333[m_ct] @ c_shm

df333.to_csv(f"{OUT}/isp2_gse333879_zeroshot_preds.csv.gz", index=False, compression="gzip")

ko_rows = []
for filter_name, mask_base in [
    ("all_QC_35047", np.ones(len(df333), dtype=bool)),
    ("clean_is_dbl0", (df333["is_dbl"] == 0).values),
    ("singlet_both", ((df333["is_dbl"] == 0) & (df333["scDblFinder_class"] == "singlet")).values),
]:
    sub_f = df333[mask_base]
    for ct in ["ALL", "Microglia", "Astrocyte", "Oligodendrocyte", "OPC", "Neuron", "Pericyte"]:
        sub = sub_f if ct == "ALL" else sub_f[sub_f["cell_type"] == ct]
        if len(sub) == 0:
            continue
        shm = sub[sub["condition_arm"] == "sham"]
        cci = sub[sub["condition_arm"] == "injury"]
        ko = sub[sub["condition_arm"] == "injury_KO"]

        sub_sc = sub[sub["condition_arm"].isin(["sham", "injury"])]
        y_sc = (sub_sc["condition_arm"] == "injury").astype(int).values
        auroc_sc = float(roc_auc_score(y_sc, sub_sc["P_injury"].values)) if len(np.unique(y_sc)) == 2 else np.nan
        acc_sc = float((sub_sc["pred_grp"] == sub_sc["grp"]).mean()) if len(sub_sc) > 0 else np.nan
        ct_acc_broad = float((sub["pred_broad_ct"] == sub["cell_type"]).mean())

        _, p_cci_vs_shm = sp_stats.mannwhitneyu(cci["P_injury"], shm["P_injury"], alternative="two-sided") if (len(cci) > 0 and len(shm) > 0) else (np.nan, np.nan)
        _, p_ko_vs_cci = sp_stats.mannwhitneyu(ko["P_injury"], cci["P_injury"], alternative="two-sided") if (len(ko) > 0 and len(cci) > 0) else (np.nan, np.nan)
        _, p_cos_ko_vs_cci = sp_stats.mannwhitneyu(ko["cos_to_sham_centroid"].dropna(), cci["cos_to_sham_centroid"].dropna(), alternative="two-sided") if (len(ko) > 0 and len(cci) > 0) else (np.nan, np.nan)

        ko_rows.append({
            "filter": filter_name,
            "cell_type": ct,
            "n_sham": len(shm),
            "n_CCI_Scr": len(cci),
            "n_CCI_KO": len(ko),
            "ct_acc_broad": round(ct_acc_broad, 4),
            "Sham_vs_CCI_grp_acc": round(acc_sc, 4),
            "Sham_vs_CCI_AUROC": round(auroc_sc, 4) if not np.isnan(auroc_sc) else np.nan,
            "mean_P_injury_Sham": round(float(shm["P_injury"].mean()), 4),
            "mean_P_injury_CCI_Scr": round(float(cci["P_injury"].mean()), 4),
            "mean_P_injury_CCI_KO": round(float(ko["P_injury"].mean()), 4),
            "delta_P_KO_minus_CCI": round(float(ko["P_injury"].mean() - cci["P_injury"].mean()), 4),
            "p_mwu_CCI_vs_Sham": float(p_cci_vs_shm),
            "p_mwu_KO_vs_CCI": float(p_ko_vs_cci),
            "mean_cos_to_Sham_CCI_Scr": round(float(cci["cos_to_sham_centroid"].mean()), 5),
            "mean_cos_to_Sham_CCI_KO": round(float(ko["cos_to_sham_centroid"].mean()), 5),
            "delta_cos_KO_minus_CCI": round(float(ko["cos_to_sham_centroid"].mean() - cci["cos_to_sham_centroid"].mean()), 5),
            "p_mwu_cos_KO_vs_CCI": float(p_cos_ko_vs_cci),
        })

ko_df = pd.DataFrame(ko_rows)
ko_df.to_csv(f"{OUT}/isp2_gse333879_ko_rescue_summary.csv", index=False)
log("GSE333879 Zero-Shot & KO Summary (clean_is_dbl0):\n" + ko_df[ko_df["filter"] == "clean_is_dbl0"].to_string())

# ==============================================================================
# ISP PHASE 2 PART B: Anti-Circularity Zero-Shot In Silico Deletion on GSE333879
# ==============================================================================
log("=== ISP PHASE 2 PART B: Zero-Shot In Silico Deletion on Unseen GSE333879 D30 Microglia & Astrocytes ===")
PAPER1_CORE = ["CDKN2A", "CDKN1A", "MCL1", "CST7", "ITGAX", "STAT1", "STAT2", "IRF7", "MIF",
               "CD9", "LPL", "SPP1", "BAX", "BCL2", "IL1B", "E2F1"]
IDENTITY_ANCHORS = ["P2RY12", "TMEM119", "HEXB", "CSF1R", "AIF1", "APOE", "TREM2"]
CANDIDATES = PAPER1_CORE + IDENTITY_ANCHORS

name_id = pickle.load(open(NAME_PKL, "rb"))
tok_dict = pickle.load(open(TOK_PKL, "rb"))
def resolve_sym(sym):
    if sym in name_id and name_id[sym] in tok_dict:
        return tok_dict[name_id[sym]]
    return None

cand_toks = {g: resolve_sym(g) for g in CANDIDATES if resolve_sym(g) is not None}
log(f"Resolved {len(cand_toks)}/{len(CANDIDATES)} candidate genes for ISP Phase 2: {list(cand_toks.keys())}")

COMBO_SETS = {
    "COMBO_TOP2_APOE_IL1B": [cand_toks["APOE"], cand_toks["IL1B"]],
    "COMBO_TOP3_APOE_IL1B_BAX": [cand_toks["APOE"], cand_toks["IL1B"], cand_toks["BAX"]],
    "COMBO_TOP5_HUB": [cand_toks["APOE"], cand_toks["IL1B"], cand_toks["BAX"], cand_toks["MIF"], cand_toks["CD9"]],
}

rng = np.random.RandomState(42)
idx_mg_cci = df333[(df333["is_dbl"] == 0) & (df333["cell_type"] == "Microglia") & (df333["condition_arm"] == "injury")].index.tolist()
idx_as_cci = df333[(df333["is_dbl"] == 0) & (df333["cell_type"] == "Astrocyte") & (df333["condition_arm"] == "injury")].index.tolist()
idx_mg_shm_all = df333[(df333["is_dbl"] == 0) & (df333["cell_type"] == "Microglia") & (df333["condition_arm"] == "sham")].index.tolist()
idx_mg_shm = rng.choice(idx_mg_shm_all, min(120, len(idx_mg_shm_all)), replace=False).tolist()

idx_as_shm_all = df333[(df333["is_dbl"] == 0) & (df333["cell_type"] == "Astrocyte") & (df333["condition_arm"] == "sham")].index.tolist()
centroid_mg_sham = emb_ft_333[idx_mg_shm_all].mean(axis=0)
centroid_mg_sham /= np.linalg.norm(centroid_mg_sham) + 1e-12
centroid_as_sham = emb_ft_333[idx_as_shm_all].mean(axis=0)
centroid_as_sham /= np.linalg.norm(centroid_as_sham) + 1e-12

del emb_ft_333
torch.cuda.empty_cache()

sel_333 = idx_mg_cci + idx_mg_shm + idx_as_cci
arms_333 = (["MG_injury_D30"] * len(idx_mg_cci)) + (["MG_sham_D30"] * len(idx_mg_shm)) + (["Astro_injury_D30"] * len(idx_as_cci))
log(f"Selected GSE333879 cells for ISP Phase 2: MG_injury_D30={len(idx_mg_cci)}, MG_sham_D30={len(idx_mg_shm)}, Astro_injury_D30={len(idx_as_cci)}")

def forward_isp_variants_fp16(seqs_list, target_centroid, batch_size=12):
    out_p, out_t, out_cos = [], [], []
    t_cent = torch.tensor(target_centroid, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for s in range(0, len(seqs_list), batch_size):
            chunk = seqs_list[s:s + batch_size]
            tensors = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(tensors, batch_first=True, padding_value=0).cuda()
            lengths = torch.tensor([len(x) for x in chunk], dtype=torch.long, device="cuda")
            attn = (torch.arange(input_ids.shape[1], device="cuda")[None, :] < lengths[:, None]).long()
            mask_f = attn.unsqueeze(-1).float()
            with torch.autocast("cuda", dtype=torch.float16):
                bert_out = model.bert(input_ids=input_ids, attention_mask=attn)
                seq_out = bert_out.last_hidden_state
                pooled_att = model.attention_pool(seq_out) if model.use_attention_pooling else seq_out[:, 0, :]
                l1 = model.classification_heads[0](pooled_att).float()
                l2 = model.classification_heads[1](pooled_att).float()
                hidden_ft = seq_out.float()
                pooled_ft = (hidden_ft * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
                pooled_ft = F.normalize(pooled_ft, p=2, dim=-1)
            pr = torch.softmax(l2, dim=-1)[:, INJ_IDX].cpu().numpy()
            t1 = l1.argmax(dim=-1).cpu().numpy()
            cos_sim = (pooled_ft @ t_cent).cpu().numpy()
            out_p.append(pr)
            out_t.append(t1)
            out_cos.append(cos_sim)
    return np.concatenate(out_p), np.concatenate(out_t), np.concatenate(out_cos)

N_NULL = 30
isp333_rows = []
cand_tok_set = set(cand_toks.values())

for ci, (row_idx, arm_name) in enumerate(zip(sel_333, arms_333)):
    ids = np.array(seqs333[row_idx], dtype=np.int64)
    t_cent = centroid_as_sham if "Astro" in arm_name else centroid_mg_sham
    variants, vnames = [ids], ["__orig__"]
    rng_ci = np.random.RandomState(1000 + ci)

    for g, t in cand_toks.items():
        if t in ids:
            variants.append(ids[ids != t])
            vnames.append(g)

    for cname, ctoks in COMBO_SETS.items():
        present_ctoks = [t for t in ctoks if t in ids]
        if len(present_ctoks) >= 2:
            variants.append(ids[~np.isin(ids, present_ctoks)])
            vnames.append(cname)

    null_pool = np.array([t for t in ids[1:-1] if t not in cand_tok_set])
    n_null = min(N_NULL, len(null_pool))
    if n_null > 0:
        null_choice = rng_ci.choice(null_pool, n_null, replace=False)
        for t in null_choice:
            variants.append(ids[ids != t])
            vnames.append("__null__")

    if len(null_pool) >= 5:
        for _ in range(5):
            k_choice = rng_ci.choice(null_pool, 5, replace=False)
            variants.append(ids[~np.isin(ids, k_choice)])
            vnames.append("__null5__")

    P_arr, T1_arr, COS_arr = forward_isp_variants_fp16(variants, t_cent, batch_size=12)
    p0, t0_, cos0 = float(P_arr[0]), int(T1_arr[0]), float(COS_arr[0])

    for name, p, t1, cos_v in zip(vnames, P_arr, T1_arr, COS_arr):
        isp333_rows.append({
            "cell_i": ci,
            "orig_row": int(row_idx),
            "arm": arm_name,
            "variant": name,
            "P_orig": round(p0, 5),
            "P_injury": round(float(p), 5),
            "dP": round(p0 - float(p), 5),
            "cos_orig": round(cos0, 5),
            "cos_sham": round(float(cos_v), 5),
            "dCos_to_sham": round(float(cos_v) - cos0, 5),
            "task1_flip": int(t1 != t0_),
        })
    if ci % 100 == 0:
        log(f"  ISP GSE333879 cell {ci}/{len(sel_333)} ({arm_name}) P_orig={p0:.4f}")

long333_df = pd.DataFrame(isp333_rows)
long333_df.to_csv(f"{OUT}/isp2_gse333879_deletion_long.csv.gz", index=False, compression="gzip")

def summarize_isp_df(df_long, subset_name, p_lo, p_hi):
    orig_df = df_long[df_long["variant"] == "__orig__"]
    if p_lo is not None and p_hi is not None:
        keep_cells = set(orig_df[(orig_df["P_orig"] > p_lo) & (orig_df["P_orig"] < p_hi)]["cell_i"])
    else:
        keep_cells = set(orig_df["cell_i"])
    sub = df_long[df_long["cell_i"].isin(keep_cells)]
    out_rows = []
    rng_mc = np.random.RandomState(42)

    for arm_name in ["MG_injury_D30", "Astro_injury_D30", "MG_sham_D30"]:
        arm_sub = sub[sub["arm"] == arm_name]
        n_arm_cells = len(arm_sub[arm_sub["variant"] == "__orig__"])
        if n_arm_cells == 0:
            continue
        null1 = arm_sub[arm_sub["variant"] == "__null__"]
        null5 = arm_sub[arm_sub["variant"] == "__null5__"]

        all_vars = list(cand_toks.keys()) + list(COMBO_SETS.keys())
        for v in all_vars:
            x = arm_sub[arm_sub["variant"] == v]
            if len(x) == 0:
                continue
            ref_null = null5 if v.startswith("COMBO_") else null1
            nm = float(ref_null["dP"].mean()) if len(ref_null) > 0 else 0.0
            nsd = float(ref_null["dP"].std(ddof=1)) if len(ref_null) > 1 else 1e-6
            dpm = float(x["dP"].mean())
            dcosm = float(x["dCos_to_sham"].mean())
            z = (dpm - nm) / (nsd / np.sqrt(len(x))) if nsd > 0 else 0.0

            cell_pools = [grp["dP"].values for _, grp in ref_null[ref_null["cell_i"].isin(x["cell_i"])].groupby("cell_i") if len(grp) > 0]
            if len(cell_pools) > 0:
                B_mc = 10000
                sim_means = np.zeros(B_mc, dtype=np.float64)
                for pool in cell_pools:
                    sim_means += pool[rng_mc.randint(0, len(pool), size=B_mc)]
                sim_means /= len(cell_pools)
                if dpm >= nm:
                    p_mc = float((1 + np.sum(sim_means >= dpm)) / (1 + B_mc))
                else:
                    p_mc = float((1 + np.sum(sim_means <= dpm)) / (1 + B_mc))
            else:
                p_mc = np.nan

            out_rows.append({
                "subset": subset_name,
                "arm": arm_name,
                "n_arm_cells": n_arm_cells,
                "variant": v,
                "n_present": len(x),
                "presence_rate": round(len(x) / n_arm_cells, 4),
                "mean_P_orig": round(float(x["P_orig"].mean()), 5),
                "mean_dP": round(dpm, 5),
                "median_dP": round(float(x["dP"].median()), 5),
                "frac_dP_pos": round(float((x["dP"] > 0).mean()), 4),
                "mean_dCos_to_sham": round(dcosm, 6),
                "null_mean_dP": round(nm, 5),
                "z_vs_null": round(z, 2),
                "p_mc_matched": round(p_mc, 5) if not np.isnan(p_mc) else np.nan,
                "task1_flip_rate": round(float(x["task1_flip"].mean()), 4),
            })
    res_df = pd.DataFrame(out_rows)
    if len(res_df) > 0:
        for arm_name in res_df["arm"].unique():
            m = (res_df["arm"] == arm_name) & (~res_df["variant"].str.startswith("COMBO_")) & (res_df["p_mc_matched"].notna())
            if m.sum() > 0:
                pvals = res_df.loc[m, "p_mc_matched"].values
                order = np.argsort(pvals)
                ranked = pvals[order]
                q = np.minimum.accumulate((ranked * len(pvals) / (np.arange(len(pvals)) + 1))[::-1])[::-1]
                q = np.clip(q, 0.0, 1.0)
                q_out = np.empty_like(q)
                q_out[order] = q
                res_df.loc[m, "q_BH_mc"] = np.round(q_out, 5)
    return res_df

sum333_full = summarize_isp_df(long333_df, "Full_AllCells", None, None)
sum333_nonsat = summarize_isp_df(long333_df, "NonSat_05_099", 0.5, 0.99)
sum333_all = pd.concat([sum333_full, sum333_nonsat], ignore_index=True)
sum333_all.to_csv(f"{OUT}/isp2_gse333879_deletion_summary.csv", index=False)
log("GSE333879 ISP Phase 2 Summary (MG_injury_D30 Full & NonSat):\n" +
    sum333_all[sum333_all["arm"] == "MG_injury_D30"].sort_values(["subset", "mean_dP"], ascending=[True, False]).head(30).to_string())

log(f"ALL ROUTE A TASKS COMPLETED SUCCESSFULLY in {time.time() - t0:.1f}s")
