#!/usr/bin/env python3
"""Exam 1 (Temporal Extrapolation): Restricted <=7d fine-tuning + D28 & 6mo held-out evaluation.

Design (Morphology alpha / v2 Framework Exam 1):
  - Source corpus: /root/autodl-tmp/data/tbi_562k.dataset (562,633 tokenized -> 536,619 clean after is_dbl==0)
  - Training/Val/In-domain Test Pool (<=7d + sham/Ctrl):
      Exclude chronic injury endpoints: time_point NOT IN ['D28', '6mo']
      Exact clean cell count = 456,627 cells:
        * GSE269748: 24h (73,468) + 7d (37,963) + Ctrl (53,851) = 165,282
        * GSE271769: D1 (24,799) + D7 (40,796) + NoTBI (21,598) = 87,193
        * GSE277487: group1/2/3 + Ctrl (6h/2d/4d/sham) = 158,641
        * GSE290150: acute injury (20,579) + acute sham (24,932) = 45,511
      Stratified 80/10/10 by cell_type (seed=42):
        * train_le7d = 365,296 cells (1 epoch @ batch 16 bf16 + megabatch length grouping)
        * val_le7d   = 45,660 cells
        * test_le7d  = 45,671 cells
  - Held-Out Chronic Extrapolation Endpoints (NEVER seen during training):
      1) Holdout_D28 (GSE271769 D28 injury, 36,850 cells) + held-out test_le7d GSE271769 NoTBI controls
      2) Holdout_6mo (GSE269748 6mo injury, 43,142 cells) + held-out test_le7d GSE269748 Ctrl controls
      Total chronic held-out injury cells = 79,992 cells.
  - 3-Way Model Comparison on Identical Splits:
      A) TBI-GF-le7d (Newly trained strictly on <=7d train split)
      B) Frozen GF-V2-104M (Pretrained zero-shot baseline, 768-d embedding shift)
      C) TBI-GF-full Oracle (run2, if epoch_end_model.pt is provided via --oracle_pt)
"""
import os, sys, json, time, shutil, pickle, hashlib, argparse, traceback
import numpy as np, pandas as pd

RUN2_FULL_MD5 = "8f5cf1c4f0343132a8b04281504679f2"

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/autodl-tmp/data/tbi_562k.dataset")
ap.add_argument("--pretrained", default="/root/autodl-tmp/data/gf_v2_104m")
ap.add_argument("--oracle_pt", default="/root/autodl-tmp/data/run2_epoch_end_model.pt")
ap.add_argument("--out", default="/root/autodl-tmp/run_exam1_le7d")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--eval_only", action="store_true")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
RES = os.path.join(args.out, "res"); os.makedirs(RES, exist_ok=True)

# ===== G1: GPU gate =====
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu

assert torch.cuda.is_available(), "G1 FAIL: no CUDA"
dev_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
log(f"G1 PASS: {dev_name} | VRAM {vram_gb:.1f} GB | AMP dtype={amp_dtype}")

# ===== G0.5: host load gate =====
if os.path.exists('/proc/loadavg'):
    _load = open('/proc/loadavg').read().split()[0]
    log(f"host load: {_load}")
    if float(_load) > 35:
        log(f"G0.5 FAIL: host load {_load} > 35 — oversubscribed host, aborting")
        sys.exit(2)

# ===== Vendored auto-patches =====
import geneformer
_pkgroot = os.path.dirname(geneformer.__file__)
for _rel in ["mtl/train.py", "mtl/model.py", "mtl/utils.py", "mtl/eval_utils.py"]:
    _p = os.path.join(_pkgroot, _rel)
    if os.path.exists(_p):
        _t = open(_p, encoding="utf-8").read()
        _mod = False
        if 'BertModel.from_pretrained(pretrained_path)' in _t and 'attn_implementation' not in _t and 'model.py' in _rel:
            _t = _t.replace('BertModel.from_pretrained(pretrained_path)',
                           'BertModel.from_pretrained(pretrained_path, attn_implementation="sdpa")'); _mod = True
        if _mod:
            open(_p, "w", encoding="utf-8").write(_t)
log("vendored SDPA patch verified")

from geneformer.mtl.model import GeneformerMultiTask
from geneformer.mtl.utils import set_seed
from datasets import load_from_disk

set_seed(args.seed)

# ===== G2: Data & Temporal Split Gates =====
ds = load_from_disk(args.data)
assert len(ds) == 562633, f"G2 FAIL: expected 562,633 total cells, got {len(ds)}"
ds = ds.filter(lambda x: x["is_dbl"] == 0, num_proc=4)
assert len(ds) == 536619, f"G2 FAIL: expected 536,619 clean cells, got {len(ds)}"

meta = pd.DataFrame({
    "ct": ds["cell_type"],
    "grp": ds["grp"],
    "tp": ds["time_point"],
    "dataset": ds["dataset"],
    "sample": ds["sample"],
    "length": ds["length"] if "length" in ds.column_names else [len(x) for x in ds["input_ids"]],
})

CHRONIC_TPS = {"D28", "6mo"}
is_chronic = meta["tp"].isin(CHRONIC_TPS)
idx_d28 = np.where(meta["tp"] == "D28")[0]
idx_6mo = np.where(meta["tp"] == "6mo")[0]
idx_le7d = np.where(~is_chronic)[0]

log(f"Temporal partition: <=7d pool = {len(idx_le7d):,} | D28 holdout = {len(idx_d28):,} | 6mo holdout = {len(idx_6mo):,}")
assert len(idx_le7d) == 456627, f"G2 FAIL: expected 456,627 <=7d cells, got {len(idx_le7d)}"
assert len(idx_d28) == 36850, f"G2 FAIL: expected 36,850 D28 cells, got {len(idx_d28)}"
assert len(idx_6mo) == 43142, f"G2 FAIL: expected 43,142 6mo cells, got {len(idx_6mo)}"

# Exact deterministic label mappings (identical to StreamingMultiTaskDataset sorted unique values)
labels1 = sorted(meta["ct"].unique().tolist())
labels2 = sorted(meta["grp"].unique().tolist())
assert len(labels1) == 14 and labels2 == ["injury", "sham"], f"Unexpected labels: {labels1}, {labels2}"
map1 = {k: i for i, k in enumerate(labels1)}
map2 = {k: i for i, k in enumerate(labels2)}
INJ = map2["injury"]
pickle.dump({"task1": map1, "task2": map2}, open(os.path.join(RES, "task_label_mappings.pkl"), "wb"))

# Stratified 80/10/10 split within <=7d pool
meta_le7d = meta.iloc[idx_le7d].copy()
meta_le7d["orig_row"] = idx_le7d
splits = {"train": [], "val": [], "test_le7d": []}
for ct, grp_df in meta_le7d.groupby("ct"):
    rows = np.array(sorted(grp_df["orig_row"].values))
    rng2 = np.random.RandomState(args.seed)
    rng2.shuffle(rows)
    n = len(rows); n_tr = int(n * 0.8); n_va = int(n * 0.1)
    splits["train"].extend(rows[:n_tr])
    splits["val"].extend(rows[n_tr:n_tr + n_va])
    splits["test_le7d"].extend(rows[n_tr + n_va:])

for k in splits:
    splits[k] = sorted(splits[k])

log(f"<=7d splits: train={len(splits['train']):,} | val={len(splits['val']):,} | test_le7d={len(splits['test_le7d']):,}")
assert len(splits["train"]) + len(splits["val"]) + len(splits["test_le7d"]) == 456627, f"Expected 456,627 total <=7d cells"

# Save manifest (vectorized, <1s)
man_path = os.path.join(RES, "exam1_split_manifest.csv.gz")
if not os.path.exists(man_path):
    split_col = np.empty(len(meta), dtype=object)
    split_col[splits["train"]] = "train"
    split_col[splits["val"]] = "val"
    split_col[splits["test_le7d"]] = "test_le7d"
    split_col[idx_d28] = "holdout_D28"
    split_col[idx_6mo] = "holdout_6mo"
    pd.DataFrame({
        "split": split_col,
        "row": np.arange(len(meta), dtype=np.int64),
        "cell_type": meta["ct"].values,
        "grp": meta["grp"].values,
        "time_point": meta["tp"].values,
        "dataset": meta["dataset"].values,
        "sample": meta["sample"].values,
    }).to_csv(man_path, index=False, compression="gzip")
log("exam1_split_manifest.csv.gz ready")

# Pre-encode label arrays for all 536,619 cells
all_lab1 = np.array([map1[c] for c in meta["ct"]], dtype=np.int64)
all_lab2 = np.array([map2[g] for g in meta["grp"]], dtype=np.int64)
all_len  = meta["length"].values.astype(np.int32)

model_pt = os.path.join(args.out, "epoch_end_model_le7d.pt")
ckpt_pt  = os.path.join(args.out, "ckpt_exam1_le7d.pt")

# ===== G3: Direct Megabatch-Length-Grouped Fine-Tuning (Identical Architecture & Hyperparams to Run 2) =====
if not args.eval_only and not os.path.exists(model_pt):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_ds = ds.select(splits["train"])
    n_train = len(train_ds)
    train_lab1 = all_lab1[splits["train"]]
    train_lab2 = all_lab2[splits["train"]]
    train_len  = all_len[splits["train"]]

    # Build megabatch length-grouped batch indices (megabatch = 50 * batch_size)
    # Preserves stochastic SGD while matching sequence lengths inside each mini-batch (3x faster, zero wasted 4096-padding)
    B = args.batch
    rng_tr = np.random.RandomState(args.seed)
    perm = rng_tr.permutation(n_train)
    mega_sz = B * 50
    batches = []
    for s in range(0, n_train, mega_sz):
        chunk = perm[s:s + mega_sz]
        # Sort by descending length within megabatch
        chunk_sorted = chunk[np.argsort(-train_len[chunk])]
        for b_start in range(0, len(chunk_sorted), B):
            batches.append(chunk_sorted[b_start:b_start + B])
    # Shuffle mini-batch order deterministically, putting a max-length (4096-token) batch at step 0 to pre-warm VRAM allocator
    batch_order = rng_tr.permutation(len(batches))
    batches = [batches[i] for i in batch_order]
    max_b_idx = int(np.argmax([train_len[b[0]] for b in batches]))
    batches[0], batches[max_b_idx] = batches[max_b_idx], batches[0]
    total_steps = len(batches) * args.epochs
    log(f"Built {len(batches):,} length-grouped mini-batches (batch={B}, total_steps={total_steps:,}, step0_maxlen={train_len[batches[0][0]]})")

    model = GeneformerMultiTask(
        args.pretrained,
        [len(labels1), len(labels2)],
        dropout_rate=0.1,
        use_task_weights=True,
        task_weights=[1, 1],
        max_layers_to_freeze=2,
        use_attention_pooling=True,
    ).cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    warmup_steps = int(total_steps * 0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(not use_bf16))

    start_step = 0
    running_loss = 0.0
    if os.path.exists(ckpt_pt):
        try:
            ck = torch.load(ckpt_pt, map_location="cuda")
            model.load_state_dict(ck["model"], strict=False)
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_step = ck["step"]
            running_loss = ck.get("running_loss", 0.0)
            log(f"[RESUME] Loaded checkpoint at step {start_step}/{total_steps}")
        except Exception as e:
            log(f"[RESUME WARN] Failed to load {ckpt_pt}: {e}; starting from step 0")

    model.train()
    optimizer.zero_grad()
    t_train_start = time.time()

    for step_idx in range(start_step, len(batches)):
        b_idxs = batches[step_idx].tolist()
        recs = train_ds[b_idxs]["input_ids"]
        seqs = [torch.tensor(x, dtype=torch.long) for x in recs]
        lens = torch.tensor([len(x) for x in recs], dtype=torch.long)
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=0).cuda(non_blocking=True)
        attn_mask = (torch.arange(input_ids.shape[1], device="cuda")[None, :] < lens.cuda(non_blocking=True)[:, None]).long()
        y1 = torch.tensor(train_lab1[b_idxs], dtype=torch.long, device="cuda")
        y2 = torch.tensor(train_lab2[b_idxs], dtype=torch.long, device="cuda")

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss, _, _ = model(input_ids, attn_mask, [y1, y2])

        if use_bf16:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        done_steps = step_idx + 1

        if done_steps in (10, 50, 200) or done_steps % 500 == 0 or done_steps == len(batches):
            elapsed = time.time() - t_train_start
            steps_done_session = done_steps - start_step
            rate = steps_done_session / max(elapsed, 1e-3)
            eta_min = (len(batches) - done_steps) / max(rate, 1e-5) / 60.0
            avg_l = running_loss / done_steps
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            log(f"Step {done_steps:5d}/{len(batches)} ({100*done_steps/len(batches):5.1f}%) | avg_loss={avg_l:.4f} | {rate:.2f} it/s | ETA={eta_min:.1f}m | maxVRAM={vram_used:.2f}GB")

        if done_steps % 2000 == 0:
            tmp_ck = ckpt_pt + ".tmp"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": done_steps,
                "running_loss": running_loss,
            }, tmp_ck)
            os.replace(tmp_ck, ckpt_pt)

    # Save final epoch_end_model_le7d.pt IMMEDIATELY before evaluation (defa241 lesson)
    torch.save(model.state_dict(), model_pt)
    if os.path.exists(ckpt_pt):
        os.remove(ckpt_pt)
    log(f"Saved trained <=7d model to {model_pt} ({os.path.getsize(model_pt)/1e6:.1f} MB)")
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

# ===== Verify Trained Weights =====
assert os.path.exists(model_pt) and os.path.getsize(model_pt) > 100e6, f"G3 FAIL: missing {model_pt}"
pt_md5 = hashlib.md5(open(model_pt, "rb").read()).hexdigest()
log(f"Evaluating Exam 1 <=7d weights: {model_pt} (md5={pt_md5})")
assert pt_md5 != RUN2_FULL_MD5, f"G3 FAIL: {model_pt} matches Run 2 full-data model MD5!"

# ===== Fast Single-Pass Evaluation Function (Task Heads + 768-d Mean-Pooled Hidden Embedding) =====
def evaluate_mtl_model(weights_path, model_tag, eval_indices):
    m = GeneformerMultiTask(args.pretrained, [len(labels1), len(labels2)], use_attention_pooling=True)
    sd = torch.load(weights_path, map_location="cpu")
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, f"Weight mismatch for {model_tag}: {miss} / {unexp}"
    m = m.cuda().eval()

    sub = ds.select(eval_indices)
    n = len(sub)
    lens = all_len[eval_indices]
    order = np.argsort(-lens)
    B_eval = 16

    p1 = np.zeros(n, dtype=np.int32)
    p2 = np.zeros(n, dtype=np.int32)
    pr_inj = np.zeros(n, dtype=np.float32)
    embs = np.zeros((n, 768), dtype=np.float32)

    t_ev = time.time()
    with torch.no_grad():
        for s in range(0, n, B_eval):
            b_pos = order[s:s + B_eval]
            recs = sub[b_pos.tolist()]["input_ids"]
            seqs = [torch.tensor(x, dtype=torch.long) for x in recs]
            l_b  = torch.tensor([len(x) for x in recs], dtype=torch.long, device="cuda")
            input_ids = pad_sequence(seqs, batch_first=True, padding_value=0).cuda(non_blocking=True)
            attn_mask = (torch.arange(input_ids.shape[1], device="cuda")[None, :] < l_b[:, None]).long()

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                bert_out = m.bert(input_ids=input_ids, attention_mask=attn_mask)
                last_h = bert_out.last_hidden_state
                pooled = m.attention_pool(last_h) if m.use_attention_pooling else last_h[:, 0, :]
                logits1 = m.classification_heads[0](pooled).float()
                logits2 = m.classification_heads[1](pooled).float()

                mask_f = attn_mask.unsqueeze(-1).float()
                mean_h = (last_h.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                mean_h = F.normalize(mean_h, p=2, dim=-1)

            p1[b_pos] = logits1.argmax(dim=-1).cpu().numpy()
            p2[b_pos] = logits2.argmax(dim=-1).cpu().numpy()
            pr_inj[b_pos] = torch.softmax(logits2, dim=-1)[:, INJ].cpu().numpy()
            embs[b_pos] = mean_h.cpu().numpy()

            if (s // B_eval) % 400 == 0 or s + B_eval >= n:
                log(f"  [{model_tag}] eval {min(s+B_eval, n):,}/{n:,} ({time.time()-t_ev:.1f}s)")

    del m
    torch.cuda.empty_cache()
    return p1, p2, pr_inj, embs

def evaluate_frozen_bert(eval_indices):
    fb = BertModel.from_pretrained(args.pretrained, attn_implementation="sdpa").cuda().eval()
    sub = ds.select(eval_indices)
    n = len(sub)
    lens = all_len[eval_indices]
    order = np.argsort(-lens)
    B_eval = 16
    embs = np.zeros((n, 768), dtype=np.float32)
    t_ev = time.time()
    with torch.no_grad():
        for s in range(0, n, B_eval):
            b_pos = order[s:s + B_eval]
            recs = sub[b_pos.tolist()]["input_ids"]
            seqs = [torch.tensor(x, dtype=torch.long) for x in recs]
            l_b  = torch.tensor([len(x) for x in recs], dtype=torch.long, device="cuda")
            input_ids = pad_sequence(seqs, batch_first=True, padding_value=0).cuda(non_blocking=True)
            attn_mask = (torch.arange(input_ids.shape[1], device="cuda")[None, :] < l_b[:, None]).long()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = fb(input_ids=input_ids, attention_mask=attn_mask)
                mask_f = attn_mask.unsqueeze(-1).float()
                mean_h = (out.last_hidden_state.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                mean_h = F.normalize(mean_h, p=2, dim=-1)
            embs[b_pos] = mean_h.cpu().numpy()
            if (s // B_eval) % 400 == 0 or s + B_eval >= n:
                log(f"  [Frozen_GF_V2] eval {min(s+B_eval, n):,}/{n:,} ({time.time()-t_ev:.1f}s)")
    del fb
    torch.cuda.empty_cache()
    return embs

# Combined evaluation pool = test_le7d (45,671) + holdout_D28 (36,850) + holdout_6mo (43,142) = 125,663 cells
eval_indices = splits["test_le7d"] + idx_d28.tolist() + idx_6mo.tolist()
split_tags = (
    ["test_le7d"] * len(splits["test_le7d"])
    + ["holdout_D28"] * len(idx_d28)
    + ["holdout_6mo"] * len(idx_6mo)
)
log(f"Starting combined evaluation across {len(eval_indices):,} held-out cells...")

p1_le7d, p2_le7d, pr_le7d, emb_le7d = evaluate_mtl_model(model_pt, "TBI_GF_le7d", eval_indices)
emb_frozen = evaluate_frozen_bert(eval_indices)

has_oracle = os.path.exists(args.oracle_pt) and os.path.getsize(args.oracle_pt) > 100e6
if has_oracle:
    p1_ora, p2_ora, pr_ora, emb_ora = evaluate_mtl_model(args.oracle_pt, "TBI_GF_full_Oracle", eval_indices)
else:
    p1_ora = p2_ora = pr_ora = emb_ora = None

# Compute per-cell-type Sham centroids from test_le7d Sham cells for 768-d embedding shift scoring
eval_df = meta.iloc[eval_indices].copy().reset_index(drop=True)
eval_df["orig_row"] = eval_indices
eval_df["split"] = split_tags
eval_df["true1"] = all_lab1[eval_indices]
eval_df["true2"] = all_lab2[eval_indices]
eval_df["y_injury"] = (eval_df["grp"] == "injury").astype(int)
eval_df["pred1_le7d"] = p1_le7d
eval_df["pred2_le7d"] = p2_le7d
eval_df["P_injury_le7d"] = np.round(pr_le7d, 5)

def compute_emb_shift_to_sham(embs_mat, df_meta):
    shifts = np.zeros(len(df_meta), dtype=np.float32)
    sham_mask_global = (df_meta["split"] == "test_le7d") & (df_meta["grp"] == "sham")
    global_centroid = embs_mat[sham_mask_global.values].mean(axis=0)
    global_centroid /= max(np.linalg.norm(global_centroid), 1e-8)
    for ct, idxs in df_meta.groupby("ct").indices.items():
        ct_sham = df_meta.iloc[idxs]
        ct_sham_rows = idxs[(ct_sham["split"] == "test_le7d").values & (ct_sham["grp"] == "sham").values]
        if len(ct_sham_rows) >= 5:
            c_vec = embs_mat[ct_sham_rows].mean(axis=0)
            c_vec /= max(np.linalg.norm(c_vec), 1e-8)
        else:
            c_vec = global_centroid
        cos_sim = embs_mat[idxs] @ c_vec
        shifts[idxs] = 1.0 - cos_sim
    return np.round(shifts, 6)

eval_df["emb_shift_le7d"] = compute_emb_shift_to_sham(emb_le7d, eval_df)
eval_df["emb_shift_frozen"] = compute_emb_shift_to_sham(emb_frozen, eval_df)
if has_oracle:
    eval_df["pred1_oracle"] = p1_ora
    eval_df["pred2_oracle"] = p2_ora
    eval_df["P_injury_oracle"] = np.round(pr_ora, 5)
    eval_df["emb_shift_oracle"] = compute_emb_shift_to_sham(emb_ora, eval_df)

eval_df.to_csv(os.path.join(RES, "exam1_all_eval_preds.csv.gz"), index=False, compression="gzip")
log("Saved exam1_all_eval_preds.csv.gz")

# ===== Build Comprehensive Cohort & Cell-Type Extrapolation Tables =====
# Define 3 evaluation cohorts:
# 1) InDomain_le7d: split == 'test_le7d' (45,671 cells: acute/subacute injury vs sham)
# 2) Extrap_D28_GSE271769: holdout_D28 (36,850 D28 injury cells) + test_le7d GSE271769 NoTBI controls
# 3) Extrap_6mo_GSE269748: holdout_6mo (43,142 6mo injury cells) + test_le7d GSE269748 Ctrl controls
cohorts = {
    "InDomain_le7d": eval_df["split"] == "test_le7d",
    "Extrap_D28_GSE271769": (eval_df["split"] == "holdout_D28") | ((eval_df["split"] == "test_le7d") & (eval_df["dataset"] == "GSE271769") & (eval_df["grp"] == "sham")),
    "Extrap_6mo_GSE269748": (eval_df["split"] == "holdout_6mo") | ((eval_df["split"] == "test_le7d") & (eval_df["dataset"] == "GSE269748") & (eval_df["grp"] == "sham")),
}

summary_rows = []
sample_rows = []

for coh_name, mask in cohorts.items():
    cdf = eval_df[mask].copy()
    # Sample-level summary
    for (smp, grp_lbl), sdf in cdf.groupby(["sample", "grp"]):
        row_s = {
            "cohort": coh_name, "sample": smp, "grp": grp_lbl, "y_injury": int(grp_lbl == "injury"),
            "n_cells": len(sdf),
            "ct_acc_le7d": round(float((sdf["true1"] == sdf["pred1_le7d"]).mean()), 4),
            "mean_P_injury_le7d": round(float(sdf["P_injury_le7d"].mean()), 4),
            "mean_emb_shift_le7d": round(float(sdf["emb_shift_le7d"].mean()), 5),
            "mean_emb_shift_frozen": round(float(sdf["emb_shift_frozen"].mean()), 5),
        }
        if has_oracle:
            row_s["ct_acc_oracle"] = round(float((sdf["true1"] == sdf["pred1_oracle"]).mean()), 4)
            row_s["mean_P_injury_oracle"] = round(float(sdf["P_injury_oracle"].mean()), 4)
            row_s["mean_emb_shift_oracle"] = round(float(sdf["emb_shift_oracle"].mean()), 5)
        sample_rows.append(row_s)

    for ct_name in ["ALL"] + labels1:
        sub_c = cdf if ct_name == "ALL" else cdf[cdf["ct"] == ct_name]
        if len(sub_c) == 0:
            continue
        inj_c = sub_c[sub_c["y_injury"] == 1]
        shm_c = sub_c[sub_c["y_injury"] == 0]
        ct_acc = float((sub_c["true1"] == sub_c["pred1_le7d"]).mean())
        grp_acc = float((sub_c["true2"] == sub_c["pred2_le7d"]).mean())
        inj_recall = float((inj_c["true2"] == inj_c["pred2_le7d"]).mean()) if len(inj_c) > 0 else np.nan

        if len(inj_c) >= 5 and len(shm_c) >= 5:
            auroc_prob = float(roc_auc_score(sub_c["y_injury"], sub_c["P_injury_le7d"]))
            auprc_prob = float(average_precision_score(sub_c["y_injury"], sub_c["P_injury_le7d"]))
            auroc_emb_le7d = float(roc_auc_score(sub_c["y_injury"], sub_c["emb_shift_le7d"]))
            auroc_emb_froz = float(roc_auc_score(sub_c["y_injury"], sub_c["emb_shift_frozen"]))
            _, p_mwu = mannwhitneyu(inj_c["P_injury_le7d"], shm_c["P_injury_le7d"], alternative="two-sided")
        else:
            auroc_prob = auprc_prob = auroc_emb_le7d = auroc_emb_froz = p_mwu = np.nan

        row = {
            "cohort": coh_name, "cell_type": ct_name,
            "n_total": len(sub_c), "n_injury": len(inj_c), "n_sham": len(shm_c),
            "ct_acc_le7d": round(ct_acc, 4),
            "grp_acc_le7d": round(grp_acc, 4),
            "injury_recall_le7d": round(inj_recall, 4) if not np.isnan(inj_recall) else np.nan,
            "mean_P_injury_TBI": round(float(inj_c["P_injury_le7d"].mean()), 4) if len(inj_c) else np.nan,
            "mean_P_injury_Sham": round(float(shm_c["P_injury_le7d"].mean()), 4) if len(shm_c) else np.nan,
            "AUROC_prob_le7d": round(auroc_prob, 4) if not np.isnan(auroc_prob) else np.nan,
            "AUPRC_prob_le7d": round(auprc_prob, 4) if not np.isnan(auprc_prob) else np.nan,
            "AUROC_emb_le7d": round(auroc_emb_le7d, 4) if not np.isnan(auroc_emb_le7d) else np.nan,
            "AUROC_emb_frozen": round(auroc_emb_froz, 4) if not np.isnan(auroc_emb_froz) else np.nan,
            "uplift_emb_vs_frozen": round(auroc_emb_le7d - auroc_emb_froz, 4) if not np.isnan(auroc_emb_le7d) else np.nan,
            "mwu_p_prob": float(p_mwu) if not np.isnan(p_mwu) else np.nan,
        }
        if has_oracle:
            row["ct_acc_oracle"] = round(float((sub_c["true1"] == sub_c["pred1_oracle"]).mean()), 4)
            row["injury_recall_oracle"] = round(float((inj_c["true2"] == inj_c["pred2_oracle"]).mean()), 4) if len(inj_c) else np.nan
            if len(inj_c) >= 5 and len(shm_c) >= 5:
                row["AUROC_prob_oracle"] = round(float(roc_auc_score(sub_c["y_injury"], sub_c["P_injury_oracle"])), 4)
                row["AUROC_emb_oracle"] = round(float(roc_auc_score(sub_c["y_injury"], sub_c["emb_shift_oracle"])), 4)
        summary_rows.append(row)

df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(RES, "exam1_extrapolation_by_celltype.csv"), index=False)
df_smp = pd.DataFrame(sample_rows)
df_smp.to_csv(os.path.join(RES, "exam1_extrapolation_by_sample.csv"), index=False)

# 5-Trajectory Consensus Table (T1..T5) across the 3 Evaluation Arms
traj_rows = []
for coh_name, mask in cohorts.items():
    cdf = eval_df[mask].copy().reset_index(drop=True)
    for t_name, t_seed, t_glia_only in [
        ("T1_Full", None, False),
        ("T2_MajorGlia", None, True),
        ("T3_Bootstrap_S42", 42, False),
        ("T4_Bootstrap_S101", 101, False),
        ("T5_Bootstrap_S202", 202, False),
    ]:
        sub_t = cdf[cdf["ct"].isin(["Microglia", "Astrocyte", "Oligo", "OPC"])] if t_glia_only else cdf
        if t_seed is not None:
            rng_t = np.random.default_rng(t_seed)
            idx_b = rng_t.choice(len(sub_t), size=len(sub_t), replace=True)
            sub_t = sub_t.iloc[idx_b]
        auroc_p = float(roc_auc_score(sub_t["y_injury"], sub_t["P_injury_le7d"]))
        auroc_e_le7d = float(roc_auc_score(sub_t["y_injury"], sub_t["emb_shift_le7d"]))
        auroc_e_froz = float(roc_auc_score(sub_t["y_injury"], sub_t["emb_shift_frozen"]))
        row_t = {
            "cohort": coh_name,
            "trajectory": t_name,
            "n_cells": len(sub_t),
            "ct_acc_le7d": round(float((sub_t["true1"] == sub_t["pred1_le7d"]).mean()), 4),
            "grp_acc_le7d": round(float((sub_t["true2"] == sub_t["pred2_le7d"]).mean()), 4),
            "AUROC_prob_le7d": round(auroc_p, 4),
            "AUROC_emb_le7d": round(auroc_e_le7d, 4),
            "AUROC_emb_frozen": round(auroc_e_froz, 4),
        }
        if has_oracle:
            row_t["AUROC_prob_oracle"] = round(float(roc_auc_score(sub_t["y_injury"], sub_t["P_injury_oracle"])), 4)
            row_t["AUROC_emb_oracle"] = round(float(roc_auc_score(sub_t["y_injury"], sub_t["emb_shift_oracle"])), 4)
        traj_rows.append(row_t)

df_traj = pd.DataFrame(traj_rows)
df_traj.to_csv(os.path.join(RES, "exam1_5traj_consensus.csv"), index=False)

all_only = df_sum[df_sum["cell_type"] == "ALL"].set_index("cohort").to_dict(orient="index")
final_json = {
    "model_md5": pt_md5,
    "n_train_le7d": len(splits["train"]),
    "n_val_le7d": len(splits["val"]),
    "n_test_le7d": len(splits["test_le7d"]),
    "n_holdout_D28": len(idx_d28),
    "n_holdout_6mo": len(idx_6mo),
    "cohorts_ALL": all_only,
    "runtime_s": round(time.time() - t0, 1),
}
json.dump(final_json, open(os.path.join(RES, "exam1_final_metrics.json"), "w"), indent=2)
log("EXAM 1 COMPLETE: " + json.dumps(final_json))
