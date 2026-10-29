#!/usr/bin/env python3
"""TBI-GF v2: full-pool MTL fine-tune with bf16 + checkpoint + SDPA (target: 5-6h on 4090D).

v2 changes vs v1 (which ran 13h fp32 then got SIGKILLed at 75%):
  - vendored train.py now has bf16 autocast (2-3x speedup) + periodic checkpoint + auto-resume
  - batch 16 (bf16 halves activation memory)
  - pre-flight host load gate: abort if load > 30
  - all patches pushed to SJT503/geneformer-src commit ecd23e3

Usage (on rental box):
  PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128' python -u train_tbi_gf_full_v2.py \
    --data /root/autodl-tmp/data/tbi_562k.dataset \
    --pretrained /root/autodl-tmp/data/gf_v2_104m \
    --out /root/autodl-tmp/run2 --batch 16
"""
import os, sys, json, time, glob, shutil, subprocess, argparse, traceback

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pretrained", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
RES = os.path.join(args.out, "res"); os.makedirs(RES, exist_ok=True)

# ===== G1: GPU gate =====
import torch
assert torch.cuda.is_available(), "G1 FAIL: no CUDA"
log(f"G1 PASS: {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ===== G0.5: host load gate (v2 — the 606-machine lesson) =====
_load = open('/proc/loadavg').read().split()[0]
log(f"host load: {_load}")
if float(_load) > 30:
    log(f"G0.5 FAIL: host load {_load} > 30 — this machine is oversubscribed, DO NOT train here")
    sys.exit(2)

# ===== vendored auto-patches (idempotent; ecd23e3 already has them, this is belt-and-suspenders) =====
import geneformer
_pkgroot = os.path.dirname(geneformer.__file__)
for _rel in ["mtl/train.py", "mtl/model.py", "mtl/utils.py", "mtl/eval_utils.py"]:
    _p = os.path.join(_pkgroot, _rel)
    if os.path.exists(_p):
        _t = open(_p, encoding="utf-8").read()
        _mod = False
        # SDPA patch
        if 'BertModel.from_pretrained(pretrained_path)' in _t and 'attn_implementation' not in _t and 'model.py' in _rel:
            _t = _t.replace('BertModel.from_pretrained(pretrained_path)',
                           'BertModel.from_pretrained(pretrained_path, attn_implementation="sdpa")'); _mod = True
        # strict=False patches
        _t = _t.replace('best_model.load_state_dict(torch.load(best_model_path))',
                       'best_model.load_state_dict(torch.load(best_model_path), strict=False)')
        _t = _t.replace('model.load_state_dict(torch.load(model_path))',
                       'model.load_state_dict(torch.load(model_path), strict=False)')
        if _mod or 'strict=False' in _t:
            open(_p, "w", encoding="utf-8").write(_t)
log("vendored patches verified")

from geneformer import MTLClassifier
from datasets import load_from_disk
import numpy as np, pandas as pd
log("geneformer imported")

# ===== G2: data gates =====
ds = load_from_disk(args.data)
log(f"loaded {len(ds):,} cells")
assert len(ds) == 562633, f"G2 FAIL: expected 562,633, got {len(ds)}"
ds = ds.filter(lambda x: x["is_dbl"] == 0, num_proc=4)
log(f"after is_dbl==0: {len(ds):,}")
assert len(ds) == 536619, f"G2 FAIL: expected 536,619"
ds = ds.add_column("unique_cell_id", [f"cell_{i}" for i in range(len(ds))])

rng = np.random.RandomState(args.seed)
meta = pd.DataFrame({"ct": ds["cell_type"], "grp": ds["grp"], "dataset": ds["dataset"], "sample": ds["sample"]})
splits = {"train": [], "val": [], "test": []}
for ct, idxs in meta.groupby("ct").indices.items():
    idxs = np.array(sorted(idxs)); rng2 = np.random.RandomState(args.seed); rng2.shuffle(idxs)
    n = len(idxs); n_tr = int(n * 0.8); n_va = int(n * 0.1)
    splits["train"].extend(idxs[:n_tr]); splits["val"].extend(idxs[n_tr:n_tr + n_va]); splits["test"].extend(idxs[n_tr + n_va:])

man_rows = []
for name, idxs in splits.items():
    for i in idxs:
        man_rows.append((name, i, meta['ct'].iloc[i], meta['grp'].iloc[i], meta['dataset'].iloc[i], meta['sample'].iloc[i]))
pd.DataFrame(man_rows, columns=["split", "row", "cell_type", "grp", "dataset", "sample"]).to_csv(
    os.path.join(RES, "split_manifest.csv.gz"), index=False, compression="gzip")
log("split manifest saved")

paths = {}
for name, idxs in splits.items():
    part = ds.select(sorted(idxs))
    p = os.path.join(args.out, f"{name}.dataset"); part.save_to_disk(p); paths[name] = p
    log(f"{name}: {len(part):,} cells")
    assert len(pd.Series(part["grp"]).value_counts()) == 2
log("G2 PASS")

# ===== Fermi estimate =====
steps = int(np.ceil(len(splits["train"]) / args.batch))
# bf16 ~0.4-0.5s/step at batch 16 on 4090D
est_h = steps * 0.5 / 3600
log(f"Fermi: {steps:,} steps @ batch {args.batch} bf16 -> est {est_h:.1f} h")
assert est_h < 12, "estimate exceeds budget"

# ===== G3: train =====
manual = {
    "learning_rate": args.lr, "warmup_ratio": 0.01, "weight_decay": 0.1,
    "dropout_rate": 0.1, "lr_scheduler_type": "cosine",
    "task_weights": [1, 1], "max_layers_to_freeze": 2,
}
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
batch = args.batch
for attempt, batch in enumerate([args.batch, 12, 8]):
    try:
        mc = MTLClassifier(
            task_columns=["cell_type", "grp"], study_name=f"tbi_gf_v2_b{batch}",
            pretrained_path=args.pretrained,
            train_path=paths["train"], val_path=paths["val"], test_path=paths["test"],
            model_save_path=os.path.join(RES, "model"), results_dir=RES,
            tensorboard_log_dir=os.path.join(RES, "tb"),
            manual_hyperparameters=manual, use_manual_hyperparameters=True,
            epochs=args.epochs, batch_size=batch, seed=args.seed,
        )
        log(f"training start (batch {batch}, bf16, attempt {attempt+1})")
        mc.run_manual_tuning()
        log("training done")
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch > 8:
            torch.cuda.empty_cache(); log(f"OOM at batch {batch} — falling back"); continue
        open(os.path.join(RES, "crash_report.txt"), "w").write(traceback.format_exc())
        raise
else:
    raise RuntimeError("all batch sizes OOMed")

# ===== test eval (non-fatal) =====
try:
    mc.load_and_evaluate_test_model()
    log("test evaluation done")
except Exception:
    open(os.path.join(RES, "crash_report_test.txt"), "w").write(traceback.format_exc())
    log("test evaluation failed (non-fatal)")

# ===== save-verify =====
bin_path = os.path.join(RES, "model", "GeneformerMultiTask", "pytorch_model.bin")
sz = os.path.getsize(bin_path) / 1e6 if os.path.exists(bin_path) else 0
log(f"saved model bin: {sz:.1f} MB")
assert sz > 100, f"G3 FAIL: model bin only {sz:.1f} MB"
_sd = torch.load(bin_path, map_location="cpu")
log(f"state_dict keys: {len(_sd)}")
del _sd

# ===== final metrics =====
vp = os.path.join(RES, "manual_run_val_preds.csv")
if os.path.exists(vp):
    df = pd.read_csv(vp)
    a1 = float((df.task1_true_label == df.task1_pred_label).mean())
    a2 = float((df.task2_true_label == df.task2_pred_label).mean())
    metrics = {"task1_cell_type_acc": round(a1, 4), "task2_grp_acc": round(a2, 4),
               "batch": batch, "lr": args.lr, "n_train": len(splits["train"]),
               "verdict": "PASS" if (a1 > 0.5 and a2 > 0.6) else "CHECK",
               "runtime_s": round(time.time() - t0)}
    json.dump(metrics, open(os.path.join(RES, "final_metrics.json"), "w"), indent=1)
    log(f"FINAL METRICS: {json.dumps(metrics)}")

# ===== cleanup big files =====
for _big in [os.path.join(args.out, f"{n}.dataset") for n in ["train", "val", "test"]]:
    if os.path.exists(_big):
        shutil.rmtree(_big, ignore_errors=True)
log("cleanup done")
log(f"ALL DONE ({time.time()-t0:.0f}s)")
