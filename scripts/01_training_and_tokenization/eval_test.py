#!/usr/bin/env python3
"""Standalone test-set evaluation for run2 epoch_end_model.pt.
Replaces the official validate_model path (which SIGSEGVs on this host).
Readouts: task1 (cell_type) / task2 (grp) accuracy on the deterministic test split,
per-cell-type grp accuracy, confusion summaries -> final_metrics.json + test_preds.csv.gz
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

DATA = '/root/autodl-tmp/data/tbi_562k.dataset'
PRE = '/root/autodl-tmp/data/gf_v2_104m'
MODEL = '/root/autodl-tmp/epoch_end_model.pt'
OUT = '/root/autodl-tmp/eval_out'; os.makedirs(OUT, exist_ok=True)

import torch
assert torch.cuda.is_available(); log(f"GPU {torch.cuda.get_device_name(0)}")

from datasets import load_from_disk
from geneformer.mtl.model import GeneformerMultiTask
import pickle

# ---- labels (order must match training task_columns=["cell_type","grp"]) ----
maps = pickle.load(open('/root/autodl-tmp/run2/res/task_label_mappings.pkl','rb'))
log(f"task_label_mappings type={type(maps)}")
# structure probe
if isinstance(maps, dict):
    for k, v in list(maps.items())[:4]: log(f"  map: {k} -> {str(v)[:100]}")
# build explicit label lists
def as_list(x):
    if isinstance(x, dict): return sorted(x, key=lambda k: x[k]) if all(isinstance(v,int) for v in x.values()) else list(x)
    if isinstance(x, (list, tuple)): return list(x)
    raise ValueError(str(type(x)))
try:
    labels1, labels2 = as_list(maps['task1']), as_list(maps['task2'])
except Exception:
    labels1, labels2 = as_list(maps['cell_type']), as_list(maps['grp'])
log(f"task1 labels n={len(labels1)}: {labels1}")
log(f"task2 labels n={len(labels2)}: {labels2}")

# ---- deterministic split (identical to training: seed42 groupby cell_type 80/10/10) ----
ds = load_from_disk(DATA)
ds = ds.filter(lambda x: x['is_dbl'] == 0, num_proc=4)
rng = np.random.RandomState(42)
meta = pd.DataFrame({'ct': ds['cell_type']})
test_idx = []
for ct, idxs in meta.groupby('ct').indices.items():
    idxs = np.array(sorted(idxs)); r2 = np.random.RandomState(42); r2.shuffle(idxs)
    n = len(idxs); n_tr = int(n*0.8); n_va = int(n*0.1)
    test_idx.extend(sorted(idxs[n_tr+n_va:]))
test = ds.select(test_idx)
log(f"test cells: {len(test):,} (expect 53,674-ish)")
lab1 = np.array([labels1.index(c) for c in test['cell_type']])
lab2 = np.array([labels2.index(g) for g in test['grp']])
cell_types = np.array(test['cell_type']); grps = np.array(test['grp'])
datasets_col = np.array(test['dataset'])

# ---- model ----
model = GeneformerMultiTask(PRE, [len(labels1), len(labels2)], use_attention_pooling=True)
sd = torch.load(MODEL, map_location='cpu')
missing, unexpected = model.load_state_dict(sd, strict=False)
log(f"state_dict: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
assert len(unexpected) == 0 and len(missing) == 0, f"key mismatch: miss={missing[:3]} unexp={unexpected[:3]}"
model = model.cuda().eval()

# ---- forward ----
B = 32
p1 = np.zeros(len(test), dtype=np.int32); p2 = np.zeros(len(test), dtype=np.int32)
pr2 = np.zeros(len(test), dtype=np.float32)  # P(injury)=P(grp class idx of 'injury')
INJ = labels2.index('injury') if 'injury' in labels2 else 1
from torch.nn.utils.rnn import pad_sequence
ids_col = test['input_ids']
with torch.no_grad():
    for s in range(0, len(test), B):
        batch = ids_col[s:s+B]
        seq = [torch.tensor(x, dtype=torch.long) for x in batch]
        input_ids = pad_sequence(seq, batch_first=True, padding_value=0).cuda()
        attn = (torch.arange(input_ids.shape[1])[None,:] < torch.tensor([len(x) for x in batch])[:,None]).long().cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, logits, _ = model(input_ids, attn)  # returns (0, [t1_logits, t2_logits], same) when labels=None
        l1 = logits[0].float().argmax(-1).cpu().numpy(); l2 = logits[1].float().argmax(-1).cpu().numpy()
        p1[s:s+len(l1)] = l1; p2[s:s+len(l2)] = l2
        pr2[s:s+len(l2)] = torch.softmax(logits[1].float(), -1)[:, INJ].cpu().numpy()
        if (s // B) % 200 == 0: log(f"  {s:,}/{len(test):,}")

acc1 = float((p1 == lab1).mean()); acc2 = float((p2 == lab2).mean())
log(f"task1 cell_type acc = {acc1:.4f}")
log(f"task2 grp acc       = {acc2:.4f}")

# per-cell-type grp accuracy + P(injury) means
rows = []
for ct in sorted(set(cell_types)):
    m = cell_types == ct
    for g in ['injury','sham']:
        mm = m & (grps == g)
        if mm.sum() > 0:
            rows.append({'cell_type': ct, 'grp': g, 'n': int(mm.sum()),
                         'grp_acc': round(float((p2[mm]==lab2[mm]).mean()),4),
                         'mean_P_injury': round(float(pr2[mm].mean()),4)})
perct = pd.DataFrame(rows); perct.to_csv(f'{OUT}/grp_by_celltype.csv', index=False)
pd.DataFrame({'cell_type': cell_types, 'grp': grps, 'dataset': datasets_col,
              'true1': lab1, 'pred1': p1, 'true2': lab2, 'pred2': p2, 'P_injury': pr2}
            ).to_csv(f'{OUT}/test_preds.csv.gz', index=False, compression='gzip')

# confusion (task1) compact
cm = pd.crosstab(pd.Series(lab1, name='true').map(dict(enumerate(labels1))),
                 pd.Series(p1, name='pred').map(dict(enumerate(labels1))))
cm.to_csv(f'{OUT}/confusion_task1.csv')

metrics = {'task1_cell_type_acc': round(acc1,4), 'task2_grp_acc': round(acc2,4),
           'n_test': len(test), 'model': 'epoch_end_model.pt',
           'n_params_tensors': len(sd), 'eval_time_s': round(time.time()-t0)}
json.dump(metrics, open(f'{OUT}/final_metrics.json','w'), indent=1)
log("FINAL METRICS: " + json.dumps(metrics))
log("EVAL DONE")
