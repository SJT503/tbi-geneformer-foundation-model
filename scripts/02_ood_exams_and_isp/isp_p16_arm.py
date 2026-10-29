#!/usr/bin/env python3
"""ISP Phase 1: who maintains the injury (p16-arm) signature in microglia?
Design (deletion-only, Pbx1/NatProtoc-standard):
  - Cells : injury Microglia stratified by dataset x time_point (cap 120/stratum, <=800)
            + injury Astrocyte arm (<=240, specificity contrast)
            + sham Microglia reference (<=120, baseline sanity)
  - Genes : 24 Paper-1-anchored candidates (human ortholog symbols; resolved against
            token bank on load, hits/misses reported) + per-cell 60 random expressed
            tokens as empirical null
  - Readout: dP_injury = P_injury(orig) - P_injury(delete gene X), task2 head;
             task1-flip counts as identity-stability side metric
  - Stats : per gene, mean dP among cells EXPRESSING the token, presence rate,
            z vs per-cell null, two-sided empirical p, BH-FDR; stratified by time_point
Outputs: isp_cells_long.csv.gz, isp_gene_summary.csv, isp_metrics.json
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd, torch

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

DATA = '/root/autodl-tmp/data/tbi_562k.dataset'
PRE = '/root/autodl-tmp/data/gf_v2_104m'
MODEL = '/root/autodl-tmp/epoch_end_model.pt'
OUT = '/root/autodl-tmp/isp_out'; os.makedirs(OUT, exist_ok=True)
N_NULL = 60
SEED = 42

# Paper-1-anchored candidates (extracted from R8.6 Manuscript_FULL.md gene inventory)
PAPER1_CORE = ['CDKN2A','CDKN1A','MCL1','CST7','ITGAX','STAT1','STAT2','IRF7','MIF',
               'CD9','LPL','SPP1','BAX','BCL2','IL1B','E2F1']
IDENTITY_ANCHORS = ['P2RY12','TMEM119','HEXB','CSF1R','AIF1','APOE','TREM2']
CANDIDATES = PAPER1_CORE + IDENTITY_ANCHORS

assert torch.cuda.is_available(); log(f"GPU {torch.cuda.get_device_name(0)}")
from datasets import load_from_disk
from geneformer.mtl.model import GeneformerMultiTask
from torch.nn.utils.rnn import pad_sequence

# ---- resolve gene symbols -> token ids ----
name_id = pickle.load(open(f'{PRE}/gene_name_id_dict_gc104M.pkl','rb'))
tok_dict = pickle.load(open(f'{PRE}/token_dictionary_gc104M.pkl','rb'))
log(f"name_id type={type(name_id)} n={len(name_id)}; tok_dict n={len(tok_dict)}")
sample = list(name_id.items())[:3] if isinstance(name_id, dict) else name_id[:3]
log(f"name_id sample: {sample}")

def resolve(sym):
    """symbol -> (name_id) ENSG -> (tok_dict) token id; direct-int shortcut first."""
    for d in (name_id, tok_dict):
        if isinstance(d, dict) and sym in d and isinstance(d[sym], int):
            return d[sym]
    if isinstance(name_id, dict) and sym in name_id:
        ens = name_id[sym]
        if isinstance(tok_dict, dict) and isinstance(tok_dict.get(ens), int):
            return tok_dict[ens]
    return None

gene_tok = {}
for g in CANDIDATES:
    t = resolve(g) or resolve(g.capitalize()) or resolve(g.lower())  # human sym / mouse sym / lowercase
    gene_tok[g] = t
hits = {g: t for g, t in gene_tok.items() if t is not None}
misses = [g for g, t in gene_tok.items() if t is None]
log(f"token resolution: {len(hits)}/{len(CANDIDATES)} hit; missing: {misses}")
assert len(hits) >= 18, "too many candidates unresolvable — inspect dictionaries"

# ---- model + labels ----
maps = pickle.load(open('/root/autodl-tmp/run2/res/task_label_mappings.pkl','rb'))
def as_list(x):
    if isinstance(x, dict):
        return sorted(x, key=lambda k: x[k]) if all(isinstance(v,int) for v in x.values()) else list(x)
    return list(x)
try: labels1, labels2 = as_list(maps['task1']), as_list(maps['task2'])
except Exception: labels1, labels2 = as_list(maps['cell_type']), as_list(maps['grp'])
INJ = labels2.index('injury'); MGL = labels1.index('Microglia')
model = GeneformerMultiTask(PRE, [len(labels1), len(labels2)], use_attention_pooling=True)
sd = torch.load(MODEL, map_location='cpu')
miss, unexp = model.load_state_dict(sd, strict=False)
assert len(miss) == 0 and len(unexp) == 0
model = model.cuda().eval(); log("model loaded")

# ---- cell selection ----
ds = load_from_disk(DATA).filter(lambda x: x['is_dbl'] == 0, num_proc=4)
df = pd.DataFrame({'ct': ds['cell_type'], 'grp': ds['grp'],
                   'tp': ds['time_point'], 'dset': ds['dataset']})
df['row'] = np.arange(len(df))
log("strata (dataset x time_point x grp) for Microglia/Astrocyte:")
log(df[df.ct.isin(['Microglia','Astrocyte'])].groupby(['ct','dset','tp','grp']).size().to_string())

rng = np.random.RandomState(SEED)
def sample_strata(ct, grp, cap_per_stratum, total_cap):
    sub = df[(df.ct == ct) & (df.grp == grp)]
    pick = []
    for _, gdf in sub.groupby(['dset','tp']):
        k = min(cap_per_stratum, len(gdf))
        pick += rng.choice(gdf.row.values, k, replace=False).tolist()
    rng.shuffle(pick)
    return pick[:total_cap]

cells_mg_inj = sample_strata('Microglia', 'injury', 120, 800)
cells_mg_shm = sample_strata('Microglia', 'sham', 60, 120)
cells_as_inj = sample_strata('Astrocyte', 'injury', 60, 240)
sel = cells_mg_inj + cells_mg_shm + cells_as_inj
log(f"selected: MG-inj={len(cells_mg_inj)} MG-sham={len(cells_mg_shm)} Astro-inj={len(cells_as_inj)}")

sub = ds.select(sel)
arm = (['MG_injury']*len(cells_mg_inj)) + (['MG_sham']*len(cells_mg_shm)) + (['Astro_injury']*len(cells_as_inj))
tp_arr = np.array(sub['time_point']); dset_arr = np.array(sub['dataset'])

# ---- perturbation forwards ----
def forward_variants(seqs):
    """seqs: list of np arrays -> (P_injury array, task1 argmax array)"""
    out_p, out_t = [], []
    B = 64
    with torch.no_grad():
        for s in range(0, len(seqs), B):
            chunk = seqs[s:s+B]
            seq = [torch.tensor(x, dtype=torch.long) for x in chunk]
            input_ids = pad_sequence(seq, batch_first=True, padding_value=0).cuda()
            attn = (torch.arange(input_ids.shape[1])[None,:] <
                    torch.tensor([len(x) for x in chunk])[:,None]).long().cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, logits, _ = model(input_ids, attn)  # returns (0, [t1_logits, t2_logits], same) when labels=None
            pr = torch.softmax(logits[1].float(), -1)[:, INJ].cpu().numpy()
            t1 = logits[0].float().argmax(-1).cpu().numpy()
            out_p.append(pr); out_t.append(t1)
    return np.concatenate(out_p), np.concatenate(out_t)

rows = []
cand_toks = {g: t for g, t in hits.items()}
for ci in range(len(sub)):
    ids = np.array(sub['input_ids'][ci], dtype=np.int64)
    variants, vnames = [ids], ['__orig__']
    rng_ci = np.random.RandomState(SEED + ci)
    null_pool = np.array([t for t in set(ids.tolist()) if t not in cand_toks.values()])
    n_null = min(N_NULL, len(null_pool))
    null_choice = rng_ci.choice(null_pool, n_null, replace=False) if n_null else []
    for g, t in cand_toks.items():
        if t in ids:
            variants.append(ids[ids != t]); vnames.append(g)
    for t in null_choice:
        variants.append(ids[ids != t]); vnames.append(f'__null__')
    P, T1 = forward_variants(variants)
    p0, t0_ = P[0], T1[0]
    null_P = P[1 + sum(1 for g in cand_toks if cand_toks[g] in ids):]  # nulls appended after candidates
    null_mean = float(null_P.mean()) if len(null_P) else None
    for name, p, t1 in zip(vnames, P, T1):
        rows.append({'cell_i': ci, 'arm': arm[ci], 'dataset': dset_arr[ci], 'time_point': tp_arr[ci],
                     'variant': name, 'present': 1 if name != '__orig__' else np.nan,
                     'P_injury': round(float(p), 5), 'dP': round(float(p0 - p), 5),
                     'task1_flip': int(t1 != t0_)})
    if ci % 100 == 0: log(f"cell {ci}/{len(sub)} | null_mean={null_mean}")

long_df = pd.DataFrame(rows)
long_df.to_csv(f'{OUT}/isp_cells_long.csv.gz', index=False, compression='gzip')

# ---- per-gene summary ----
cand_rows = long_df[(long_df.variant != '__orig__')]
summ = []
for g in cand_toks:
    gg = cand_rows[(cand_rows.variant == g)]
    for a in gg.arm.unique():
        x = gg[(gg.arm == a)]
        n_cells_arm = (long_df.arm == a).sum() / 1  # approx via orig rows
        n_present = len(x)
        if n_present == 0: continue
        nulls = long_df[(long_df.arm == a) & (long_df.variant == '__null__')]
        nm, nsd = nulls.dP.mean(), nulls.dP.std()
        dpm = x.dP.mean()
        z = (dpm - nm) / (nsd / np.sqrt(n_present)) if nsd and n_present else np.nan
        summ.append({'gene': g, 'arm': a, 'n_present': n_present,
                     'presence_rate': round(n_present / max(1, int((long_df[(long_df.arm==a)&(long_df.variant=='__orig__')]).shape[0])), 3),
                     'mean_dP': round(float(dpm), 5), 'median_dP': round(float(x.dP.median()), 5),
                     'task1_flip_rate': round(float(x.task1_flip.mean()), 4),
                     'null_mean_dP': round(float(nm), 5), 'null_sd_dP': round(float(nsd), 5),
                     'z_vs_null': round(float(z), 2)})
summ_df = pd.DataFrame(summ)
# BH-FDR on |z| within each arm (two-sided empirical proxy)
for a in summ_df.arm.unique():
    m = summ_df.arm == a
    p = 2 * (1 - 0.5 * (1 + np.array([abs(z) for z in summ_df.loc[m,'z_vs_null']]) / max(1e-9, 1 + abs(summ_df.loc[m,'z_vs_null']).max())))
    order = np.argsort(p); ranked = p[order]; ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = ranked * len(p) / (np.arange(len(p)) + 1); q = np.clip(q, 0, 1)
    qq = np.empty_like(q); qq[order] = q
    summ_df.loc[m, 'q_approx'] = np.round(qq, 4)
summ_df = summ_df.sort_values(['arm', 'mean_dP'], ascending=[True, False])
summ_df.to_csv(f'{OUT}/isp_gene_summary.csv', index=False)
log(summ_df.head(30).to_string())

meta = {'n_cells': len(sub), 'genes': {g: int(t) for g, t in hits.items()},
        'missing_genes': misses, 'n_null_per_cell': N_NULL,
        'arms': dict(pd.Series(arm).value_counts()),
        'runtime_s': round(time.time() - t0)}
json.dump(meta, open(f'{OUT}/isp_metrics.json', 'w'), indent=1)
log("ISP DONE")
