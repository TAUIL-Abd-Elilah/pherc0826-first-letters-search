"""Score ~9 um ink models on the held-out segments of the six-scroll corpus (copy of dist9/eval9.py; D9 layout).

For each segment (prep_pair.py + align_pair.py done) and each checkpoint: run the stock ink_9um
inference CLI (forward), then
  human AUC  - positives: human_al >= 0.5; negatives: human_al < 0.05 within 2 mm (214 px) of labelled
               ink (annotators labelled around their text, not the whole segment); sv valid only
  canon r    - Pearson r with the aligned canonical 2.4 um prediction, valid pixels, 2x subsampled
  canon AUC  - canon >= 0.5 vs canon < 0.2
usage: eval9.py --segments A B --ckpt name=path [name=path ...] --out results.json
"""
import argparse
import json
import os

import numpy as np
import zarr
from scipy import ndimage

import subprocess, sys, tifffile
OUT = os.environ.get('D9_DATA', 'corpus_9um')
P = os.environ.get('FLS_ROOT', '.')  # holds _fl/ckpt/<checkpoints> and villa/_worktrees/ink9um
ENV = dict(os.environ, PYTHONPATH=f'{P}/villa/_worktrees/ink9um/vesuvius/src')


def infer(d, ckpt, name, extra=()):
    out = os.path.join(d, name + '.tif')
    if not os.path.exists(out):
        subprocess.check_call([sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', os.path.join(d, 'sv.zarr'), ckpt, out,
                               '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'forward',
                               '--no-compile', *extra], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    a = tifffile.imread(out).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def auc(score, pos, neg):
    s_pos, s_neg = score[pos], score[neg]
    if s_pos.size < 50 or s_neg.size < 50:
        return None
    rng = np.random.default_rng(0)
    if s_pos.size > 400000:
        s_pos = rng.choice(s_pos, 400000, replace=False)
    if s_neg.size > 400000:
        s_neg = rng.choice(s_neg, 400000, replace=False)
    allv = np.concatenate([s_pos, s_neg])
    rank = allv.argsort(kind='mergesort').argsort() + 1
    return float((rank[:s_pos.size].sum() - s_pos.size * (s_pos.size + 1) / 2) / (s_pos.size * s_neg.size))


def score(seg, pred):
    d = os.path.join(OUT, seg)
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    canon = np.load(os.path.join(d, 'canon_al.npy')).astype(np.float32)
    H, W = min(pred.shape[0], canon.shape[0]), min(pred.shape[1], canon.shape[1])
    pred, canon = pred[:H, :W], canon[:H, :W]
    valid = np.asarray(sv[sv.shape[0] // 2, :H, :W]) > 0
    out = {}
    m = valid & np.isfinite(canon)
    x, y = pred[m][::4], canon[m][::4]
    out['canon_r'] = float(np.corrcoef(x, y)[0, 1])
    out['canon_auc'] = auc(pred, m & (canon >= 0.5), m & (canon < 0.2))
    hp = os.path.join(d, 'human_al.npy')
    if os.path.exists(hp):
        h = np.load(hp).astype(np.float32)[:H, :W]
        near = ndimage.distance_transform_edt(h < 0.5) <= 214
        out['human_auc'] = auc(pred, valid & (h >= 0.5), valid & near & (h < 0.05))
        out['human_pos_px'] = int((valid & (h >= 0.5)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', nargs='+', required=True)
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.ckpt:
        name, path = spec.split('=', 1)
        for seg in a.segments:
            key = f'{name}|{seg}'
            if key in res:
                continue
            pred = infer(os.path.join(OUT, seg), path, 'pred_' + name)
            res[key] = score(seg, pred)
            print(key, json.dumps(res[key]), flush=True)
            json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
