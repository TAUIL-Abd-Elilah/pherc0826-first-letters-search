"""Measure the offset between each segment's ~9 um surface volume and its stretched ~2.4 um prediction.

Same method as cross-scan-ink-transfer/src/dist9/align_pair.py (validated there): run ink_9um (hybrid_3d2d seed 42,
step 75k, forward) on sv.zarr, high-pass both maps (sigma 16), NCC over integer offsets within +-24 px (2x coarse,
then full), per-quadrant check. Only fetched pixels count. Writes align.json and, after `--apply`, canon_al.npy /
human_al.npy shifted so that label[y, x] sits under sv[:, y, x]:
  ONE offset per scroll, from the per-scroll POOLED correlation surface (sum over its segments of the NCC at each
  integer offset): training segments hold only ~100 sampled blocks, so single-segment peaks are noisy, while the
  canvas offset is systematic per scroll (as in dist9/fix_offset.py, run b). Check: on the full test segments
  PHerc0139 w030/w045 and PHerc0841 x3 give (-7,-7)..(-8,-7) individually, matching pherc0139-scan-transform-check.
usage: align9.py measure <seg_dir> [...] ; align9.py apply"""
import glob, json, os, subprocess, sys
from collections import defaultdict
import numpy as np
import tifffile, zarr
from scipy import ndimage

D9 = os.environ.get('D9_DATA', 'corpus_9um')
P = os.environ.get('FLS_ROOT', '.')  # holds _fl/ckpt/<checkpoints> and villa/_worktrees/ink9um
CKPT = f'{P}/_fl/ckpt/hybrid_3d2d-seed42/step-075000.pth'
ENV = dict(os.environ, PYTHONPATH=f'{P}/villa/_worktrees/ink9um/vesuvius/src')
R = 24


def shift(a, dy, dx, fill=np.nan):
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def ncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def highpass(a, s=16.0):
    b = np.nan_to_num(a)
    return b - ndimage.gaussian_filter(b, s)


def measure(d):
    if os.path.exists(os.path.join(d, 'align.json')):
        return
    out = os.path.join(d, 'ink9um_s42.tif')
    if not os.path.exists(out):
        subprocess.check_call([sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', os.path.join(d, 'sv.zarr'), CKPT, out,
                               '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'forward', '--no-compile'],
                              env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
    pred = tifffile.imread(out).astype(np.float32)
    pred = pred / 255.0 if pred.max() > 1.5 else pred
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    mid = np.asarray(sv[sv.shape[0] // 2])
    H, W = mid.shape
    valid = ndimage.binary_erosion(mid > 0, iterations=8)
    canon = np.load(os.path.join(d, 'canon.npy')).astype(np.float32)
    pr = np.where(valid, highpass(np.where(valid, pred[:H, :W], 0)), np.nan)
    ca = highpass(canon)
    q = 2
    prq, caq = pr[::q, ::q], ca[::q, ::q]
    S = {(dy, dx): ncc(prq, shift(caq, dy, dx)) for dy in range(-R // q, R // q + 1) for dx in range(-R // q, R // q + 1)}
    by, bx = max((k for k in S if np.isfinite(S[k])), key=lambda k: S[k])
    best = (-2.0, 0, 0)
    for dy in range(by * q - 2, by * q + 3):
        for dx in range(bx * q - 2, bx * q + 3):
            r = ncc(pr[::2, ::2], shift(ca, dy, dx)[::2, ::2])
            if np.isfinite(r) and r > best[0]:
                best = (r, dy, dx)
    vals = np.array([v for v in S.values() if np.isfinite(v)])
    rec = {'offset_yx': [best[1], best[2]], 'r_best': round(best[0], 4), 'r_zero': round(ncc(pr[::2, ::2], ca[::2, ::2]), 4),
           'r_search_p99': round(float(np.percentile(vals, 99)), 4), 'valid_px': int(valid.sum()),
           'aligner': 'ink_9um hybrid_3d2d-seed42 step-075000 forward, high-passed sigma 16'}
    json.dump(rec, open(os.path.join(d, 'align.json'), 'w'), indent=1)
    print(os.path.basename(d)[:50], rec, flush=True)


def maps(d):
    pred = tifffile.imread(os.path.join(d, 'ink9um_s42.tif')).astype(np.float32)
    pred = pred / 255.0 if pred.max() > 1.5 else pred
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    mid = np.asarray(sv[sv.shape[0] // 2]); H, W = mid.shape
    valid = ndimage.binary_erosion(mid > 0, iterations=8)
    canon = np.load(os.path.join(d, 'canon.npy')).astype(np.float32)
    return np.where(valid, highpass(np.where(valid, pred[:H, :W], 0)), np.nan), highpass(canon)


def apply():
    R2 = 24
    dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(D9, '*', 'align.json')))
    scroll = {d: os.path.basename(d).split('__')[0] for d in dirs}
    # coarse: half-resolution maps, offsets in 2 px steps over +-R2, pooled per scroll
    coarse, n = {}, defaultdict(int)
    for d in dirs:
        sp = os.path.join(d, 'ncc_coarse.npy')
        if not os.path.exists(sp):
            pr, ca = maps(d)
            prh, cah = pr[::2, ::2], ca[::2, ::2]
            g = np.array([[ncc(prh, shift(cah, dy, dx)) for dx in range(-R2 // 2, R2 // 2 + 1)] for dy in range(-R2 // 2, R2 // 2 + 1)])
            np.save(sp, g)
        coarse[scroll[d]] = coarse.get(scroll[d], 0) + np.nan_to_num(np.load(sp)); n[scroll[d]] += 1
    # fine: full resolution, +-2 px around each scroll's coarse peak, pooled again
    off = {}
    for s, g in coarse.items():
        i, j = np.unravel_index(np.argmax(g), g.shape)
        cy, cx = 2 * (i - R2 // 2), 2 * (j - R2 // 2)
        fine = np.zeros((5, 5))
        for d in [d for d in dirs if scroll[d] == s]:
            pr, ca = maps(d)
            for a, dy in enumerate(range(cy - 2, cy + 3)):
                for b, dx in enumerate(range(cx - 2, cx + 3)):
                    fine[a, b] += np.nan_to_num(ncc(pr[::2, ::2], shift(ca, dy, dx)[::2, ::2]))
        a, b = np.unravel_index(np.argmax(fine), fine.shape)
        off[s] = [int(cy - 2 + a), int(cx - 2 + b)]
        print(s, 'segments', n[s], 'pooled offset', off[s], 'mean NCC at peak', round(float(fine.max() / n[s]), 4),
              'coarse at 0', round(float(g[R2 // 2, R2 // 2] / n[s]), 4), flush=True)
    json.dump({'pooled_offset_yx': off, 'segments': dict(n)}, open(os.path.join(D9, 'scroll_offsets.json'), 'w'), indent=1)
    for d in dirs:
        if not os.path.exists(os.path.join(d, 'canon.npy')):
            continue
        s = os.path.basename(d).split('__')[0]
        r = json.load(open(os.path.join(d, 'align.json')))
        dy, dx = off[s]
        r['applied_offset_yx'] = [dy, dx]; r['applied_from'] = 'scroll_pooled'
        json.dump(r, open(os.path.join(d, 'align.json'), 'w'), indent=1)
        sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
        mid = np.asarray(sv[sv.shape[0] // 2]); valid = mid > 0
        canon = np.load(os.path.join(d, 'canon.npy')).astype(np.float32)
        np.save(os.path.join(d, 'canon_al.npy'), np.where(valid, shift(canon, dy, dx), np.nan).astype(np.float16))
        os.remove(os.path.join(d, 'canon.npy'))           # disk: the shifted copy replaces it (rebuild from the TIF if needed)
        hp = os.path.join(d, 'human.npy')
        if os.path.exists(hp):
            meta = json.load(open(os.path.join(d, 'meta.json')))
            h = np.load(hp).astype(np.float32)
            # labels drawn on the ~9 um canvas are already on the texture; labels from the ~2.4 um canvas get the offset
            hs = h if meta.get('human_on_9um_canvas') else shift(h, dy, dx, 0.0)
            np.save(os.path.join(d, 'human_al.npy'), hs.astype(np.float16))


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        for d in sys.argv[2:]:
            try:
                measure(d)
            except Exception as e:
                print('FAILED', d, repr(e)[:200], flush=True)
    else:
        apply()
