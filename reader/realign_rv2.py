"""Residual label-offset check with an independent strong reader.
Why: align9's pooled offsets came from ink_9um s42 maps; on the held-out segments two independent readers (Reader v2
and Scheirer's ft) both put the labels of PHerc0500P2 and PHerc0814 ~8 px off ((-8,-8) and (-8,-6)), while 0841,
0009B and 0139 sit within 0-4 px. 65 of 132 training segments come from those two scrolls.
measure: for up to N segments per scroll (test segments first, then training ones, fixed order), run Reader v2
         forward on sv.zarr (pred_reader_v2.tif, kept), and NCC its high-passed map against canon_al over integer
         shifts +-R (1 px, 2x subsampled), only on fetched pixels -> residual.json per segment.
remeasure: edge-free residuals (NaN-aware high-pass, +-24 px) from the kept Reader-v2 maps -> residual2.json.
apply:   PER-SCROLL TARGET (waits for <D9>/apply_go) from residual2.json; see apply() for the rule. Shifts canon_al
         (+ human_al from the 2.4 um canvas); log residual_applied.json.
usage: realign_rv2.py measure [N] ; realign_rv2.py apply"""
import glob, json, os, subprocess, sys
from collections import defaultdict
import numpy as np
import tifffile, zarr
from scipy import ndimage
from align9 import shift, ncc, highpass, D9, P, ENV

RV2 = f'{P}/_fl/ckpt/reader_v2/reader-v2-step040000.pth'
R = 12
HERE = os.path.dirname(os.path.abspath(__file__))


def segs():
    test = [l.strip().replace('/', '__') for l in open(os.path.join(HERE, 'test.txt')) if l.strip()]
    train = [l.strip().replace('/', '__') for l in open(os.path.join(HERE, 'train.txt')) if l.strip()]
    return test, train


def nan_highpass(a, m, sigma=16.0):
    """High-pass inside mask m only (normalised convolution), NaN outside: no edge response at block borders."""
    a = np.where(m, a, 0).astype(np.float32)
    den = ndimage.gaussian_filter(m.astype(np.float32), sigma)
    low = ndimage.gaussian_filter(a, sigma) / np.maximum(den, 1e-3)
    return np.where(m, a - low, np.nan)


def remeasure_one(d):
    """Edge-free residual from the kept Reader-v2 map (training segments are sparse 256-px blocks: the plain
    high-pass put block-border edges into both maps -> a flat NCC plateau over +-8 px with peaks at its edge)."""
    out = os.path.join(d, 'pred_reader_v2.tif')
    if not os.path.exists(out):
        return None
    p = tifffile.imread(out).astype(np.float32)
    p = p / 255.0 if p.max() > 1.5 else p
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    mid = np.asarray(sv[sv.shape[0] // 2]); H, W = mid.shape
    c = np.load(os.path.join(d, 'canon_al.npy')).astype(np.float32)[:H, :W]
    mp = ndimage.binary_erosion(mid > 0, iterations=4)
    mc = ndimage.binary_erosion(np.isfinite(c) & (mid > 0), iterations=4)
    pr = nan_highpass(p[:H, :W], mp)
    ca = nan_highpass(np.nan_to_num(c), mc)
    # coarse: +-RW px in 2 px steps on half-res maps, then fine +-2 px at full shifts (offsets can exceed 12 px:
    # 0500P2 / 0814 climb to the edge of a +-12 search)
    RW = 24
    prh, cah = pr[::2, ::2], ca[::2, ::2]
    g = np.array([[ncc(prh, shift(cah, dy, dx)) for dx in range(-RW // 2, RW // 2 + 1)] for dy in range(-RW // 2, RW // 2 + 1)])
    i, j = np.unravel_index(np.nanargmax(g), g.shape)
    cy, cx = 2 * (i - RW // 2), 2 * (j - RW // 2)
    best = (-2.0, cy, cx)
    for dy in range(cy - 2, cy + 3):
        for dx in range(cx - 2, cx + 3):
            r = ncc(pr[::2, ::2], shift(ca, dy, dx)[::2, ::2])
            if np.isfinite(r) and r > best[0]:
                best = (r, dy, dx)
    gi, gj = i, j
    ring = [g[y, x] for y in range(g.shape[0]) for x in range(g.shape[1]) if 3 <= max(abs(y - gi), abs(x - gj)) <= 4]
    rec = {'peak_yx': [int(best[1]), int(best[2])], 'r_peak': round(float(best[0]), 4),
           'r_zero': round(float(ncc(pr[::2, ::2], ca[::2, ::2])), 4),
           'sharpness': round(float(g[gi, gj] - np.nanmean(ring)), 4) if ring else None,
           'edge_peak': bool(max(abs(gi - RW // 2), abs(gj - RW // 2)) >= RW // 2 - 1), 'aligner': 'reader-v2, NaN-aware high-pass, +-24 px (v3)'}
    np.save(os.path.join(d, 'ncc_rv2b.npy'), g.astype(np.float32))
    json.dump(rec, open(os.path.join(d, 'residual2.json'), 'w'), indent=1)
    return rec


def measure_one(d):
    rp = os.path.join(d, 'residual.json')
    if os.path.exists(rp):
        return json.load(open(rp))
    out = os.path.join(d, 'pred_reader_v2.tif')
    if not os.path.exists(out):
        subprocess.check_call([sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', os.path.join(d, 'sv.zarr'), RV2, out,
                               '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'forward', '--no-compile'],
                              env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2400)
    p = tifffile.imread(out).astype(np.float32)
    p = p / 255.0 if p.max() > 1.5 else p
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    mid = np.asarray(sv[sv.shape[0] // 2]); H, W = mid.shape
    valid = ndimage.binary_erosion(mid > 0, iterations=8)
    pr = np.where(valid, highpass(np.where(valid, p[:H, :W], 0)), np.nan)
    ca = highpass(np.nan_to_num(np.load(os.path.join(d, 'canon_al.npy')).astype(np.float32)))
    g = np.array([[ncc(pr[::2, ::2], shift(ca, dy, dx)[::2, ::2]) for dx in range(-R, R + 1)] for dy in range(-R, R + 1)])
    i, j = np.unravel_index(np.nanargmax(g), g.shape)
    rec = {'peak_yx': [int(i - R), int(j - R)], 'r_peak': round(float(g[i, j]), 4), 'r_zero': round(float(g[R, R]), 4),
           'valid_px': int(valid.sum()), 'aligner': 'reader-v2 step040000 forward'}
    np.save(os.path.join(d, 'ncc_rv2.npy'), g.astype(np.float32))
    json.dump(rec, open(rp, 'w'), indent=1)
    return rec


def measure(n):
    test, train = segs()
    per = defaultdict(list)
    for s in test + train:
        per[s.split('__')[0]].append(s)
    for scroll, ss in sorted(per.items()):
        for s in ss[:n]:
            try:
                rec = measure_one(os.path.join(D9, s))
                print(scroll, s.split('__')[1][:40], rec['peak_yx'], rec['r_zero'], '->', rec['r_peak'], flush=True)
            except Exception as e:
                print('FAILED', s, repr(e)[:200], flush=True)


def apply():
    # hold until the residuals were reviewed (per-segment offsets differ between segment generations, e.g. 0009B
    # May-2025 segments (7, 8) vs the Sept-2025 test segment (0, 3)): create <D9>/apply_go to proceed
    import time
    go = os.path.join(D9, 'apply_go')
    while not os.path.exists(go):
        print('apply waiting for', go, flush=True)
        time.sleep(120)
    # PER-SCROLL TARGET from the edge-free residuals (residual2.json): total(seg) = applied pooled offset + residual.
    # Reliable segments (sharpness >= 0.015, peak not at the search edge) agree on ~(-7,-7) from the raw 2.4 um canvas
    # for every scroll (0814/0500P2/0841/0139), matching the geometric 0139 transform check (-8,-7); align9's pooled
    # +4/+5 for 0500P2/0814 came from the block-edge plateau artefact. Target(scroll) = median total of its reliable
    # segments if >= 3 agree within 2 px (MAD), else the global median; correction = target - applied, applied to
    # every segment of the scroll when >= 4 px (smaller = within reader bias, rv2 vs Scheirer ~2 px).
    log = json.load(open(os.path.join(D9, 'residual_applied.json'))) if os.path.exists(os.path.join(D9, 'residual_applied.json')) else {}
    test, train = segs()
    tot = defaultdict(list)
    applied = {}
    for sname in test + train:
        d = os.path.join(D9, sname)
        if not os.path.exists(os.path.join(d, 'align.json')):
            continue
        sc = sname.split('__')[0]
        applied[sc] = json.load(open(os.path.join(d, 'align.json')))['applied_offset_yx']
        rp = os.path.join(d, 'residual2.json')
        if os.path.exists(rp):
            r = json.load(open(rp))
            if (r.get('sharpness') or 0) >= 0.015 and not r.get('edge_peak'):
                a = applied[sc]
                tot[sc].append([a[0] + r['peak_yx'][0], a[1] + r['peak_yx'][1]])
    allt = np.array([t for v in tot.values() for t in v])
    gmed = np.median(allt, 0).round().astype(int).tolist() if len(allt) else None
    print('global median total offset', gmed, 'from', len(allt), 'reliable segments', flush=True)
    for sc in sorted(applied):
        t = np.array(tot.get(sc, []))
        if len(t) >= 3 and np.median(np.abs(t - np.median(t, 0)).max(1)) <= 2:
            target, src = np.median(t, 0).round().astype(int).tolist(), f'scroll median of {len(t)}'
        else:
            target, src = gmed, f'global median (scroll reliable n={len(t)})'
        corr = [target[0] - applied[sc][0], target[1] - applied[sc][1]]
        do = max(abs(corr[0]), abs(corr[1])) >= 4 and sc not in log
        print(sc, 'applied', applied[sc], 'target', target, '(' + src + ')', 'correction', corr, '-> APPLY' if do else '-> keep', flush=True)
        if not do:
            log.setdefault(sc, {'target': target, 'applied': applied[sc], 'correction': corr, 'src': src, 'shifted': 0})
            continue
        n = 0
        for d in sorted(glob.glob(os.path.join(D9, sc + '__*'))):
            cp = os.path.join(d, 'canon_al.npy')
            if not os.path.exists(cp):
                continue
            c = np.load(cp).astype(np.float32)
            np.save(cp, shift(c, corr[0], corr[1]).astype(np.float16))           # NaN fill outside = unlabelled
            hp = os.path.join(d, 'human_al.npy')
            meta = json.load(open(os.path.join(d, 'meta.json')))
            if os.path.exists(hp) and not meta.get('human_on_9um_canvas'):
                np.save(hp, shift(np.load(hp).astype(np.float32), corr[0], corr[1], 0.0).astype(np.float16))
            a = json.load(open(os.path.join(d, 'align.json')))
            a['residual_rv2_correction_yx'] = corr; a['final_offset_yx'] = target
            json.dump(a, open(os.path.join(d, 'align.json'), 'w'), indent=1)
            n += 1
        log[sc] = {'target': target, 'applied': applied[sc], 'correction': corr, 'src': src, 'shifted': n}
    json.dump(log, open(os.path.join(D9, 'residual_applied.json'), 'w'), indent=1)
    print('scrolls corrected:', [k for k, v in log.items() if v['shifted']], flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'remeasure':
        from concurrent.futures import ProcessPoolExecutor
        test, train = segs()
        todo = [s for s in test + train if os.path.exists(os.path.join(D9, s, 'pred_reader_v2.tif'))
                and not os.path.exists(os.path.join(D9, s, 'residual2.json'))]
        print(len(todo), 'segments to re-measure', flush=True)
        with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as ex:
            for sname, r in zip(todo, ex.map(remeasure_one, [os.path.join(D9, s) for s in todo])):
                if r:
                    print(sname[:50], r['peak_yx'], r['r_zero'], '->', r['r_peak'], 'sharp', r['sharpness'], 'EDGE' if r['edge_peak'] else '', flush=True)
    elif sys.argv[1] == 'measure':
        measure(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        apply()
