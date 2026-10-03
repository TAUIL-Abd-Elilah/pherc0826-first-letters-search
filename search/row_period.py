"""Text-line periodicity of an ink map.
Text lines run around the scroll (lateral) and stack along z, so a page of text gives a periodic z-profile once the
map is averaged laterally over a window: period = line pitch (a few mm). Blob speckle and fibre texture should not.
score(map): split laterally into windows win_mm wide; per window, z-profile = mean over valid pixels (rows with
< 30 % valid -> gap-filled); detrend (gaussian 6 mm), Hann, power spectrum; line_frac = power in periods
[p_lo, p_hi] mm / power in [1.2, 15] mm; also the peak period. Averaged over windows (power-weighted).
Calibrated on PHerc0841 (unseen-scroll text, d9v2 corpus test segments: human labels + Reader v2 maps) and
PHerc0139 w030/w045, then applied to the PHerc0826 campaign maps.
usage: row_period.py"""
import glob, json, os, sys
import cv2
import numpy as np
import tifffile, zarr
from scipy import ndimage

D9 = os.environ.get('D9_DATA', 'corpus_9um')
OUT = os.environ.get('CAMPAIGN_DIR', 'campaign_0826')


def profiles(m, valid, px_mm, win_mm=10.0):
    w = max(8, int(round(win_mm / px_mm)))
    out = []
    for c0 in range(0, m.shape[1] - w + 1, w):
        mm, vv = m[:, c0:c0 + w], valid[:, c0:c0 + w]
        frac = vv.mean(1)
        if (frac > 0.3).mean() < 0.6:
            continue
        p = np.where(frac > 0.3, (mm * vv).sum(1) / np.maximum(vv.sum(1), 1), np.nan)
        rows = np.flatnonzero(np.isfinite(p))
        r0, r1 = rows.min(), rows.max() + 1
        p = p[r0:r1]
        if (r1 - r0) * px_mm < 10:                          # need >= 10 mm of z to see ~2 lines (site boxes are 12 mm)
            continue
        good = np.isfinite(p)
        p = np.interp(np.arange(p.size), np.flatnonzero(good), p[good])
        out.append(p)
    return out


def spectrum_score(m, valid, px_mm, p_lo=3.5, p_hi=8.0):
    num = den = 0.0
    peaks = []
    for p in profiles(m, valid, px_mm):
        p = p - ndimage.gaussian_filter1d(p, 6.0 / px_mm)
        p = p * np.hanning(p.size)
        f = np.fft.rfftfreq(p.size, d=px_mm)                # cycles / mm
        P = np.abs(np.fft.rfft(p)) ** 2
        per = np.where(f > 0, 1 / np.maximum(f, 1e-9), np.inf)
        band = (per >= 1.2) & (per <= 15)
        line = (per >= p_lo) & (per <= p_hi)
        if P[band].sum() <= 0:
            continue
        num += P[line].sum(); den += P[band].sum()
        peaks.append(float(per[band][np.argmax(P[band])]))
    return (round(num / den, 3) if den else None), (round(float(np.median(peaks)), 2) if peaks else None), len(peaks)


def seg_maps(seg):
    d = os.path.join(D9, seg)
    sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
    valid = np.asarray(sv[sv.shape[0] // 2]) > 0
    valid = ndimage.binary_erosion(valid, iterations=8)
    H, W = valid.shape
    res = {}
    for name in ('pred_reader_v2', 'pred_d9v2_12k'):
        f = os.path.join(d, name + '.tif')
        if os.path.exists(f):
            a = tifffile.imread(f).astype(np.float32)
            res[name] = (a / 255 if a.max() > 1.5 else a)[:H, :W]
    hp = os.path.join(d, 'human_al.npy')
    if os.path.exists(hp):
        h = np.load(hp).astype(np.float32)[:H, :W]
        res['human'] = np.nan_to_num(h)
    return res, valid


def main():
    px = 9.4e-3
    rows = []
    print('--- known text (d9v2 test segments; px ~9.4 um)')
    for seg in sorted(glob.glob(os.path.join(D9, 'PHerc0841__*')) + glob.glob(os.path.join(D9, 'PHerc0139__*w030*'))
                      + glob.glob(os.path.join(D9, 'PHerc0139__*w045*'))):
        seg = os.path.basename(seg)
        maps, valid = seg_maps(seg)
        for k, m in maps.items():
            for axis in ('z', 'lateral'):
                mm, vv = (m, valid) if axis == 'z' else (m.T, valid.T)
                s, pk, n = spectrum_score(mm, vv, px)
                rows.append({'set': 'known', 'item': f'{seg[:40]}|{k}|{axis}', 'line_frac': s, 'peak_mm': pk, 'windows': n})
                print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in rows[-1].items()}, flush=True)
    print('--- PHerc0826 campaign sites (best face, z axis = scroll axis)')
    for sp in sorted(glob.glob(os.path.join(OUT, 'z*_x*_d*', 'scores.json'))):
        d = os.path.dirname(sp); sc = json.load(open(sp))
        if 'ens_fwd' not in sc:
            continue
        face = 'fwd' if sc['ens_fwd']['best_2mm'] >= sc['ens_rev']['best_2mm'] else 'rev'
        m = cv2.imread(os.path.join(d, f'ens_{face}.png'), 0).astype(np.float32) / 255
        valid = np.load(os.path.join(d, 'sheet_valid.npy'))
        valid = ndimage.binary_erosion(valid, iterations=8)
        s, pk, n = spectrum_score(m, valid, 9.362e-3)
        s2, pk2, n2 = spectrum_score(m.T, valid.T, 9.362e-3)
        rows.append({'set': '0826', 'item': f'{os.path.basename(d)}|ens_{face}', 'line_frac': s, 'peak_mm': pk, 'windows': n,
                     'lateral_frac': s2})
    for name in ('z14300_x2_d2', 'z3900_x0_d1'):
        mp = os.path.join(OUT, '_ext', name, 'mosaic_ens_fwd.png')
        if os.path.exists(mp):
            m = cv2.imread(mp, 0).astype(np.float32) / 255
            s, pk, n = spectrum_score(m, m > 0, 3 * 9.362e-3)        # mosaics are 1/3 scale
            s2, _, _ = spectrum_score(m.T, (m > 0).T, 3 * 9.362e-3)
            rows.append({'set': '0826_mosaic', 'item': name, 'line_frac': s, 'peak_mm': pk, 'windows': n, 'lateral_frac': s2})
            print(rows[-1], flush=True)
    v = [r['line_frac'] for r in rows if r['set'] == '0826' and r['line_frac'] is not None]
    vl = [r['lateral_frac'] for r in rows if r['set'] == '0826' and r.get('lateral_frac') is not None]
    print('0826 lateral (null) median', round(float(np.median(vl)), 3) if vl else None)
    print('0826 sites: n', len(v), 'line_frac median', round(float(np.median(v)), 3), 'p90', round(float(np.percentile(v, 90)), 3),
          'max', round(float(max(v)), 3))
    top = sorted([r for r in rows if r['set'] == '0826' and r['line_frac'] is not None], key=lambda r: -r['line_frac'])[:8]
    for r in top:
        print('  top', r)
    json.dump(rows, open(os.path.join(OUT, '_review', 'row_period.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
