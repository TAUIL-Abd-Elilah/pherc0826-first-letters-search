"""Rank and montage dense-campaign sites. Score per site and face: ensemble best 2 mm window, row (band)
score, and one-sidedness (that face's best 2 mm minus the other face's). Ink is one-sided; folds, voids and
crumpled papyrus tend to light both faces. Montage rows: CT | ensemble fwd | ensemble rev, for the top N.
usage: dense_review.py <campaign_dir> [N]  -> <campaign_dir>/_review/{ranking.json, top_<i>.png}"""
import glob, json, os, sys
import numpy as np
import cv2
from scipy import ndimage


def slab_fraction(png, valid_png):
    """Fraction of strong (>0.5) response in the map that belongs to its single largest connected blob.
    Text = many separate marks (low); a contact zone, edge or crust = one contiguous slab (high)."""
    m = cv2.imread(png, 0).astype(np.float32) / 255
    v = cv2.imread(valid_png, 0) > 0 if valid_png and os.path.exists(valid_png) else np.ones_like(m, bool)
    b = (m > 0.5) & v
    if b.sum() < 100:
        return 0.0
    lab, n = ndimage.label(b)
    return round(float(np.bincount(lab.ravel())[1:].max() / b.sum()), 3)

def elongation(png, valid_png):
    """Stroke-likeness of the strong (>0.5) response: per connected blob area / r^2 (r = largest inscribed radius),
    area-weighted median (liliandevarieux, villa #1907; a disc = 3.14, strokes much higher). Blob fields score low."""
    m = cv2.imread(png, 0).astype(np.float32) / 255
    v = cv2.imread(valid_png, 0) > 0 if valid_png and os.path.exists(valid_png) else np.ones_like(m, bool)
    b = (m > 0.5) & v
    if b.sum() < 100:
        return 0.0
    lab, n = ndimage.label(b)
    dt = ndimage.distance_transform_edt(b)
    area = np.bincount(lab.ravel())[1:]
    r = ndimage.maximum(dt, lab, np.arange(1, n + 1))
    keep = area >= 20
    if not keep.any():
        return 0.0
    e, w = (area / np.maximum(np.asarray(r), 1) ** 2)[keep], area[keep]
    o = np.argsort(e); c = np.cumsum(w[o])
    return round(float(e[o][np.searchsorted(c, c[-1] / 2)]), 2)


D = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def main():
    rows = []
    for sp in glob.glob(os.path.join(D, '*', 'scores.json')):
        sc = json.load(open(sp)); name = os.path.basename(os.path.dirname(sp))
        if 'ens_fwd' not in sc:
            continue
        for face, other in (('fwd', 'rev'), ('rev', 'fwd')):
            a, b = sc[f'ens_{face}'], sc[f'ens_{other}']
            slab = slab_fraction(os.path.join(os.path.dirname(sp), f'ens_{face}.png'), os.path.join(os.path.dirname(sp), 'ct.png'))
            el = elongation(os.path.join(os.path.dirname(sp), f'ens_{face}.png'), os.path.join(os.path.dirname(sp), 'ct.png'))
            rows.append({'site': name, 'face': face, 'elong': el, 'best_2mm': a['best_2mm'], 'band': a['band']['score'], 'slab': slab,
                         'one_sided': round(a['best_2mm'] - b['best_2mm'], 3), 'valid': sc.get('valid_frac'),
                         # rank: strong, row-like, and one-sided
                         # down-weight sites whose response is mostly one contiguous slab
                         'rank_score': round((a['best_2mm'] + 2 * a['band']['score'] + max(0.0, a['best_2mm'] - b['best_2mm'])) * (1 - 0.7 * slab), 3)})
    rows.sort(key=lambda r: -r['rank_score'])
    os.makedirs(os.path.join(D, '_review'), exist_ok=True)
    json.dump(rows, open(os.path.join(D, '_review', 'ranking.json'), 'w'), indent=1)
    print(len({r['site'] for r in rows}), 'sites scored')
    for i, r in enumerate(rows[:N]):
        d = os.path.join(D, r['site'])
        ims = [cv2.imread(os.path.join(d, f), 0) for f in ('ct.png', 'ens_fwd.png', 'ens_rev.png')]
        ims = [cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA) for x in ims]
        for x, t in zip(ims, ('CT', 'ensemble fwd', 'ensemble rev')):
            cv2.putText(x, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
        m = np.concatenate([np.pad(x, ((0, 4), (0, 4)), constant_values=128) for x in ims], 1)
        cv2.putText(m, f"#{i + 1} {r['site']} {r['face']} best2mm {r['best_2mm']} band {r['band']} one-sided {r['one_sided']} slab {r['slab']} elong {r['elong']}",
                    (5, m.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.imwrite(os.path.join(D, '_review', f'top_{i + 1:02d}.png'), m)
        print(i + 1, r)


if __name__ == '__main__':
    main()
