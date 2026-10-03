"""Neighbour-winding control for a dense-campaign site.
Ink sits on ONE sheet face (plus, at most, an offset copy on the face pressed against it: the adjacent winding).
A crack, void, compression zone or scan artefact crosses several windings at the same (z, lateral) place.
So: build the sheets k = -3..+3 crossings away from the site's sheet (bigsheet_v2 --neighbour-of <site>
--run-offset k: per column, count m7 runs from the site sheet), read them with the same models, and measure how
much of the site's strong response reappears.
All sheets share the (z, lateral) grid of the site (depth axis y), so the maps are pixel-aligned.
Verdict per site: response present at |k| >= 2 = structure; only at k = 0 (and maybe |k| = 1) = keep as a lead.
usage: neighbors.py <site_name> <fwd|rev> [k ...]   (default k = -3 -2 -1 1 2 3)
-> <CAMPAIGN_DIR>/_nb/<site>/o<k>/ + _nb/<site>/summary.json + montage.png"""
import json, os, subprocess, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dense0826 import VOL, PRED, D9, OUT, N  # noqa: E402


def site_args(name):
    st = {s['name']: s for s in json.load(open(os.path.join(OUT, 'sites_v2.json')))}[name]
    zc, lat, dep = st['z'], st['lat'], st['dep']
    box = [zc - 650, zc + 650, max(0, dep - 250), min(N, dep + 250), max(0, lat - 800), min(N, lat + 800)]
    return box, [str(zc), str(lat), str(dep)]


def run(name, k, out):
    sp = os.path.join(out, 'scores.json')
    if os.path.exists(sp) and 'ens_fwd' in json.load(open(sp)):
        return True
    box, seed = site_args(name)
    if not os.path.exists(os.path.join(out, 'sheet_valid.npy')):
        r = subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), VOL, PRED, out, '--box', *map(str, box),
                            '--seed', *seed, '--axis', 'y', '--normal', '--no-infer', '--clean-cache', '--run-offset', str(k),
                            '--neighbour-of', os.path.join(OUT, name)],
                           capture_output=True, text=True)
        if not os.path.exists(os.path.join(out, 'sheet_valid.npy')):
            print(name, k, 'track failed:', (r.stdout + r.stderr).strip().splitlines()[-1:], flush=True)
            return False
    subprocess.run([sys.executable, os.path.join(HERE, 'read_sheets.py'), out, '--no-v8in', '--ensemble', '--delete-render',
                    '--ckpt', f'd9v2={D9}'], capture_output=True, text=True)
    return os.path.exists(sp)


def load(d, face):
    m = cv2.imread(os.path.join(d, f'ens_{face}.png'), 0).astype(np.float32) / 255
    v = np.load(os.path.join(d, 'sheet_valid.npy'))
    return m, v


def main():
    name, face = sys.argv[1], sys.argv[2]
    ks = [int(x) for x in sys.argv[3:]] or [-3, -2, -1, 1, 2, 3]
    site = os.path.join(OUT, name)
    base = os.path.join(OUT, '_nb', name)
    m0, v0 = load(site, face)
    blobs = (m0 > 0.5) & v0                                   # the site's strong response
    ring = (cv2.dilate(blobs.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & ~blobs & v0   # its surroundings
    h0 = np.load(os.path.join(site, 'sheet_00_h.npy'))
    rows, tiles = [], []
    for k in ks:
        d = os.path.join(base, f'o{k:+d}')
        os.makedirs(d, exist_ok=True)
        if not run(name, k, d):
            rows.append({'k': k, 'status': 'failed'}); continue
        hk = np.load(os.path.join(d, 'sheet_00_h.npy'))
        for f in ('fwd', 'rev'):
            mk, vk = load(d, f)
            both = v0 & vk
            sel_b, sel_r = blobs & vk, ring & vk
            rec = {'k': k, 'face': f, 'valid_overlap': round(float(both.mean()), 3),
                   'depth_gap_vox': round(float(np.median((hk - h0)[both])), 1) if both.any() else None,
                   'gap_p10_p90': np.percentile((hk - h0)[both], [10, 90]).round(1).tolist() if both.any() else None,
                   'corr_with_site': round(float(np.corrcoef(m0[both], mk[both])[0, 1]), 3) if both.sum() > 1000 else None,
                   # mean response under the site's blobs vs around them: > ~0.1 lift = the blobs reappear here
                   'in_blobs': round(float(mk[sel_b].mean()), 3) if sel_b.sum() > 100 else None,
                   'around': round(float(mk[sel_r].mean()), 3) if sel_r.sum() > 100 else None}
            rows.append(rec)
            if f == face:
                tiles.append((k, mk * vk))
        print(name, k, [r for r in rows if r.get('k') == k], flush=True)
    s0 = {'k': 0, 'face': face, 'in_blobs': round(float(m0[blobs].mean()), 3), 'around': round(float(m0[ring].mean()), 3)}
    json.dump({'site': name, 'face': face, 'site_ref': s0, 'neighbours': rows}, open(os.path.join(base, 'summary.json'), 'w'), indent=1)
    # montage: site map with neighbour maps of the same face, ordered by k, blob outline drawn on each
    tiles = sorted(tiles + [(0, m0 * v0)], key=lambda t: t[0])
    cnt, _ = cv2.findContours(blobs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ims = []
    for k, m in tiles:
        im = cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(im, cnt, -1, (0, 0, 255), 3)
        cv2.putText(im, f'k={k:+d} {face}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
        ims.append(cv2.resize(im, (im.shape[1] // 3, im.shape[0] // 3), interpolation=cv2.INTER_AREA))
    cv2.imwrite(os.path.join(base, 'montage.png'), np.concatenate([np.pad(x, ((0, 4), (0, 4), (0, 0))) for x in ims], 1))
    print('site', s0, flush=True)


if __name__ == '__main__':
    main()
