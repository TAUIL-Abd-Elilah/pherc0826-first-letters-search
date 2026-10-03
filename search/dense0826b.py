"""Dense PHerc0826 First Letters campaign, v2 site plan.
PHerc0826 is flattened (~3:1, long axis along x): sheets run along x nearly everywhere, so every site uses depth
axis y (v1's radial +-x sites looked along the sheets and could not be tracked, and its y sites at 0.55/0.7 R fell
outside the scroll). Per height (every 1300 voxels): lateral positions x = scroll centre + {-1600, 0, +1600}
(adjacent 1600-wide tiles), and at each, depths at 0.15/0.38/0.62/0.85 of the local y thickness of the scroll mask.
Each site: one sheet tracked from m7 over z +-650 x lateral +-800 x depth +-250, rendered along the normal, read by
Reader v2 + d9v2 (ens), render deleted. Ordered so the whole scroll is covered early (middle tile, inner depths first).
-> <CAMPAIGN_DIR>/z<z>_x<i>_d<j>/   (v1 sites in the same dir are kept)
usage: dense0826b.py"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zfetch as zf  # noqa: E402
from dense0826 import VOL, PRED, D9, OUT, N, sections  # noqa: E402

LAT = (0, -1600, 1600)
LOG = os.path.join(OUT, 'campaign_v2.log')


def say(*a):
    """print and append to <OUT>/campaign_v2.log (the run may live in a terminal tab)"""
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(time.strftime('%H:%M ') + line + '\n')


DEP = (0.38, 0.62, 0.15, 0.85)


def plan():
    sp = os.path.join(OUT, 'sites_v2.json')
    if os.path.exists(sp):
        return json.load(open(sp))
    m = zf.meta(VOL, 4)
    sites = []
    for s in sections():
        zc = int(s['z'])
        sl = zf.read_box(VOL, 4, (zc // 16, 0, 0), (zc // 16 + 1, m['shape'][1], m['shape'][2]))[0] > 0
        for i, dx in enumerate(LAT):
            lat = int(s['cx'] + dx)
            c0, c1 = max(0, (lat - 400) // 16), min(sl.shape[1], (lat + 400) // 16)
            yy = np.nonzero(sl[:, c0:c1].any(1))[0]
            if yy.size < 20:
                continue
            ya, yb = np.percentile(np.nonzero(sl[:, c0:c1])[0], (1, 99)) * 16
            for j, f in enumerate(DEP):
                sites.append({'name': f'z{zc}_x{i}_d{j}', 'z': zc, 'lat': lat, 'dep': int(ya + f * (yb - ya)),
                              'thick': int(yb - ya), 'order': (j if j < 2 else j + 1) * 10 + i})
    sites.sort(key=lambda r: (r['order'], r['z']))
    json.dump(sites, open(sp, 'w'), indent=1)
    return sites


def main():
    os.makedirs(OUT, exist_ok=True)
    sites = plan()
    say(len(sites), 'sites')
    for st in sites:
        name, zc, lat, dep = st['name'], st['z'], st['lat'], st['dep']
        d = os.path.join(OUT, name)
        sp = os.path.join(d, 'scores.json')
        if os.path.exists(sp) and 'ens_fwd' in json.load(open(sp)):
            continue
        fp = os.path.join(d, 'failed.txt')
        if os.path.exists(fp):
            if not any(k in open(fp, encoding='utf-8', errors='ignore').read() for k in ('URLError', 'getaddrinfo', 'timed out', 'ConnectionReset')):
                continue
            os.remove(fp)                                   # network failure: retry the site on this pass
        t = time.time()
        box = [zc - 650, zc + 650, max(0, dep - 250), min(N, dep + 250), max(0, lat - 800), min(N, lat + 800)]
        if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
            r = subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), VOL, PRED, d, '--box', *map(str, box),
                                '--seed', str(zc), str(lat), str(dep), '--axis', 'y', '--normal', '--no-infer', '--clean-cache'],
                               capture_output=True, text=True)
            if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, 'failed.txt'), 'w').write((r.stdout + r.stderr)[-2000:])
                say(name, 'track failed', round(time.time() - t), 's')
                continue
            cov = [l for l in r.stdout.splitlines() if 'coverage' in l]
        else:
            cov = []
        v = float(np.load(os.path.join(d, 'sheet_valid.npy')).mean())
        if v < 0.05:
            open(os.path.join(d, 'failed.txt'), 'w').write(f'valid {v:.3f}\n' + '\n'.join(cov))
            say(name, 'empty sheet', round(v, 3), cov[-1:] if cov else '', round(time.time() - t), 's')
            continue
        subprocess.run([sys.executable, os.path.join(HERE, 'read_sheets.py'), d, '--no-v8in', '--ensemble', '--delete-render',
                        '--ckpt', f'd9v2={D9}'], capture_output=True, text=True)
        sc = json.load(open(sp)) if os.path.exists(sp) else {}
        e = {k: (sc[k]['best_2mm'], sc[k]['band']['score']) for k in ('ens_fwd', 'ens_rev') if k in sc}
        say(name, 'valid', sc.get('valid_frac'), e, 'one-sided', sc.get('ens_one_sided'), round(time.time() - t), 's')


if __name__ == '__main__':
    main()
