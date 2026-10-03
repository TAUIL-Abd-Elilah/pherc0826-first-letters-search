"""Dense PHerc0826 First Letters campaign. PHerc0826 = the clearest eligible 9.36 um scroll.
192 sites = 12 heights (every 1300 voxels, ~12 mm) x 4 sides (+y, -y, +x, -x of the scroll centre) x 4 depths into
the roll (0.25 / 0.4 / 0.55 / 0.7 of the radius). Each site: one sheet tracked from the m7 prediction over
z +-650 x lateral +-800 voxels (12 x 15 mm), rendered along the normal (28 layers, index increasing inward), read by
Reader v2 + d9v2 (both faces) and their average ("ens"), then the render is deleted (maps, ct.png and the surface
height map are kept). Ordered depth-major (0.4, 0.55, 0.25, 0.7) so the whole scroll is covered early.
-> <CAMPAIGN_DIR>/<site>/
usage: dense0826.py"""
import json, os, subprocess, sys, time
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zfetch as zf  # noqa: E402

P = os.environ.get('FLS_ROOT', '.')  # holds _fl/ckpt/<checkpoints> and villa/_worktrees/ink9um
VOL = 'PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr'
PRED = 'PHerc0826/representations/predictions/surfaces/20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr'
D9 = f'{P}/_fl/dist9v2/runs/d9v2_a/ft-012000.pth'
OUT = os.environ.get('CAMPAIGN_DIR', 'campaign_0826')
N = 511 * 16                                   # y and x extent (L0)


def sections():
    sp = os.path.join(OUT, 'sections.json')
    if os.path.exists(sp):
        return json.load(open(sp))
    m = zf.meta(VOL, 4); Z = m['shape'][0] * 16
    secs = []
    for zc in range(1300, Z - 650, 1300):
        sl = zf.read_box(VOL, 4, (zc // 16, 0, 0), (zc // 16 + 1, m['shape'][1], m['shape'][2]))[0] > 0
        if sl.sum() < 100:
            continue
        cy, cx = ndimage.center_of_mass(sl); yy, xx = np.nonzero(sl)
        R = float(np.percentile(np.hypot(yy - cy, xx - cx), 98))
        secs.append({'z': zc, 'cy': cy * 16, 'cx': cx * 16, 'R': R * 16})
    json.dump(secs, open(sp, 'w'), indent=1)
    return secs


def main():
    os.makedirs(OUT, exist_ok=True)
    secs = sections()
    sites = []
    for f in (0.4, 0.55, 0.25, 0.7):
        for s in secs:
            for axis, sign in (('y', 1), ('y', -1), ('x', 1), ('x', -1)):
                sites.append((f, s, axis, sign))
    print(len(sites), 'sites', flush=True)
    for f, s, axis, sign in sites:
        zc, cy, cx, R = int(s['z']), s['cy'], s['cx'], s['R']
        name = f'z{zc}_{"p" if sign > 0 else "m"}{axis}_r{int(f * 100)}'
        d = os.path.join(OUT, name)
        sp = os.path.join(d, 'scores.json')
        if os.path.exists(sp) and 'ens_fwd' in json.load(open(sp)):
            continue
        if os.path.exists(os.path.join(d, 'failed.txt')):
            continue
        t = time.time()
        if axis == 'y':
            dep = int(cy + sign * f * R); lat = int(cx)
            box = [zc - 650, zc + 650, max(0, dep - 250), min(N, dep + 250), max(0, lat - 800), min(N, lat + 800)]
        else:
            dep = int(cx + sign * f * R); lat = int(cy)
            box = [zc - 650, zc + 650, max(0, lat - 800), min(N, lat + 800), max(0, dep - 250), min(N, dep + 250)]
        if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
            r = subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), VOL, PRED, d, '--box', *map(str, box),
                                '--seed', str(zc), str(lat), str(dep), '--axis', axis, '--normal', '--no-infer', '--clean-cache'],
                               capture_output=True, text=True)
            if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, 'failed.txt'), 'w').write((r.stdout + r.stderr)[-2000:])
                print(name, 'track failed', round(time.time() - t), 's', flush=True)
                continue
        subprocess.run([sys.executable, os.path.join(HERE, 'read_sheets.py'), d, '--no-v8in', '--ensemble', '--delete-render',
                        '--ckpt', f'd9v2={D9}'], capture_output=True, text=True)
        sc = json.load(open(sp)) if os.path.exists(sp) else {}
        e = {k: (sc[k]['best_2mm'], sc[k]['band']['score']) for k in ('ens_fwd', 'ens_rev') if k in sc}
        print(name, 'valid', sc.get('valid_frac'), e, 'one-sided', sc.get('ens_one_sided'), round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    main()
