"""Dense First Letters campaign on any scroll with a 9 um volume and an m7 surface prediction.
Generalises dense0826b.py (PHerc0826 was flattened along x, so every site there used depth axis y). Here each
site's depth axis is chosen from the LOCAL sheet orientation: on the L3 axial CT slice the structure tensor gives
the sheet normal, and the site looks along whichever of y / x is closer to it (bigsheet_v2 boxes are
axis-aligned; tracking tolerates ~45 deg of tilt, more with its reseed).
Sites: heights every --step voxels; per height, centres on a --grid voxel square grid inside the scroll mask, kept
when >= 60 % of the 500 x 1600 box is inside the mask. Ordered so that the whole scroll is covered early (centre
heights and the middle of each cross-section first). Each site: bigsheet_v2 (normal render) + read_sheets
(Reader v2 + d9v2 ensemble, both faces; the d9v2 slot takes D9_CKPT, logged in campaign.json), render
deleted. Network failures are retried on the next pass.
usage: dense_any.py --scroll PHerc0358 --vol <volume path> --pred <m7 path> --out <dir> [--step 1300] [--grid 1200]
-> <out>/sites.json, <out>/<site>/{scores.json, ens_*.png, ct.png, sheet_*.npy}, <out>/campaign.log"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zfetch as zf  # noqa: E402

P = os.environ.get('FLS_ROOT', '.')
D9 = os.environ.get('D9_CKPT', f'{P}/_fl/dist9v2/runs/d9v2_a/ft-012000.pth')
NET = ('URLError', 'getaddrinfo', 'timed out', 'ConnectionReset', 'No space left', 'Errno 28',
       '.part')                                                   # retryable (.part = cache removed mid-write)


def plan(a):
    sp = os.path.join(a.out, 'sites.json')
    if os.path.exists(sp):
        return json.load(open(sp))
    m = zf.meta(a.vol, 3)
    Z, H, W = m['shape'][0] * 8, m['shape'][1] * 8, m['shape'][2] * 8
    sites = []
    for zc in range(a.step, Z - 650, a.step):
        sl = zf.read_box(a.vol, 3, (zc // 8, 0, 0), (zc // 8 + 1, m['shape'][1], m['shape'][2]))[0].astype(np.float32)
        mask = ndimage.binary_opening(sl > 0, iterations=2)
        if mask.sum() < 500:
            continue
        # structure tensor of the CT texture (sheets = lines at 75 um/px): the dominant gradient = sheet normal
        g = ndimage.gaussian_filter(sl, 1.0)
        gy, gx = np.gradient(g)
        Jxx, Jyy, Jxy = (ndimage.gaussian_filter(v, 8.0) for v in (gx * gx, gy * gy, gx * gy))
        theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)              # normal angle from +x
        coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-6)
        cy0, cx0 = ndimage.center_of_mass(mask)
        for gy_ in range(a.grid // 2, H, a.grid):
            for gx_ in range(a.grid // 2, W, a.grid):
                py, px = gy_ // 8, gx_ // 8
                if not (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]) or not mask[py, px]:
                    continue
                t = theta[py, px]
                axis = 'x' if abs(np.cos(t)) >= abs(np.sin(t)) else 'y'
                hy, hx = (250, 800) if axis == 'y' else (800, 250)
                y0, y1, x0, x1 = max(0, gy_ - hy) // 8, min(H, gy_ + hy) // 8, max(0, gx_ - hx) // 8, min(W, gx_ + hx) // 8
                inside = mask[y0:y1, x0:x1].mean() if y1 > y0 and x1 > x0 else 0
                if inside < 0.6:
                    continue
                r = np.hypot(py - cy0, px - cx0) / np.sqrt(mask.sum() / np.pi)
                sites.append({'name': f'z{zc}_y{gy_}_x{gx_}', 'z': zc, 'cy': int(gy_), 'cx': int(gx_), 'axis': axis,
                              'normal_deg': round(float(np.degrees(t)), 1), 'coherence': round(float(coh[py, px]), 3),
                              'inside': round(float(inside), 2), 'r_rel': round(float(r), 2)})
    zs = sorted({s['z'] for s in sites}); zmid = np.median(zs) if zs else 0
    sites.sort(key=lambda s: (round(abs(s['z'] - zmid) / (2 * a.step)), s['r_rel']))   # middle heights + centre first
    os.makedirs(a.out, exist_ok=True)
    json.dump(sites, open(sp, 'w'), indent=1)
    return sites


def say(a, *x):
    line = ' '.join(str(v) for v in x)
    print(line, flush=True)
    with open(os.path.join(a.out, 'campaign.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%m-%d %H:%M ') + line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scroll', required=True); ap.add_argument('--vol', required=True); ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--step', type=int, default=1300); ap.add_argument('--grid', type=int, default=1200)
    ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump({'scroll': a.scroll, 'vol': a.vol, 'pred': a.pred, 'd9v2_slot': D9}, open(os.path.join(a.out, 'campaign.json'), 'w'), indent=1)   # for the controls
    sites = plan(a)
    say(a, a.scroll, len(sites), 'sites', {'y': sum(s['axis'] == 'y' for s in sites), 'x': sum(s['axis'] == 'x' for s in sites)})
    if a.plan_only:
        return
    N = zf.meta(a.vol, 0)['shape']
    for s in sites:
        d = os.path.join(a.out, s['name'])
        sp = os.path.join(d, 'scores.json')
        if os.path.exists(sp) and 'ens_fwd' in json.load(open(sp)):
            continue
        fp = os.path.join(d, 'failed.txt')
        if os.path.exists(fp):
            if not any(k in open(fp, encoding='utf-8', errors='ignore').read() for k in NET):
                continue
            os.remove(fp)
        import shutil
        warned = False
        while shutil.disk_usage(a.out).free < 3e9:                # never burn sites on a full disk: wait instead
            if not warned:
                say(a, 'LOW DISK: %.1f GB free, pausing before %s' % (shutil.disk_usage(a.out).free / 1e9, s['name'])); warned = True
            time.sleep(300)
        t = time.time()
        zc, cy, cx, axis = s['z'], s['cy'], s['cx'], s['axis']
        if axis == 'y':
            box = [zc - 650, zc + 650, max(0, cy - 250), min(N[1], cy + 250), max(0, cx - 800), min(N[2], cx + 800)]
            seed = [zc, cx, cy]
        else:
            box = [zc - 650, zc + 650, max(0, cy - 800), min(N[1], cy + 800), max(0, cx - 250), min(N[2], cx + 250)]
            seed = [zc, cy, cx]
        if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
            r = subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), a.vol, a.pred, d, '--box', *map(str, box),
                                '--seed', *map(str, seed), '--axis', axis, '--normal', '--no-infer', '--clean-cache'],
                               capture_output=True, text=True)
            if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
                os.makedirs(d, exist_ok=True)
                open(fp, 'w').write((r.stdout + r.stderr)[-2000:])
                say(a, s['name'], 'track failed', round(time.time() - t), 's'); continue
        v = float(np.load(os.path.join(d, 'sheet_valid.npy')).mean())
        if v < 0.05:
            open(fp, 'w').write(f'valid {v:.3f}\n')
            say(a, s['name'], 'empty sheet', round(v, 3)); continue
        subprocess.run([sys.executable, os.path.join(HERE, 'read_sheets.py'), d, '--no-v8in', '--ensemble', '--delete-render',
                        '--ckpt', f'd9v2={D9}'], capture_output=True, text=True)
        sc = json.load(open(sp)) if os.path.exists(sp) else {}
        e = {k: (sc[k]['best_2mm'], sc[k]['band']['score']) for k in ('ens_fwd', 'ens_rev') if k in sc}
        say(a, s['name'], axis, 'valid', sc.get('valid_frac'), e, 'one-sided', sc.get('ens_one_sided'), round(time.time() - t), 's')


if __name__ == '__main__':
    main()
