"""Read normal-rendered sheets (bigsheet_v2 --normal: 28 layers, index increasing inward) with Reader v2 (both
orders, villa CLI) and v8-in (centre 24 layers, outward -> inward, clip 200), plus any extra villa checkpoints
given as name=path. Writes <dir>/{ct.png, <model>.png, scores.json}.
usage: read_sheets.py <sheet_dir> [...] [--no-v8in] [--ensemble] [--delete-render] [--ckpt name=path ...]
--ensemble averages Reader v2 and d9v2 per face (needs --ckpt d9v2=...), --delete-render removes sheet_00.zarr after."""
import json, os, subprocess, sys
import numpy as np
import cv2, tifffile, zarr
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blockscan_infer import MODELS, ENV, PY  # noqa: E402
from band_score import band_score  # noqa: E402

P = os.environ.get('FLS_ROOT', '.')  # holds _fl/ckpt/<checkpoints> and villa/_worktrees/ink9um


def best2mm(prob, valid, px_mm=9.362e-3):
    v = ndimage.binary_erosion(valid, iterations=48)
    b = ((prob > 0.5) & v).astype(np.float32)
    w = int(round(2.0 / px_mm))
    loc = ndimage.uniform_filter(b, w) if min(b.shape) > w else b
    return {'frac': round(float(b[v].mean()) if v.any() else 0.0, 4), 'best_2mm': round(float(loc.max()), 3)}


def villa(zp, name, ck, d):
    out = os.path.join(d, f'_{name}.tif')
    for attempt in range(2):                       # a shared GPU can make one run fail (OOM): retry once
        subprocess.call([PY, '-m', 'vesuvius.ink_detection.inference.infer', zp, ck, out, '--overlap', '0.5', '--blend-mode', 'hann',
                         '--batch-size', '8', '--direction', 'both', '--no-compile'], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(out.replace('.tif', '_reverse.tif')):
            break
    else:
        return None
    maps = []
    for f in (out, out.replace('.tif', '_reverse.tif')):
        q = tifffile.imread(f).astype(np.float32)
        q = q / 255 if q.max() > 1.5 else q
        maps.append(np.clip((q - 0.25) / 0.5, 0, 1))
        os.remove(f)
    return maps


def main():
    args = sys.argv[1:]
    flags = {f for f in ('--no-v8in', '--ensemble', '--delete-render') if f in args}
    args = [x for x in args if x not in flags]
    extra = {}
    if '--ckpt' in args:
        i = args.index('--ckpt')
        for spec in args[i + 1:]:
            k, v = spec.split('=', 1); extra[k] = v
        args = args[:i]
    sys.path.insert(0, f'{P}/_fl/ckpt/v8in')
    from ink8um import InkDetector, predict_stack
    v8 = None
    for d in args:
        zp = os.path.join(d, 'sheet_00.zarr')
        if not os.path.exists(zp):
            continue
        sp = os.path.join(d, 'scores.json')
        sc = json.load(open(sp)) if os.path.exists(sp) else {}
        s = np.asarray(zarr.open_array(zp, mode='r')[:])
        L = s.shape[0]; c = L // 2
        valid = s[c] > 0
        if valid.mean() < 0.05:
            continue
        ct = s[c - 2:c + 2].astype(np.float32).mean(0); lo, hi = np.percentile(ct[valid], (1, 99))
        cv2.imwrite(os.path.join(d, 'ct.png'), (np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1) * 255 * valid).astype(np.uint8))
        todo = {'rv2': MODELS['rv2'], **extra}
        for name, ck in todo.items():
            if f'{name}_fwd' in sc:
                continue
            maps = villa(zp, name, ck, d)
            if maps is None:
                sc[f'{name}_error'] = 'inference failed twice'
                continue
            for k2, m in zip(('fwd', 'rev'), maps):
                cv2.imwrite(os.path.join(d, f'{name}_{k2}.png'), (m * 255).astype(np.uint8))
                sc[f'{name}_{k2}'] = {**best2mm(m, valid), 'band': band_score(m, valid)}
        if '--ensemble' in flags and 'rv2_fwd' in sc and 'd9v2_fwd' in sc and 'ens_fwd' not in sc:
            for k2 in ('fwd', 'rev'):
                e = (cv2.imread(os.path.join(d, f'rv2_{k2}.png'), 0).astype(np.float32) + cv2.imread(os.path.join(d, f'd9v2_{k2}.png'), 0)) / 510.0
                cv2.imwrite(os.path.join(d, f'ens_{k2}.png'), (e * 255).astype(np.uint8))
                sc[f'ens_{k2}'] = {**best2mm(e, valid), 'band': band_score(e, valid)}
            # one-sidedness: ink is on one face; structure tends to read on both
            sc['ens_one_sided'] = round(sc['ens_fwd']['best_2mm'] - sc['ens_rev']['best_2mm'], 3)
        if 'v8in' not in sc and '--no-v8in' not in flags:
            if v8 is None:
                v8 = InkDetector.from_pretrained(f'{P}/_fl/ckpt/v8in').cuda().eval()
            st = np.ascontiguousarray(np.clip(s[c - 12:c + 12], 0, 200).astype(np.uint8).transpose(1, 2, 0))
            p = predict_stack(v8, st, reverse=False, stride=32, batch_size=8, num_workers=0)
            cv2.imwrite(os.path.join(d, 'v8in.png'), (p * 255).astype(np.uint8))
            sc['v8in'] = {**best2mm(p, valid), 'band': band_score(p, valid)}
        sc['valid_frac'] = round(float(valid.mean()), 3)
        json.dump(sc, open(sp, 'w'), indent=1)
        if '--delete-render' in flags and not any(k.endswith('_error') for k in sc):
            import shutil
            shutil.rmtree(zp, ignore_errors=True)           # keep maps, ct.png, sheet_00_h.npy (re-render if needed)
        print(os.path.basename(d.rstrip('/')), {k: (v['best_2mm'], v['band']['score']) for k, v in sc.items() if isinstance(v, dict)}, flush=True)


if __name__ == '__main__':
    main()
