"""Re-read campaign sites with v8-in on BOTH faces. v8-in (Youssef Nader's 9 um recipe) found the PHerc1447
strokes, and is a different architecture from Reader v2 / d9v2, so it is an independent check of a candidate.
Per site: re-render the kept sheet (bigsheet_v2 --reuse-h, same box/seed as the campaign), run v8-in on the centre
24 layers clipped at 200 in both layer orders (outward->inward = 'fwd', reversed = 'rev'), save v8in_fwd/rev.png,
add v8in_fwd / v8in_rev scores to scores.json, delete the render.
usage: reread_v8in.py <site_name> [...]   (sites from <CAMPAIGN_DIR>/sites_v2.json)"""
import json, os, shutil, subprocess, sys
import numpy as np
import cv2, zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dense0826 import VOL, PRED, OUT  # noqa: E402
from neighbors import site_args  # noqa: E402
from read_sheets import best2mm, P  # noqa: E402
from band_score import band_score  # noqa: E402


def main():
    sys.path.insert(0, f'{P}/_fl/ckpt/v8in')
    from ink8um import InkDetector, predict_stack
    v8 = InkDetector.from_pretrained(f'{P}/_fl/ckpt/v8in').cuda().eval()
    for name in sys.argv[1:]:
        d = os.path.join(OUT, name)
        sp = os.path.join(d, 'scores.json')
        sc = json.load(open(sp))
        if 'v8in_rev' in sc:
            continue
        zp = os.path.join(d, 'sheet_00.zarr')
        if not os.path.exists(zp):
            box, seed = site_args(name)
            subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), VOL, PRED, d, '--box', *map(str, box), '--seed', *seed,
                            '--axis', 'y', '--normal', '--no-infer', '--clean-cache', '--reuse-h', d], capture_output=True, text=True)
        s = np.asarray(zarr.open_array(zp, mode='r')[:])
        c = s.shape[0] // 2
        valid = s[c] > 0
        st = np.ascontiguousarray(np.clip(s[c - 12:c + 12], 0, 200).astype(np.uint8).transpose(1, 2, 0))
        for face, rev in (('fwd', False), ('rev', True)):
            p = predict_stack(v8, st, reverse=rev, stride=32, batch_size=8, num_workers=0)
            cv2.imwrite(os.path.join(d, f'v8in_{face}.png'), (p * 255).astype(np.uint8))
            sc[f'v8in_{face}'] = {**best2mm(p, valid), 'band': band_score(p, valid)}
        json.dump(sc, open(sp, 'w'), indent=1)
        shutil.rmtree(zp, ignore_errors=True)
        print(name, {k: (sc[k]['best_2mm'], sc[k]['band']['score']) for k in ('v8in_fwd', 'v8in_rev', 'ens_fwd', 'ens_rev')}, flush=True)


if __name__ == '__main__':
    main()
