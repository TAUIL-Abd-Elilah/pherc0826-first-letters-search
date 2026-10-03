"""v8-in over the survey blocks, in the depth order the model expects (outward -> inward; see v8in_sheets.py).

Our sheet stacks are sampled along the block's +depth axis (y or x). v8-in reads 24 layers ordered from the outward
side to the inward (recto) side, measured on the released PHerc1447 renders. So per block: find the scroll centre
at the block's height (level-4 mask centroid), and use the stack as is when +depth points inward, reversed otherwise.
Writes <block>/v8in/<sheet>_in.png and <block>/v8in/scores_in.json.
usage: v8in_survey.py [--stride N] <block dirs...>"""
import argparse, glob, json, os, sys
import numpy as np, cv2, zarr
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zfetch as zf  # noqa: E402

P = os.path.join(os.environ.get('FLS_ROOT', '.'), '_fl', 'ckpt')
_centres = {}


def centre(vol, z):
    key = (vol, z // 512)
    if key not in _centres:
        s = zf.meta(vol, 4)['shape']
        zz = min(s[0] - 1, z // 16)
        sl = zf.read_box(vol, 4, (zz, 0, 0), (zz + 1, s[1], s[2]))[0] > 0
        cy, cx = ndimage.center_of_mass(sl)
        _centres[key] = (cy * 16, cx * 16)
    return _centres[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--ckpt', default=f'{P}/v8in')
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, a.ckpt)
    from ink8um import InkDetector, predict_stack
    model = InkDetector.from_pretrained(a.ckpt).cuda().eval()
    for d in a.dirs:
        sj = json.load(open(os.path.join(d, 'sheets.json')))
        lo, hi, ax = sj['lo_zyx'], sj['hi_zyx'], sj['axis']
        zc = (lo[0] + hi[0]) // 2
        cy, cx = centre(sj['volume'], zc)
        if ax == 'y':
            inward_plus = cy > (lo[1] + hi[1]) / 2
        else:
            inward_plus = cx > (lo[2] + hi[2]) / 2
        od = os.path.join(d, 'v8in'); os.makedirs(od, exist_ok=True)
        sp = os.path.join(od, 'scores_in.json')
        scores = json.load(open(sp)) if os.path.exists(sp) else {}
        scores['_meta'] = {'inward_is_plus_depth': bool(inward_plus), 'stride': a.stride, 'ckpt': os.path.basename(a.ckpt)}
        for zp in sorted(glob.glob(os.path.join(d, 'sheet_*.zarr'))):
            name = os.path.basename(zp).replace('.zarr', '')
            if name in scores:
                continue
            s = np.asarray(zarr.open_array(zp, mode='r')[:])
            L = s.shape[0]; k0 = (L - 24) // 2
            stack = np.clip(s[k0:k0 + 24], 0, 200).astype(np.uint8).transpose(1, 2, 0)
            valid = ndimage.binary_erosion(s[L // 2 - 1:L // 2 + 1].max(0) > 0, iterations=48)
            prob = predict_stack(model, np.ascontiguousarray(stack), reverse=not inward_plus, stride=a.stride,
                                 batch_size=a.batch, num_workers=0)
            cv2.imwrite(os.path.join(od, f'{name}_in.png'), (prob * 255).astype(np.uint8))
            b = ((prob > 0.5) & valid).astype(np.float32)
            w = int(round(2.0 / 8.64e-3))
            loc = ndimage.uniform_filter(b, w) if min(b.shape) > w else b
            scores[name] = {'frac': round(float(b[valid].mean()) if valid.any() else 0.0, 4),
                            'best_2mm': round(float(loc.max()), 3)}
            json.dump(scores, open(sp, 'w'), indent=1)
        print(os.path.basename(d), sj['volume'].split('/')[0], 'inward=+depth' if inward_plus else 'inward=-depth',
              'best', max(((v['best_2mm'], k) for k, v in scores.items() if k != '_meta'), default=(0, '')), flush=True)


if __name__ == '__main__':
    main()
