"""Re-render tracked sheets along their local normal (not along the block axis), 24 layers ordered outward -> inward,
the input v8-in expects. Then run v8-in on it.

A sheet is a height map H(z, w) along the block's depth axis (y or x). Its normal in (depth, z, w) coordinates is
(1, -dH/dz, -dH/dw), normalised (H smoothed first). Layers sample the CT at s + k n, k = -11.5 .. +11.5 voxels,
stacked so that the index increases towards the scroll centre (inward).
usage: normal_stack.py <dir> --volume V --lo z y x --hi z y x --axis y|x [--sheets 00 03 ...] [--stride N]
       (a blockscan dir can pass --from-sheets-json instead of volume/lo/hi/axis)
Writes <dir>/v8in_n/<sheet>.png (probability x 255) and <dir>/v8in_n/scores.json."""
import argparse, glob, json, os, sys
import numpy as np, cv2
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zfetch as zf  # noqa: E402
from v8in_survey import centre  # noqa: E402

P = os.path.join(os.environ.get('FLS_ROOT', '.'), '_fl', 'ckpt')


def render(CT, H, valid, inward_plus, layers=24, smooth=3.0, clip=200):
    """CT: (D, Z, W) depth-first box; H: (Z, W) surface depth. Returns (Z, W, layers) uint8, outward -> inward."""
    Hs = ndimage.gaussian_filter(H.astype(np.float64), smooth)
    gz, gw = np.gradient(Hs)
    n = np.stack([np.ones_like(Hs), -gz, -gw], 0)
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    if not inward_plus:
        n = -n                                     # n now points inward
    Z, W = H.shape
    zz, ww = np.mgrid[0:Z, 0:W].astype(np.float64)
    out = np.zeros((Z, W, layers), np.uint8)
    for i, k in enumerate(np.arange(layers) - (layers - 1) / 2.0):
        c = [H + k * n[0], zz + k * n[1], ww + k * n[2]]
        out[..., i] = np.clip(ndimage.map_coordinates(CT, c, order=1, mode="constant"), 0, clip).astype(np.uint8)
    out *= valid[..., None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--volume'); ap.add_argument('--lo', type=int, nargs=3); ap.add_argument('--hi', type=int, nargs=3)
    ap.add_argument('--axis', choices=['y', 'x'])
    ap.add_argument('--from-sheets-json', action='store_true')
    ap.add_argument('--sheets', nargs='*')
    ap.add_argument('--ckpt', default=f'{P}/v8in')
    ap.add_argument('--stride', type=int, default=21)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tag', default='v8in_n', help='output subdirectory')
    a = ap.parse_args()
    if a.from_sheets_json:
        sj = json.load(open(os.path.join(a.dir, 'sheets.json')))
        a.volume, a.lo, a.hi, a.axis = sj['volume'], sj['lo_zyx'], sj['hi_zyx'], sj['axis']
    lo, hi = a.lo, a.hi
    zc = (lo[0] + hi[0]) // 2
    cy, cx = centre(a.volume, zc)
    inward_plus = (cy > (lo[1] + hi[1]) / 2) if a.axis == 'y' else (cx > (lo[2] + hi[2]) / 2)
    ct = zf.read_box(a.volume, 0, tuple(lo), tuple(hi))
    CT = ct.transpose(1, 0, 2) if a.axis == 'y' else ct.transpose(2, 0, 1)
    del ct
    sys.path.insert(0, a.ckpt)
    from ink8um import InkDetector, predict_stack
    model = InkDetector.from_pretrained(a.ckpt).cuda().eval()
    od = os.path.join(a.dir, a.tag); os.makedirs(od, exist_ok=True)
    sp = os.path.join(od, 'scores.json')
    scores = json.load(open(sp)) if os.path.exists(sp) else {}
    scores['_meta'] = {'inward_is_plus_depth': bool(inward_plus), 'stride': a.stride, 'render': 'normal, 24 layers', 'ckpt': os.path.basename(a.ckpt.rstrip('/'))}
    hs = sorted(glob.glob(os.path.join(a.dir, 'sheet_*_h.npy')))
    for hp in hs:
        name = os.path.basename(hp).replace('_h.npy', '')
        if (a.sheets and name.split('_')[1] not in a.sheets) or name in scores:
            continue
        H = np.load(hp)
        vp = os.path.join(a.dir, 'sheet_valid.npy')
        if os.path.exists(vp) and len(hs) == 1:
            valid = np.load(vp)
        else:
            import zarr
            s = zarr.open_array(hp.replace('_h.npy', '.zarr'), mode='r')
            valid = np.asarray(s[s.shape[0] // 2]) > 0
        stack = render(CT, H, valid, inward_plus)
        prob = predict_stack(model, stack, reverse=False, stride=a.stride, batch_size=a.batch, num_workers=0)
        cv2.imwrite(os.path.join(od, f'{name}.png'), (prob * 255).astype(np.uint8))
        v = ndimage.binary_erosion(valid, iterations=48)
        b = ((prob > 0.5) & v).astype(np.float32)
        w = int(round(2.0 / 8.64e-3))
        loc = ndimage.uniform_filter(b, w) if min(b.shape) > w else b
        scores[name] = {'frac': round(float(b[v].mean()) if v.any() else 0.0, 4), 'best_2mm': round(float(loc.max()), 3)}
        json.dump(scores, open(sp, 'w'), indent=1)
        print(os.path.basename(a.dir), name, scores[name], flush=True)


if __name__ == '__main__':
    main()
