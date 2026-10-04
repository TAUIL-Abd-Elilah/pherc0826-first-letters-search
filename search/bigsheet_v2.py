"""Track ONE sheet over a large region from a seed point, render it, and (optionally) run the ink models.
v2: --normal renders along the smoothed sheet normal (normal_stack.render: layer index increases INWARD, the
order v8-in and the villa models read as "forward"); --no-infer skips inference; --clean-cache deletes this
volume's level-0 chunk cache after rendering (disk).

Region: z [z0, z1), depth axis y [y0, y1), x [x0, x1) (L0). The m7 mask and CT are read (from the cache
when already downloaded), the sheet is grown by sheet_grow.grow() from the seed column (zs, xs) at the run
centre nearest to depth ds, then snapped and rendered with 28 layers exactly like blockscan.py.
Writes <out>/sheet.zarr, sheet_h.npy, sheet_valid.npy and runs blockscan_infer on it.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import zarr
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zfetch as zf  # noqa: E402
from fetch_pred import read_box as read_pred  # noqa: E402
from sheet_grow import fill_smooth, grow, run_centres  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('volume'); ap.add_argument('pred'); ap.add_argument('out')
    ap.add_argument('--box', type=int, nargs=6, required=True, help='z0 z1 y0 y1 x0 x1 (L0), depth axis = y')
    ap.add_argument('--seed', type=float, nargs=3, required=True, help='z x depth(y, L0 absolute)')
    ap.add_argument('--layers', type=int, default=28)
    ap.add_argument('--tol', type=float, default=4.0)
    ap.add_argument('--reuse-h', default=None, help='sheet_00_h.npy + sheet_valid.npy dir to reuse instead of tracking')
    ap.add_argument('--shift', type=float, default=0.0, help='render the stack centred at H + shift (depth voxels)')
    ap.add_argument('--normal', action='store_true'); ap.add_argument('--no-infer', action='store_true')
    ap.add_argument('--clean-cache', action='store_true')
    ap.add_argument('--axis', choices=['y', 'x'], default='y', help='depth axis; seed = (z, lateral, depth)')
    ap.add_argument('--run-offset', type=int, default=0, help='track the k-th sheet crossing beyond the seed one (+ = deeper index)')
    ap.add_argument('--neighbour-of', default=None, help='site dir: build the --run-offset k-th sheet by counting m7 runs '
                    'from that site sheet, column by column (parallel neighbour over the same area)')
    a = ap.parse_args()
    z0, z1, y0, y1, x0, x1 = a.box
    os.makedirs(a.out, exist_ok=True)
    ct = zf.read_box(a.volume, 0, (z0, y0, x0), (z1, y1, x1))
    m7 = read_pred(f'{a.pred}/0', (z0, y0, x0), (z1, y1, x1)) > 0
    print('region', ct.shape, 'MB downloaded', round(zf.downloaded() / 1e6), flush=True)
    if a.axis == 'y':
        M = m7.transpose(1, 0, 2); CT = ct.transpose(1, 0, 2)        # (depth=y, z, lateral=x)
        d0, l0 = y0, x0
    else:
        M = m7.transpose(2, 0, 1); CT = ct.transpose(2, 0, 1)        # (depth=x, z, lateral=y)
        d0, l0 = x0, y0
    del ct, m7
    D, Z, X = M.shape
    if a.reuse_h:
        H = np.load(os.path.join(a.reuse_h, 'sheet_00_h.npy'))
        V = np.load(os.path.join(a.reuse_h, 'sheet_valid.npy'))
        vals = zz = None
    elif a.neighbour_of:
        # neighbour-winding control: in every coarse column where the site sheet is valid, find the m7 run it sits on
        # and take the run k crossings beyond it (run centres ascend in depth); drop columns whose gap to the site
        # sheet departs from the local median gap (a missing or merged run there), then fill/snap as usual
        s = 4
        C = run_centres(M[:, ::s, ::s], kmax=64)
        h, w = C.shape[1:]
        H0 = np.load(os.path.join(a.neighbour_of, 'sheet_00_h.npy'))[::s, ::s][:h, :w]
        V0 = np.load(os.path.join(a.neighbour_of, 'sheet_valid.npy'))[::s, ::s][:h, :w]
        dist = np.where(np.isfinite(C), np.abs(C - H0[None]), np.inf)
        i0 = dist.argmin(0)
        n = np.isfinite(C).sum(0)
        j = i0 + a.run_offset
        ok = V0 & (dist.min(0) <= 6) & (j >= 0) & (j < n)
        Hc = np.where(ok, np.take_along_axis(C, np.clip(j, 0, C.shape[0] - 1)[None], 0)[0], np.nan).astype(np.float32)
        gap = Hc - H0
        g = float(np.nanmedian(gap)) if np.isfinite(gap).any() else 0.0
        loc = ndimage.median_filter(np.where(np.isfinite(gap), gap, g), size=9)
        Hc[np.abs(gap - loc) > 8] = np.nan
        Hc[~np.isfinite(gap)] = np.nan
        cov = float(np.isfinite(Hc).mean())
        print('neighbour', a.run_offset, 'median gap', round(g, 1), 'coarse coverage', round(cov, 3), flush=True)
        if cov < 0.02:
            raise SystemExit('neighbour sheet not found')
        F, validc = fill_smooth(Hc)
        Hf = ndimage.zoom(F, (Z / h, X / w), order=1)[:Z, :X]
        Hf = np.pad(Hf, ((0, Z - Hf.shape[0]), (0, X - Hf.shape[1])), mode='edge')
        V = ndimage.zoom(validc.astype(np.float32), (Z / h, X / w), order=0)[:Z, :X] > 0.5
        V = np.pad(V, ((0, Z - V.shape[0]), (0, X - V.shape[1])), mode='edge')
        ks = np.arange(-3, 4)
        zz = np.clip(np.rint(Hf)[None] + ks[:, None, None], 0, D - 1).astype(int)
        vals = np.take_along_axis(M, zz, 0).astype(np.float32)
        wsum = vals.sum(0)
        off = np.where(wsum > 0, (vals * ks[:, None, None]).sum(0) / np.maximum(wsum, 1), 0)
        H = ndimage.gaussian_filter(np.rint(Hf) + off, 1.5).astype(np.float32)
    else:
        s = 4
        C = run_centres(M[:, ::s, ::s], kmax=64)
        zs, xs, ds = a.seed
        cy, cx = int(round((zs - z0) / s)), int(round((xs - l0) / s))
        col = C[:, cy, cx]
        col = col[np.isfinite(col)]
        if col.size == 0:                                    # empty seed column: nearest column with a sheet run
            have = np.isfinite(C).any(0)
            yy, xx = np.nonzero(have)
            if yy.size == 0:
                raise SystemExit('no m7 sheet anywhere in the box')
            k = int(np.argmin((yy - cy) ** 2 + (xx - cx) ** 2))
            print('seed column empty; moved seed by', int(yy[k] - cy), int(xx[k] - cx), 'coarse cells', flush=True)
            cy, cx = int(yy[k]), int(xx[k])
            col = C[:, cy, cx]; col = col[np.isfinite(col)]
        sd = float(col[np.argmin(np.abs(col - (ds - d0)))])
        print('seed depth', ds - d0, '-> run centre', sd, flush=True)
        if a.run_offset:
            # neighbour-winding control: the k-th sheet crossing beyond the seed's along depth (run centres ascend)
            j = int(np.argmin(np.abs(col - sd))) + a.run_offset
            if not 0 <= j < col.size:
                raise SystemExit(f'run offset {a.run_offset} leaves the seed column ({col.size} runs)')
            sd = float(col[j]); ds = sd + d0
            print('run offset', a.run_offset, '-> run centre', sd, flush=True)
        Hc = grow(C, (cy, cx), sd, tol=a.tol)
        cov = float(np.isfinite(Hc).mean())
        print('coarse coverage', round(cov, 3), flush=True)
        if cov < 0.3:
            # the seed run is often a fragment (oblique or broken sheet): try the other runs near the seed depth in
            # nearby columns, nearest first, then a looser tolerance; keep the widest sheet
            h, w = C.shape[1:]
            cands = []
            for oy in (0, -12, 12):
                for ox in (0, -20, 20, -40, 40):
                    yy, xx = cy + oy, cx + ox
                    if 0 <= yy < h and 0 <= xx < w:
                        cc = C[:, yy, xx]; cc = cc[np.isfinite(cc)]
                        cands += [(abs(v - (ds - d0)), yy, xx, float(v)) for v in cc if abs(v - (ds - d0)) <= 120]
            cands.sort()
            for tol in (a.tol, 2 * a.tol):
                for _, yy, xx, v in cands[:24]:
                    if np.isfinite(Hc[yy, xx]) and abs(Hc[yy, xx] - v) < 2:
                        continue
                    Ht = grow(C, (yy, xx), v, tol=tol)
                    ct_ = float(np.isfinite(Ht).mean())
                    if ct_ > cov:
                        Hc, cov = Ht, ct_
                        print('reseed', yy - cy, xx - cx, 'depth', v, 'tol', tol, '-> coverage', round(cov, 3), flush=True)
                    if cov >= 0.5:
                        break
                if cov >= 0.3:
                    break
        F, validc = fill_smooth(Hc)
        h, w = Hc.shape
        Hf = ndimage.zoom(F, (Z / h, X / w), order=1)[:Z, :X]
        Hf = np.pad(Hf, ((0, Z - Hf.shape[0]), (0, X - Hf.shape[1])), mode='edge')
        V = ndimage.zoom(validc.astype(np.float32), (Z / h, X / w), order=0)[:Z, :X] > 0.5
        V = np.pad(V, ((0, Z - V.shape[0]), (0, X - V.shape[1])), mode='edge')
        ks = np.arange(-3, 4)
        zz = np.clip(np.rint(Hf)[None] + ks[:, None, None], 0, D - 1).astype(int)
        vals = np.take_along_axis(M, zz, 0).astype(np.float32)
        wsum = vals.sum(0)
        off = np.where(wsum > 0, (vals * ks[:, None, None]).sum(0) / np.maximum(wsum, 1), 0)
        H = ndimage.gaussian_filter(np.rint(Hf) + off, 1.5).astype(np.float32)
    del M, vals, zz
    half = a.layers // 2
    if a.normal:
        from normal_stack import render as nrender
        from v8in_survey import centre
        cy_, cx_ = centre(a.volume, (z0 + z1) // 2)
        inward_plus = (cy_ > (y0 + y1) / 2) if a.axis == 'y' else (cx_ > (x0 + x1) / 2)
        inside = (V > 0) & (H - half >= 0) & (H + half < D)
        stack = nrender(CT, H, inside, inward_plus, layers=a.layers, clip=255).transpose(2, 0, 1).copy()
        open(os.path.join(a.out, 'orientation.txt'), 'w').write(f'normal render, layer index increases inward (axis={a.axis}, inward_is_plus_depth={inward_plus})\n')
    else:
        kk = np.arange(a.layers) - (a.layers - 1) / 2.0
        stack = np.zeros((a.layers, Z, X), np.uint8)
        for r0 in range(0, Z, 256):
            r1 = min(Z, r0 + 256)
            zg, xg = np.mgrid[r0:r1, 0:X].astype(np.float32)
            for i, k in enumerate(kk):
                stack[i, r0:r1] = np.clip(ndimage.map_coordinates(CT, [H[r0:r1] + a.shift + k, zg, xg], order=1, mode='constant'),
                                          0, 255).astype(np.uint8)
        inside = (V > 0) & (H + a.shift - half >= 0) & (H + a.shift + half < D)
        stack *= inside[None]
    del CT
    zp = os.path.join(a.out, 'sheet_00.zarr')
    zarr.open_array(zp, mode='w', shape=stack.shape, chunks=(a.layers, 128, 128), dtype='u1', zarr_format=2,
                    compressor=None, fill_value=0)[:] = stack
    np.save(os.path.join(a.out, 'sheet_00_h.npy'), H)
    np.save(os.path.join(a.out, 'sheet_valid.npy'), inside)
    print('rendered', stack.shape, 'valid', round(float(inside.mean()), 3), flush=True)
    if a.clean_cache:
        import shutil
        shutil.rmtree(os.path.join(zf.CACHE, a.volume.replace('/', '__'), 'L0'), ignore_errors=True)
        # the m7 prediction chunks cached by fetch_pred too (a whole campaign left 7.4 GB of them on PHerc0826)
        shutil.rmtree(os.path.join(zf.CACHE, f'{a.pred}/0'.replace('/', '__')), ignore_errors=True)
    if not a.no_infer:
        subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blockscan_infer.py'),
                         a.out, 's42', 's43', 'ftb'])


if __name__ == '__main__':
    main()
