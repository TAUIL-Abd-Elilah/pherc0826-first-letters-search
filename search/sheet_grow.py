"""Sheet tracking by best-first growth over the m7 surface-prediction mask (replaces the slope-based
tracker in blockscan.py, which flattened tilted sheets and cut across them).

M: bool (D, Z, X), depth first. Every column (z, x) has a list of run centres along depth (one per
predicted sheet crossing). A sheet = a height map h(z, x) grown from a seed run: from each claimed
column, neighbours take the run centre nearest to the predicted height (current height + local slope),
if within `tol` voxels; growth is best-first by that deviation, so the cleanest continuation wins at
merges. Done on a coarse column grid (step `s`), then upsampled and snapped to the nearest full-res run
centre (within 3 vox). Unreached columns are filled by smooth interpolation and reported as invalid.
"""
import heapq

import numpy as np
from scipy import ndimage


def run_centres(Mc, kmax=48):
    """Mc: bool (D, h, w) -> float32 (kmax, h, w) run centres along depth, NaN padded."""
    D = Mc.shape[0]
    p = np.pad(Mc, ((1, 1), (0, 0), (0, 0))).astype(np.int8)
    d = np.diff(p, axis=0)
    starts = np.argwhere(d == 1)          # (k, 3): depth, y, x
    ends = np.argwhere(d == -1)
    out = np.full((kmax,) + Mc.shape[1:], np.nan, np.float32)
    # starts and ends are both sorted by (depth, y, x) -> regroup per column
    order_s = np.lexsort((starts[:, 0], starts[:, 2], starts[:, 1]))
    order_e = np.lexsort((ends[:, 0], ends[:, 2], ends[:, 1]))
    s, e = starts[order_s], ends[order_e]
    cen = (s[:, 0] + e[:, 0] - 1) / 2.0
    col = s[:, 1] * Mc.shape[2] + s[:, 2]
    first = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    rank = np.arange(len(col)) - np.repeat(first, np.diff(np.r_[first, len(col)]))
    ok = rank < kmax
    out[rank[ok], s[ok, 1], s[ok, 2]] = cen[ok]
    return out


def grow(C, seed_yx, seed_h, tol=4.0):
    """C: (k, h, w) run centres. Returns height map (h, w) with NaN where not reached."""
    k, h, w = C.shape
    H = np.full((h, w), np.nan, np.float32)
    G = np.zeros((h, w, 2), np.float32)             # local slope estimate (dh/dy, dh/dx)
    y0, x0 = seed_yx
    H[y0, x0] = seed_h
    heap = []

    def push_neigh(y, x):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and np.isnan(H[yy, xx]):
                pred = H[y, x] + G[y, x, 0] * dy + G[y, x, 1] * dx
                c = C[:, yy, xx]
                c = c[np.isfinite(c)]
                if c.size == 0:
                    continue
                j = int(np.argmin(np.abs(c - pred)))
                dev = abs(c[j] - pred)
                if dev <= tol:
                    heapq.heappush(heap, (dev, yy, xx, float(c[j]), y, x, dy, dx))

    push_neigh(y0, x0)
    while heap:
        dev, y, x, hv, py, px, dy, dx = heapq.heappop(heap)
        if not np.isnan(H[y, x]):
            continue
        H[y, x] = hv
        g = G[py, px].copy()
        if dy:
            g[0] = 0.7 * g[0] + 0.3 * (hv - H[py, px]) * dy
        else:
            g[1] = 0.7 * g[1] + 0.3 * (hv - H[py, px]) * dx
        G[y, x] = g
        push_neigh(y, x)
    return H


def fill_smooth(H, sigma=1.0):
    valid = np.isfinite(H)
    if not valid.any():
        return H, valid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    F = H[tuple(idx)]
    return ndimage.gaussian_filter(F, sigma), valid


def track_sheets(M, s=4, tol=4.0, min_cover=0.15, dedupe_px=3.0):
    """Return list of (full-res height map, full-res valid mask) for every sheet crossing the centre column."""
    D, Z, X = M.shape
    Mc = M[:, ::s, ::s]
    C = run_centres(Mc)
    h, w = Mc.shape[1:]
    seed_cols = [(h * i // 4, w * j // 4) for i in (2, 1, 3) for j in (2, 1, 3)]
    claimed = np.zeros((0, h, w), np.float32)          # coarse height maps already grown
    Cfull = None
    out = []
    jobs = []
    for (cy, cx) in seed_cols:
        for sd in C[:, cy, cx][np.isfinite(C[:, cy, cx])]:
            jobs.append((cy, cx, float(sd)))
    for cy, cx, sd in jobs:
        # skip seeds already lying on a grown sheet
        if claimed.shape[0] and np.any(np.abs(claimed[:, cy, cx] - sd) < 2.0):
            continue
        Hc = grow(C, (cy, cx), sd, tol=tol)
        cover = float(np.isfinite(Hc).mean())
        if cover < min_cover:
            continue
        claimed = np.concatenate([claimed, np.where(np.isfinite(Hc), Hc, np.nan)[None]], 0)
        F, validc = fill_smooth(Hc)
        Hf = ndimage.zoom(F, (Z / h, X / w), order=1)[:Z, :X]
        Hf = np.pad(Hf, ((0, Z - Hf.shape[0]), (0, X - Hf.shape[1])), mode='edge')
        V = ndimage.zoom(validc.astype(np.float32), (Z / h, X / w), order=0)[:Z, :X] > 0.5
        V = np.pad(V, ((0, Z - V.shape[0]), (0, X - V.shape[1])), mode='edge')
        # snap to the nearest full-res mask run centre within +-3 vox (search a small depth window)
        ks = np.arange(-3, 4)
        zz = np.clip(np.rint(Hf)[None] + ks[:, None, None], 0, D - 1).astype(int)
        vals = np.take_along_axis(M, zz, 0).astype(np.float32)
        wsum = vals.sum(0)
        off = np.where(wsum > 0, (vals * ks[:, None, None]).sum(0) / np.maximum(wsum, 1), 0)
        Hs = ndimage.gaussian_filter(np.rint(Hf) + off, 1.5)
        dup = False
        for H0, V0 in out:
            both = V & V0
            if both.mean() > 0.05 and np.mean(np.abs(Hs - H0)[both] < dedupe_px) > 0.5:
                dup = True
                break
        if dup:
            continue
        out.append((Hs.astype(np.float32), V))
    return out
