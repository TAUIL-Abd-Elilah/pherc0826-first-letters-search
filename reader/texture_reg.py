"""Model-free check of the 2.4 um <-> 9 um canvas correspondence.
For one segment, read matching windows of the organisers' ~2.4 um and ~9 um surface volumes (same segment, two
scans), map the 2.4 um window onto the 9 um grid with the SAME stretch pack9 uses for the canon labels (canvas
extents matched, cv2 INTER_AREA), and register the papyrus texture (mean of the middle layers, high-passed) by NCC
over +-M px. No ink model is involved. Reports per window the label-equivalent offset = the shift that moves
stretched 2.4 um content onto the 9 um texture (same convention as align9/realign 'total offset', e.g. (-6,-6)).
3 x 3 windows at 25/50/75 % of the canvas show whether the offset is a constant translation or grows (extent).
usage: texture_reg.py <scroll>/segments/<seg> [win=256] [M=24]"""
import os, re, sys, urllib.request
import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_tifxyz import ChunkStore  # noqa: E402

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'


def box(st, z0, z1, y0, y1, x0, x1):
    cz, cy, cx = st.chunks
    keys = [(a, b, c) for a in range(z0 // cz, (z1 - 1) // cz + 1) for b in range(y0 // cy, (y1 - 1) // cy + 1)
            for c in range(x0 // cx, (x1 - 1) // cx + 1)]
    st.prefetch(keys)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    for a, b, c in keys:
        arr = st.cache[(a, b, c)]
        za, ya, xa = a * cz, b * cy, c * cx
        s0, s1 = max(z0, za), min(z1, za + cz); r0, r1 = max(y0, ya), min(y1, ya + cy); q0, q1 = max(x0, xa), min(x1, xa + cx)
        out[s0 - z0:s1 - z0, r0 - y0:r1 - y0, q0 - x0:q1 - x0] = arr[s0 - za:s1 - za, r0 - ya:r1 - ya, q0 - xa:q1 - xa]
    st.cache.clear()
    return out


def hp(a, s=8.0):
    a = a.astype(np.float32)
    return a - ndimage.gaussian_filter(a, s)


def ncc(a, b):
    m = (a != 0) & (b != 0)
    if m.sum() < 2000:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))


def main():
    seg = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={seg}/surface-volumes/&delimiter=/').read().decode()
    svs = re.findall(r'<Prefix>[^<]*/surface-volumes/([^<]*)/</Prefix>', lst)
    s9 = [x for x in svs if re.match(r'(8\.6|9\.3)', x)][0]
    s24 = [x for x in svs if re.match(r'2\.\d+um', x)][0]
    st9 = ChunkStore(f'{B}/{seg}/surface-volumes/{s9}', max_cached=100000, workers=16)
    st24 = ChunkStore(f'{B}/{seg}/surface-volumes/{s24}', max_cached=100000, workers=16)
    L9, H9, W9 = st9.shape; L24, H24, W24 = st24.shape
    sy, sx = H24 / H9, W24 / W9                      # pack9's stretch: canvases matched edge to edge
    print(seg, '| 9um', s9[:12], st9.shape, '| 2.4um', s24[:12], st24.shape, '| ratio', round(sy, 4), round(sx, 4), flush=True)
    m9 = L9 // 2; m24 = L24 // 2
    k9 = 2; k24 = int(round(k9 * (L24 / L9) * 0 + k9 * 3.9))   # same physical depth band (~ +-2 layers at 9 um)
    rows = []
    for fy in (0.25, 0.5, 0.75):
        for fx in (0.25, 0.5, 0.75):
            cy, cx = int(H9 * fy), int(W9 * fx)
            y0, y1, x0, x1 = cy - win // 2, cy + win // 2, cx - win // 2, cx + win // 2
            t9 = box(st9, m9 - k9, m9 + k9 + 1, y0, y1, x0, x1).astype(np.float32).mean(0)
            if (t9 > 0).mean() < 0.8:
                print('  window', fy, fx, 'mostly empty at 9 um, skipped', flush=True); continue
            Y0, Y1 = int(np.floor((y0 - M) * sy)), int(np.ceil((y1 + M) * sy)); X0, X1 = int(np.floor((x0 - M) * sx)), int(np.ceil((x1 + M) * sx))
            if Y0 < 0 or X0 < 0 or Y1 > H24 or X1 > W24:
                print('  window', fy, fx, 'too close to the 2.4 um edge, skipped', flush=True); continue
            t24 = box(st24, m24 - k24, m24 + k24 + 1, Y0, Y1, X0, X1).astype(np.float32).mean(0)
            # onto the 9 um grid of the window +- M (sub-pixel exact for the stretch: resize the exact source span)
            t24s = cv2.resize(t24, (win + 2 * M, win + 2 * M), interpolation=cv2.INTER_AREA)
            a = hp(t9) * (t9 > 0)
            b = hp(t24s) * (t24s > 0)
            g = np.full((2 * M + 1, 2 * M + 1), np.nan, np.float32)
            for dy in range(-M, M + 1):
                for dx in range(-M, M + 1):
                    g[dy + M, dx + M] = ncc(a, b[M + dy:M + dy + win, M + dx:M + dx + win])
            i, j = np.unravel_index(np.nanargmax(g), g.shape)
            dy, dx = i - M, j - M
            # stretched 2.4 um content at (y+dy, x+dx) matches 9 um texture at (y, x) -> move it by (-dy, -dx)
            off = (-int(dy), -int(dx))
            sharp = float(g[i, j] - np.nanmedian(g))
            rows.append((fy, fx, off, round(float(g[i, j]), 3), round(sharp, 3)))
            print(f'  window y{fy:.2f} x{fx:.2f}: label-equivalent offset {off}  peak NCC {g[i, j]:.3f}  peak-median {sharp:.3f}', flush=True)
    if rows:
        o = np.array([r[2] for r in rows if r[4] > 0.05])
        if len(o):
            print('MEDIAN offset (windows with peak-median > 0.05):', np.median(o, 0).tolist(), 'n', len(o),
                  '| spread y', o[:, 0].min(), '..', o[:, 0].max(), ' x', o[:, 1].min(), '..', o[:, 1].max(), flush=True)


if __name__ == '__main__':
    main()
