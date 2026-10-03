"""Text-line score for v8-in maps: a line of writing shows as a horizontal band of ink marks (text lines run
perpendicular to the scroll axis, i.e. along x / the sheet's width) with quieter papyrus above and below.

band(z) = fraction of valid columns with prob > thr in a 2 mm tall band centred at z; the score of a sheet is
max_z [band(z) - mean(band(z - 3 mm), band(z + 3 mm))] (line minus the inter-line gaps), plus the band coverage.
usage: band_score.py <png> <valid.npy or sheet zarr>  (or import band_score)"""
import numpy as np
from scipy import ndimage

PX_MM = 1 / 8.64e-3


def band_score(prob, valid, thr=0.5, band_mm=2.0, gap_mm=3.0):
    v = ndimage.binary_erosion(valid, iterations=48)
    cols = v.any(0)
    if cols.sum() < 100:
        return {'score': 0.0, 'band': 0.0, 'z': None}
    ink = (prob > thr) & v
    per_row = ink[:, cols].sum(1) / np.maximum(v[:, cols].sum(1), 1)      # fraction of valid columns inked, per row
    bw = int(band_mm * PX_MM)
    band = ndimage.uniform_filter1d(per_row.astype(np.float64), bw, mode='nearest')
    g = int(gap_mm * PX_MM)
    Z = len(band)
    best = (-1.0, 0.0, None)
    for z in range(bw // 2, Z - bw // 2):
        nb = [band[k] for k in (z - g, z + g) if bw // 2 <= k < Z - bw // 2]
        if not nb:
            continue
        s = band[z] - float(np.mean(nb))
        if s > best[0]:
            best = (s, float(band[z]), z)
    return {'score': round(best[0], 3), 'band': round(best[1], 3), 'z': best[2]}
