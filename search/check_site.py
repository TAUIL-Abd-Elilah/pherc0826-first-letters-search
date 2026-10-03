"""Quick look at one dense-campaign site face: model agreement, CT-darkness correlation, slab, elongation,
whole-site montage (CT | ens) and a full-res crop at the strongest response (CT / rv2 / d9v2).
usage: check_site.py <site_dir> <fwd|rev> <out_prefix>"""
import sys
import cv2
import numpy as np
from scipy import ndimage
d, face, out = sys.argv[1:4]
sys.argv = ['x', d]
import dense_review as dr  # noqa: E402
ct = cv2.imread(f'{d}/ct.png', 0).astype(np.float32)
im = {k: cv2.imread(f'{d}/{k}_{face}.png', 0).astype(np.float32) / 255 for k in ('rv2', 'd9v2', 'ens')}
m = np.load(f'{d}/sheet_valid.npy') & (ct > 0)
dark = -ndimage.gaussian_filter(ct, 40)
print('corr with CT darkness', {k: round(float(np.corrcoef(x[m], dark[m])[0, 1]), 3) for k, x in im.items()})
print('rv2 vs d9v2', round(float(np.corrcoef(im['rv2'][m], im['d9v2'][m])[0, 1]), 3),
      'slab', dr.slab_fraction(f'{d}/ens_{face}.png', f'{d}/ct.png'), 'elong', dr.elongation(f'{d}/ens_{face}.png', f'{d}/ct.png'))
cv2.imwrite(out + '.png', np.concatenate([cv2.resize(x.astype(np.uint8), (x.shape[1] * 2 // 3, x.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
                                          for x in (ct, im['ens'] * 255)], 1))
e = ndimage.gaussian_filter(im['ens'], 20); y, x = np.unravel_index(np.argmax(e * m), e.shape)
H, W = ct.shape
y0, x0 = max(0, min(y - 350, H - 700)), max(0, min(x - 500, W - 1000))
crop = [a[y0:y0 + 700, x0:x0 + 1000] for a in (ct, im['rv2'] * 255, im['d9v2'] * 255)]
cv2.imwrite(out + '_crop.png', np.concatenate([np.pad(c.astype(np.uint8), ((0, 4), (0, 4)), constant_values=128) for c in crop], 0))
print('peak', y, x)
