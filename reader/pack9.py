"""Corpus builder for a native ~9 um ink model.

For one segment with a published ~9 um surface volume and a published ~2.4 um ink prediction (the organisers'
recipe), write $D9_DATA/<scroll>__<segment>/:
  sv.zarr     21 layers (uint8, zarr v2, chunks 21x128x128) of the ~9 um surface volume: layers [s-2, s+19) where
              s is the start of the stock inference's 17-layer centre crop, so that crop sits at layers 2..18 here
              and training can jitter it by +-2. Only 128 px chunks within ~1 mm of predicted ink, plus a random 15 %
              of the others, are fetched; the rest stay 0 (no disk) and are excluded by the valid mask.
  canon.npy   the ~2.4 um prediction area-downsampled onto the ~9 um grid (float16, NaN outside it)
  human.npy   published human ink labels on the same grid, if any
  meta.json   sources, scale, shapes, chunks fetched
The grid offset between the two canvases is measured separately (align9.py).
usage: [D9_FULL=1] pack9.py <scroll>/<segment> [...]   (D9_FULL=1 for test segments)"""
import json, os, re, sys, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import tifffile
import zarr
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_tifxyz import ChunkStore  # noqa: E402

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
OUT = os.environ.get('D9_DATA', 'corpus_9um')
KEEP, WIN = 21, 17
NBLK = int(os.environ.get('D9_BLOCKS', '100'))
FULL = os.environ.get('D9_FULL') == '1'


def um(name):
    return float(re.match(r'(\d+\.\d+)um', name).group(1))


def get(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + '.part')
        os.replace(path + '.part', path)
    return path


def main(arg):
    scroll, seg = arg.split('/', 1)
    d = os.path.join(OUT, f'{scroll}__{seg}'); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, 'meta.json')):
        return
    base = f'{B}/{scroll}/segments/{seg}'
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/surface-volumes/&delimiter=/').read().decode()
    svs = [x for x in re.findall(r'<Prefix>[^<]*/surface-volumes/([^<]*)/</Prefix>', lst) if re.match(r'(8\.6|9\.3)', x)]
    assert len(svs) == 1, svs
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/ink-detection/').read().decode()
    keys = [k for k in re.findall(r'<Key>([^<]*)</Key>', lst) if re.search(r'-2\.[0-9]+um-', k) and k.endswith('.tif') and '/downsampled/' not in k]
    keys = sorted(keys, key=lambda k: 'new_canon' not in k)
    assert keys, 'no ~2.4 um prediction'
    scale = um(svs[0]) / um(re.search(r'-(2\.\d+um)-', keys[0]).group(1))
    st = ChunkStore(f'{base}/surface-volumes/{svs[0]}', max_cached=4, workers=16)
    L, H, W = st.shape
    s = L // 2 - WIN // 2
    z0 = s - 2
    # canonical prediction -> ~9 um grid
    p = tifffile.imread(get(f'{B}/{keys[0]}', os.path.join(d, 'canon_2p4.tif')))
    if p.ndim == 3:
        p = p[..., 0] if p.shape[-1] in (1, 3, 4) else p[0]
    # stretched onto exactly the ~9 um (H, W) canvas: the two canvases cover the same extent but differ by
    # ~0.3-0.4 % from the nominal pixel-size ratio (dist9/align_pair.py), so a pure rescale would drift ~11 px
    small = cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / (255.0 if p.dtype == np.uint8 else float(p.max() or 1))
    ratio_err = [round(p.shape[0] / scale / H - 1, 5), round(p.shape[1] / scale / W - 1, 5)]
    del p
    os.remove(os.path.join(d, 'canon_2p4.tif'))
    canon = small
    np.save(os.path.join(d, 'canon.npy'), canon.astype(np.float16))
    # chunks to fetch. Test segments: all. Training segments: up to NBLK blocks of 2x2 chunks (256 px), 70 % drawn
    # with weight (ink fraction within ~1 mm + 0.05), 30 % uniformly, so patches can sit anywhere inside a block.
    near = ndimage.binary_dilation(np.nan_to_num(canon) > 0.5, iterations=int(round(1.0 / (um(svs[0]) * 1e-3))))
    ca, cb = (H + 127) // 128, (W + 127) // 128
    if FULL:
        sel = np.ones((ca, cb), bool)
    else:
        ba, bb = (ca + 1) // 2, (cb + 1) // 2
        nb = np.zeros((ba * 256, bb * 256), np.float32); nb[:H, :W] = near
        cv_ = np.zeros((ba * 256, bb * 256), np.float32); cv_[:H, :W] = np.isfinite(canon)
        frac = nb.reshape(ba, 256, bb, 256).mean((1, 3)); cover = cv_.reshape(ba, 256, bb, 256).mean((1, 3))
        ok = np.flatnonzero(cover.ravel() > 0.5)
        rng = np.random.default_rng(zlib.crc32(seg.encode()))
        n1 = min(len(ok), int(NBLK * 0.7)); w = frac.ravel()[ok] + 0.05
        pick = set(rng.choice(ok, n1, replace=False, p=w / w.sum()).tolist()) if n1 else set()
        rest = [i for i in ok if i not in pick]
        pick |= set(rng.choice(rest, min(len(rest), NBLK - len(pick)), replace=False).tolist()) if rest else set()
        sel = np.zeros((ca, cb), bool)
        for i in pick:
            a, b = divmod(i, bb)
            sel[2 * a:2 * a + 2, 2 * b:2 * b + 2] = True
        sel = sel[:ca, :cb]
    zp = os.path.join(d, 'sv.zarr')
    z = zarr.open_array(zp, mode='w', shape=(KEEP, H, W), chunks=(KEEP, 128, 128), dtype='u1', zarr_format=2, compressor=None, fill_value=0)
    ks = [(0, a, b) for a in range(ca) for b in range(cb) if sel[a, b]]

    def fetch(k):
        return k, st._load(k)[0]

    with ThreadPoolExecutor(16) as ex:
        for (c0, a, b), blk in ex.map(fetch, ks):
            y1, x1 = min(H, (a + 1) * 128), min(W, (b + 1) * 128)
            z[:, a * 128:y1, b * 128:x1] = blk[z0:z0 + KEEP, :y1 - a * 128, :x1 - b * 128]
    # human labels
    lst = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/{seg}/ink-labels/').read().decode()
    lz = sorted(set(re.findall(r'<Key>([^<]*inklabels\.zarr)/0/zarr\.json</Key>', lst)))
    lz = [x for x in lz if re.search(r'/(8\.6|9\.3)\d*um-', x)] or lz          # prefer labels drawn on the ~9 um canvas
    human = None
    if lz:
        root = os.path.join(d, 'inklabels.zarr'); os.makedirs(os.path.join(root, '0', 'c', '0'), exist_ok=True)
        get(f'{B}/{lz[-1]}/zarr.json', os.path.join(root, 'zarr.json'))
        get(f'{B}/{lz[-1]}/0/zarr.json', os.path.join(root, '0', 'zarr.json'))
        get(f'{B}/{lz[-1]}/0/c/0/0', os.path.join(root, '0', 'c', '0', '0'))
        lab = ((zarr.open_array(os.path.join(root, '0'), mode='r')[:] > 0) * 255).astype(np.uint8)
        human = cv2.resize(lab, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        np.save(os.path.join(d, 'human.npy'), human.astype(np.float16))
    meta = {'scroll': scroll, 'segment': seg, 'sv_source': svs[0], 'sv_layers_total': L, 'layers_kept': [z0, z0 + KEEP],
            'infer_window_in_kept': [2, 2 + WIN], 'shape': [KEEP, H, W], 'canon_source': keys[0], 'scale': round(scale, 5), 'canvas_ratio_error': ratio_err,
            'chunks_fetched': len(ks), 'chunks_total': int(ca * cb), 'full': FULL, 'canon_ink_frac': round(float(np.nanmean(canon > 0.5)), 4),
            'human_source': lz[-1] if lz else None, 'human_on_9um_canvas': bool(lz and re.search(r'/(8\.6|9\.3)\d*um-', lz[-1]))}
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(scroll, seg[:40], 'chunks', len(ks), '/', ca * cb, 'ink', meta['canon_ink_frac'], 'human', lz[-1].split('/')[-3] if lz else None, flush=True)


if __name__ == '__main__':
    import time
    for a in sys.argv[1:]:
        for attempt in range(6):                     # network drops: retry with backoff (1, 2, 4, 8, 16 min)
            try:
                main(a)
                break
            except Exception as e:
                print('FAILED' if attempt == 5 else 'RETRY', a, repr(e)[:200], flush=True)
                if attempt < 5:
                    time.sleep(60 * 2 ** attempt)
