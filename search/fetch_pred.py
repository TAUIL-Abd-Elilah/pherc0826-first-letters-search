"""Fetch a box from a blosc-compressed zarr v2 (surface predictions), cached on disk."""
import json, os, sys, urllib.request
import numpy as np, numcodecs
from concurrent.futures import ThreadPoolExecutor
import zfetch as zf


def meta(path):
    return json.loads(zf._get(f'{zf.BUCKET}/{path}/.zarray'))


def read_box(path, lo, hi):
    m = meta(path)
    codec = numcodecs.get_codec(m['compressor'])
    ch = np.array(m['chunks'])
    lo, hi = np.maximum(np.array(lo), 0), np.minimum(np.array(hi), m['shape'])
    out = np.zeros(tuple(hi - lo), np.uint8)
    cdir = os.path.join(zf.CACHE, path.replace('/', '__'))
    os.makedirs(cdir, exist_ok=True)
    keys = [(a, b, c) for a in range(lo[0] // ch[0], (hi[0] - 1) // ch[0] + 1)
            for b in range(lo[1] // ch[1], (hi[1] - 1) // ch[1] + 1)
            for c in range(lo[2] // ch[2], (hi[2] - 1) // ch[2] + 1)]

    def get(k):
        f = os.path.join(cdir, '%d_%d_%d' % k)
        if os.path.exists(f):
            raw = open(f, 'rb').read()
        else:
            raw = zf._get(f'{zf.BUCKET}/{path}/' + '/'.join(map(str, k)))
            open(f, 'wb').write(raw)
        if not raw:
            return k, None
        return k, np.frombuffer(codec.decode(raw), np.uint8).reshape(ch)

    with ThreadPoolExecutor(16) as ex:
        for k, blk in ex.map(get, keys):
            if blk is None:
                continue
            c0 = np.array(k) * ch
            a, b = np.maximum(lo, c0), np.minimum(hi, c0 + ch)
            out[tuple(slice(a[i] - lo[i], b[i] - lo[i]) for i in range(3))] = \
                blk[tuple(slice(a[i] - c0[i], b[i] - c0[i]) for i in range(3))]
    return out


if __name__ == '__main__':
    path, level, out = sys.argv[1], sys.argv[2], sys.argv[3]
    lo = list(map(int, sys.argv[4:7])); hi = list(map(int, sys.argv[7:10]))
    a = read_box(f'{path}/{level}', lo, hi)
    np.save(out, a)
    print(out, a.shape, 'nonzero frac', (a > 0).mean(), 'max', a.max())
