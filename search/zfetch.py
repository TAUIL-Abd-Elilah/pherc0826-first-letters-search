"""Bounded, cached, parallel reads of public uncompressed OME-Zarr v2 volumes.

read_level(url, level) -> whole pyramid level as a numpy array (small levels only)
read_box(url, level, lo, hi) -> sub-block [lo, hi) zyx
Chunks are cached as raw bytes under CACHE/<volume-name>/L<level>/; a byte
counter enforces a per-process cap (ROUTE24_CAP_GB, default 20).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
CACHE = os.environ.get('ROUTE24_CACHE', 'cache')
CAP = int(float(os.environ.get('ROUTE24_CAP_GB', '20')) * 1e9)
_bytes = {'n': 0}


def _get(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return b''
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt == tries - 1:
                raise
        time.sleep(2.0 * (attempt + 1))


def meta(vol, level):
    path = os.path.join(CACHE, vol.replace('/', '__'), f'L{level}', '.zarray')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = _get(f'{BUCKET}/{vol}/{level}/.zarray')
        open(path, 'wb').write(raw)
    m = json.load(open(path))
    assert m['dtype'] == '|u1' and m['compressor'] is None and m.get('dimension_separator') == '/'
    return m


def chunk(vol, level, key, m):
    cz, cy, cx = key
    shape = tuple(m['chunks'])
    path = os.path.join(CACHE, vol.replace('/', '__'), f'L{level}', f'{cz}_{cy}_{cx}.bin')
    if os.path.exists(path):
        raw = open(path, 'rb').read()
    else:
        if _bytes['n'] >= CAP:
            raise RuntimeError('route24 download cap reached')
        raw = _get(f'{BUCKET}/{vol}/{level}/{cz}/{cy}/{cx}')
        _bytes['n'] += len(raw)
        with open(path + '.part', 'wb') as h:
            h.write(raw)
        os.replace(path + '.part', path)
    if not raw:
        return np.zeros(shape, np.uint8)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def read_box(vol, level, lo, hi, workers=16):
    m = meta(vol, level)
    shape, ch = np.asarray(m['shape']), np.asarray(m['chunks'])
    lo, hi = np.maximum(np.asarray(lo, int), 0), np.minimum(np.asarray(hi, int), shape)
    out = np.zeros(tuple(hi - lo), np.uint8)
    keys = [(a, b, c) for a in range(lo[0] // ch[0], (hi[0] - 1) // ch[0] + 1)
            for b in range(lo[1] // ch[1], (hi[1] - 1) // ch[1] + 1)
            for c in range(lo[2] // ch[2], (hi[2] - 1) // ch[2] + 1)]

    def fill(key):
        blk = chunk(vol, level, key, m)
        c0 = np.asarray(key) * ch
        a, b = np.maximum(lo, c0), np.minimum(hi, c0 + ch)
        out[tuple(slice(a[k] - lo[k], b[k] - lo[k]) for k in range(3))] = \
            blk[tuple(slice(a[k] - c0[k], b[k] - c0[k]) for k in range(3))]

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fill, keys))
    return out


def read_level(vol, level, workers=16):
    m = meta(vol, level)
    return read_box(vol, level, (0, 0, 0), m['shape'], workers)


def downloaded():
    return _bytes['n']
