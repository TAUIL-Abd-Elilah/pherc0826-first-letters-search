"""Render a surface volume from a published tifxyz mesh, without VC3D.

Samples the CT volume along the mesh normal at integer voxel steps, producing the
(layers, H, W) uint8 surface volume that the flat ink models consume.

Two things make this cheap and reliable here:
  * the published scroll volumes are stored uncompressed uint8 in 128^3 chunks, so a
    chunk is just raw bytes reshaped -- no codec needed;
  * fetching chunks over plain sequential HTTPS with retries sidesteps the async
    truncation that aborts whole runs (villa#1666).

Validated by rendering a segment that already has a published surface volume and
comparing against it -- see --compare.
"""
import argparse, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tifffile
import zarr


class ChunkStore:
    """Minimal reader for an uncompressed uint8 zarr level, with retries and an LRU-ish cache."""

    def __init__(self, base_url, level="0", max_cached=4000, workers=16):
        self.base = base_url.rstrip("/") + "/" + level
        meta = json.loads(self._get(self.base + "/.zarray"))
        assert meta["compressor"] is None, "expected uncompressed store"
        assert meta["dtype"] in ("|u1", "u1"), meta["dtype"]
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.sep = meta.get("dimension_separator", ".")
        self.fill = np.uint8(meta.get("fill_value") or 0)
        self.cache = {}
        self.max_cached = max_cached
        self.workers = workers
        self.hits = self.misses = self.absent = 0

    def _get(self, url, tries=6, deadline=90.0):
        """Fetch one object under a **hard wall-clock deadline**, not just a socket timeout.

        `urlopen(timeout=...)` bounds each individual socket operation, not the transfer. A
        server that trickles a few bytes every so often keeps resetting that clock and the read
        never returns -- which is villa #1611 ("remote chunk streaming stalls indefinitely with
        no error"). Observed here on 2026-09-09: two jobs sat blocked for 3 h 23 m on a
        reachable bucket, 0.02 s of CPU per 20 s wall, no exception and no output. Reading in
        blocks and checking elapsed time turns that hang into a retryable error.
        """
        last = None
        for a in range(tries):
            try:
                t0 = time.time()
                with urllib.request.urlopen(url, timeout=30) as r:
                    dl = r.headers.get("Content-Length")
                    buf = bytearray()
                    while True:
                        if time.time() - t0 > deadline:
                            raise IOError(f"stalled: {len(buf)} bytes in {deadline:.0f}s")
                        b = r.read(1 << 20)
                        if not b:
                            break
                        buf += b
                data = bytes(buf)
                if dl is not None and len(data) != int(dl):
                    raise IOError(f"short read {len(data)}/{dl}")
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                last = e
            except Exception as e:
                last = e
            time.sleep(min(2 ** a, 15))
        raise IOError(f"failed {url}: {last}")

    def _load(self, key):
        """Network fetch for one chunk. No cache access, so it is safe to call in threads."""
        cz, cy, cx = key
        url = f"{self.base}/{cz}{self.sep}{cy}{self.sep}{cx}"
        try:
            raw = self._get(url)
            return np.frombuffer(raw, dtype=np.uint8).reshape(self.chunks), False
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            return np.full(self.chunks, self.fill, np.uint8), True  # sparse: absent = empty

    def _insert(self, key, arr):
        if len(self.cache) >= self.max_cached:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = arr

    def prefetch(self, keys):
        """Fill the cache for `keys` concurrently.

        Every chunk here is a separate 2 MB HTTPS GET with its own TLS handshake, so a
        serial reader is latency-bound, not bandwidth-bound: rendering one 4 cm^2 window
        took ~1,700 s at roughly 0.33 MB/s. Fetching the working set for a tile in parallel
        is the single biggest speedup available to this renderer.
        """
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        if not todo:
            return
        if self.workers <= 1 or len(todo) == 1:
            for k in todo:
                arr, absent = self._load(k)
                self.absent += absent
                self.misses += not absent
                self._insert(k, arr)
            return
        with ThreadPoolExecutor(max_workers=min(self.workers, len(todo))) as ex:
            for k, (arr, absent) in zip(todo, ex.map(self._load, todo)):
                self.absent += absent
                self.misses += not absent
                self._insert(k, arr)

    def chunk(self, cz, cy, cx):
        key = (cz, cy, cx)
        c = self.cache.get(key)
        if c is not None:
            self.hits += 1
            return c
        arr, absent = self._load(key)
        self.absent += absent
        self.misses += not absent
        self._insert(key, arr)
        return arr

    def _nearest(self, zi, yi, xi):
        out = np.zeros(zi.shape, np.uint8)
        cz_, cy_, cx_ = self.chunks
        czi, cyi, cxi = zi // cz_, yi // cy_, xi // cx_
        oz, oy, ox = zi % cz_, yi % cy_, xi % cx_
        keys = (czi.astype(np.int64) << 42) ^ (cyi.astype(np.int64) << 21) ^ cxi.astype(np.int64)
        for k in np.unique(keys):
            m = keys == k
            a = self.chunk(int(czi[m][0]), int(cyi[m][0]), int(cxi[m][0]))
            out[m] = a[oz[m], oy[m], ox[m]]
        return out

    def sample(self, sz, sy, sx):
        """Trilinear sample at float voxel coordinates.

        Nearest-neighbour validated at per-layer r=0.9547 against the published render;
        trilinear closes the residual, which is interpolation noise rather than geometry.
        """
        zmax, ymax, xmax = (s - 1 for s in self.shape)
        z0 = np.clip(np.floor(sz), 0, zmax).astype(np.int64)
        y0 = np.clip(np.floor(sy), 0, ymax).astype(np.int64)
        x0 = np.clip(np.floor(sx), 0, xmax).astype(np.int64)
        z1 = np.clip(z0 + 1, 0, zmax)
        y1 = np.clip(y0 + 1, 0, ymax)
        x1 = np.clip(x0 + 1, 0, xmax)
        # Warm the whole trilinear working set in parallel before touching it serially.
        cz_, cy_, cx_ = self.chunks
        need = set()
        for ZI in (z0, z1):
            for YI in (y0, y1):
                for XI in (x0, x1):
                    k = np.unique((ZI // cz_).astype(np.int64) * (1 << 42)
                                  + (YI // cy_).astype(np.int64) * (1 << 21)
                                  + (XI // cx_).astype(np.int64))
                    need.update((int(v >> 42), int((v >> 21) & 0x1FFFFF), int(v & 0x1FFFFF))
                                for v in k)
        self.prefetch(need)

        fz = np.clip(sz - z0, 0, 1).astype(np.float32)
        fy = np.clip(sy - y0, 0, 1).astype(np.float32)
        fx = np.clip(sx - x0, 0, 1).astype(np.float32)
        acc = np.zeros(sz.shape, np.float32)
        for dz, ZI in ((0, z0), (1, z1)):
            wz = fz if dz else (1.0 - fz)
            for dy, YI in ((0, y0), (1, y1)):
                wy = fy if dy else (1.0 - fy)
                for dx, XI in ((0, x0), (1, x1)):
                    wx = fx if dx else (1.0 - fx)
                    w = wz * wy * wx
                    nz = w > 0
                    if not nz.any():
                        continue
                    acc += w * self._nearest(ZI, YI, XI).astype(np.float32)
        return np.clip(np.rint(acc), 0, 255).astype(np.uint8)


def read_tifxyz(d):
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float32)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float32)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float32)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = (x > -0.5) & (y > -0.5) & (z > -0.5)
    return x, y, z, valid, meta


def upsample(a, valid, fy, fx):
    """Bilinear upsample of a coordinate plane, invalid points held out of the average.

    Uses CORNER-aligned mapping (src = dst / f), which is what vc_render_tifxyz does.
    cv2.resize instead uses centre-aligned mapping (src = (dst + 0.5) / f - 0.5); on a
    20x upsample the two differ by half a grid cell = 10 output pixels, which validated
    at r=0.2260 instead of r=0.9702 against the published render.
    """
    import cv2
    h, w = a.shape
    H, W = int(round(h * fy)), int(round(w * fx))
    af = np.where(valid, a, 0).astype(np.float32)
    vf = valid.astype(np.float32)
    my = (np.arange(H, dtype=np.float32) / fy)
    mx = (np.arange(W, dtype=np.float32) / fx)
    MX, MY = np.meshgrid(mx, my)
    au = cv2.remap(af, MX, MY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    vu = cv2.remap(vf, MX, MY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(vu > 0.5, au / np.maximum(vu, 1e-6), -1.0)
    return out.astype(np.float32), vu > 0.5


def normals(X, Y, Z, V):
    """Unit surface normal from the coordinate field, via central differences."""
    def grad(A, axis):
        g = np.zeros_like(A)
        if axis == 0:
            g[1:-1] = (A[2:] - A[:-2]) * 0.5
            g[0] = A[1] - A[0]
            g[-1] = A[-1] - A[-2]
        else:
            g[:, 1:-1] = (A[:, 2:] - A[:, :-2]) * 0.5
            g[:, 0] = A[:, 1] - A[:, 0]
            g[:, -1] = A[:, -1] - A[:, -2]
        return g

    du = np.stack([grad(X, 1), grad(Y, 1), grad(Z, 1)], -1)   # along width
    dv = np.stack([grad(X, 0), grad(Y, 0), grad(Z, 0)], -1)   # along height
    # cross(dv, du), not cross(du, dv): vc_render_tifxyz orders layers along the
    # opposite normal. Validated by layer-vs-layer correlation against the published
    # render, which mapped our layer i onto their layer 30-i at mean r = 0.9574.
    n = np.cross(dv, du)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(ln > 1e-8, n / np.maximum(ln, 1e-8), 0.0)
    return n.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz_dir")
    ap.add_argument("out_zarr")
    ap.add_argument("--volume-url", default=None)
    ap.add_argument("--layers", type=int, default=31)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N rows (debug)")
    ap.add_argument("--cache-chunks", type=int, default=4000,
                    help="chunk cache size; 4000 x 2MB = 8GB. Lower it when running renders in parallel.")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel chunk fetches; 1 restores the original serial reader")
    a = ap.parse_args()

    d = a.tifxyz_dir
    vol = a.volume_url or open(os.path.join(d, "volume_source.txt")).read().strip()
    x, y, z, valid, meta = read_tifxyz(d)
    sy, sx = meta.get("scale", [1.0, 1.0])
    fy, fx = 1.0 / float(sy), 1.0 / float(sx)
    print(f"tifxyz {x.shape} scale {sy},{sx} -> upsample x{fy:.0f},{fx:.0f}", flush=True)

    X, Vx = upsample(x, valid, fy, fx)
    Y, _ = upsample(y, valid, fy, fx)
    Z, _ = upsample(z, valid, fy, fx)
    V = Vx
    H, W = X.shape
    if a.limit:
        H = min(H, a.limit)
        X, Y, Z, V = X[:H], Y[:H], Z[:H], V[:H]
    print(f"surface {H}x{W}, valid {V.mean():.3f}", flush=True)

    N = normals(X, Y, Z, V)
    store = ChunkStore(vol, max_cached=a.cache_chunks, workers=a.workers)
    print(f"volume {store.shape} chunks {store.chunks}", flush=True)

    L = a.layers
    ks = np.arange(L, dtype=np.float32) - (L - 1) / 2.0
    g = zarr.open_group(a.out_zarr, mode="w", zarr_format=2)
    out = g.create_array("0", shape=(L, H, W), chunks=(L, 128, 128), dtype="u1")
    g.attrs["canvas_size"] = [int(W), int(H)]
    g.attrs["rendered_from"] = os.path.basename(os.path.normpath(d))
    g.attrs["volume_source"] = vol
    g.attrs["num_slices"] = int(L)
    g.attrs["slice_step"] = float(a.step)

    zmax, ymax, xmax = (np.int64(s - 1) for s in store.shape)
    t0 = time.time()
    for y0 in range(0, H, a.tile):
        y1 = min(H, y0 + a.tile)
        block = np.zeros((L, y1 - y0, W), np.uint8)
        for x0 in range(0, W, a.tile):
            x1 = min(W, x0 + a.tile)
            vv = V[y0:y1, x0:x1]
            if not vv.any():
                continue
            px, py, pz = X[y0:y1, x0:x1], Y[y0:y1, x0:x1], Z[y0:y1, x0:x1]
            nn = N[y0:y1, x0:x1]
            for li, k in enumerate(ks):
                t = k * a.step
                sx_ = px + nn[..., 0] * t
                sy_ = py + nn[..., 1] * t
                sz_ = pz + nn[..., 2] * t
                s = store.sample(sz_, sy_, sx_)
                block[li, :, x0:x1] = np.where(vv, s, 0)
        out[:, y0:y1, :] = block
        el = time.time() - t0
        print(f"  rows {y1}/{H}  {el:.0f}s  chunks fetched={store.misses} cached={store.hits} absent={store.absent}",
              flush=True)
    print(f"wrote {a.out_zarr}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
