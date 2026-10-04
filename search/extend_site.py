"""Follow a campaign site's sheet into the 8 surrounding tiles and read them.
Text continues: if a site's mark is a letter, more marks should appear along the same sheet around it.
Tile (i, j), i, j in {-1, 0, +1}: box centred at z = zc + 1100 i, lateral = lat + 1300 j (200 / 300 vox overlap
with the site box; (0, 0) = the site itself). Seeded at a valid site-sheet pixel inside the overlap, nearest the
tile centre, at the site sheet depth there, so growth starts on the same sheet.
Each tile: bigsheet_v2 (normal render) + read_sheets (rv2 + d9v2 ensemble), render deleted. Then mosaics of CT
and ensemble maps on the joint (z, lateral) canvas: <OUT>/_ext/<site>/mosaic_{ct,ens_<face>}.png, plus
depth consistency at the seams (median |H_tile - H_site| over the overlap, must be small = same sheet).
usage: extend_site.py <site_name> <fwd|rev>"""
import json, os, subprocess, sys
import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dense0826 import D9, OUT as OUT0826  # noqa: E402
import zfetch as zf  # noqa: E402
import campaign_cfg  # noqa: E402

OUT = os.environ.get('CAMPAIGN_DIR', OUT0826)          # PHerc0826 dense campaign by default; any dense_any.py dir works
VOL, PRED, SITES = campaign_cfg.load(OUT)
N = zf.meta(VOL, 0)['shape']


def main():
    name, face = sys.argv[1], sys.argv[2]
    zc, lat, dep, axis = SITES[name]                                       # lateral / depth per the site's axis
    site = os.path.join(OUT, name)
    H0 = np.load(os.path.join(site, 'sheet_00_h.npy')); V0 = np.load(os.path.join(site, 'sheet_valid.npy'))
    sz0, sl0, sd0 = zc - 650, max(0, lat - 800), max(0, dep - 250)          # site box origin (z, lateral, depth)
    base = os.path.join(OUT, '_ext', name)
    tiles = {(0, 0): (site, sz0, sl0, sd0)}
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if (i, j) == (0, 0):
                continue
            tz, tl = zc + 1100 * i, lat + 1300 * j
            # seed: a valid site-sheet pixel INSIDE the site/tile overlap (20 px margin), nearest the tile centre
            bz0, bz1 = max(sz0, tz - 650) + 20, min(sz0 + H0.shape[0], tz + 650) - 20
            bl0, bl1 = max(sl0, tl - 800) + 20, min(sl0 + H0.shape[1], tl + 800) - 20
            zz, ll = np.nonzero(V0[bz0 - sz0:bz1 - sz0, bl0 - sl0:bl1 - sl0])
            if zz.size == 0:
                print('tile', i, j, 'no site sheet in the overlap -> skipped', flush=True)
                continue
            q = int(np.argmin((zz + bz0 - tz) ** 2 + (ll + bl0 - tl) ** 2))
            qz, ql = zz[q] + bz0 - sz0, ll[q] + bl0 - sl0
            sz, sl, sdep = int(sz0 + qz), int(sl0 + ql), float(sd0 + H0[qz, ql])
            box = campaign_cfg.box_for(tz, tl, int(sdep), axis, N)
            d = os.path.join(base, f't{i:+d}{j:+d}')
            os.makedirs(d, exist_ok=True)
            sp = os.path.join(d, 'scores.json')
            if not (os.path.exists(sp) and 'ens_fwd' in json.load(open(sp))):
                if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
                    r = subprocess.run([sys.executable, os.path.join(HERE, 'bigsheet_v2.py'), VOL, PRED, d, '--box', *map(str, box),
                                        '--seed', str(sz), str(sl), str(int(sdep)), '--axis', axis, '--normal', '--no-infer', '--clean-cache'],
                                       capture_output=True, text=True)
                    if not os.path.exists(os.path.join(d, 'sheet_valid.npy')):
                        print('tile', i, j, 'track failed', (r.stdout + r.stderr).strip().splitlines()[-1:], flush=True)
                        continue
                subprocess.run([sys.executable, os.path.join(HERE, 'read_sheets.py'), d, '--no-v8in', '--ensemble', '--delete-render',
                                '--ckpt', f'd9v2={D9}'], capture_output=True, text=True)
            json.dump({'box': box, 'seed': [sz, sl, sdep]}, open(os.path.join(d, 'tile.json'), 'w'))
            tiles[(i, j)] = (d, box[0], box[4], box[2]) if axis == 'y' else (d, box[0], box[2], box[4])   # (dir, z0, lateral0, depth0)
            sc = json.load(open(sp)) if os.path.exists(sp) else {}
            print('tile', i, j, {k: (sc[k]['best_2mm'], sc[k]['band']['score']) for k in ('ens_fwd', 'ens_rev') if k in sc}, flush=True)
    # mosaics on the joint canvas (z rows, lateral cols); later tiles do not overwrite the site
    z_min = min(t[1] for t in tiles.values()); l_min = min(t[2] for t in tiles.values())
    Z = max(t[1] for t in tiles.values()) + 1300 - z_min; L = max(t[2] for t in tiles.values()) + 1600 - l_min
    out = {k: np.zeros((Z, L), np.uint8) for k in ('ct', f'ens_{face}')}
    seams = {}
    order = sorted(tiles, key=lambda t: (t == (0, 0), t))                     # site last = on top
    for t in order:
        d, z0, l0, d0 = tiles[t]
        if not os.path.exists(os.path.join(d, 'ct.png')):
            continue
        H = np.load(os.path.join(d, 'sheet_00_h.npy')); V = np.load(os.path.join(d, 'sheet_valid.npy'))
        if t != (0, 0):
            # same-sheet check on the overlap with the site box (absolute depth)
            oz0, oz1 = max(z0, sz0), min(z0 + H.shape[0], sz0 + H0.shape[0]); ol0, ol1 = max(l0, sl0), min(l0 + H.shape[1], sl0 + H0.shape[1])
            if oz1 > oz0 and ol1 > ol0:
                a = H[oz0 - z0:oz1 - z0, ol0 - l0:ol1 - l0] + d0; b = H0[oz0 - sz0:oz1 - sz0, ol0 - sl0:ol1 - sl0] + sd0
                m = V[oz0 - z0:oz1 - z0, ol0 - l0:ol1 - l0] & V0[oz0 - sz0:oz1 - sz0, ol0 - sl0:ol1 - sl0]
                seams[f'{t[0]:+d}{t[1]:+d}'] = round(float(np.median(np.abs(a - b)[m])), 1) if m.sum() > 500 else None
        for k in out:
            im = cv2.imread(os.path.join(d, ('ct' if k == 'ct' else k) + '.png'), 0)
            if im is None:
                continue
            sub = out[k][z0 - z_min:z0 - z_min + im.shape[0], l0 - l_min:l0 - l_min + im.shape[1]]
            msk = V[:sub.shape[0], :sub.shape[1]]
            sub[msk] = im[:sub.shape[0], :sub.shape[1]][msk]
    for k, im in out.items():
        cv2.imwrite(os.path.join(base, f'mosaic_{k}.png'), cv2.resize(im, (im.shape[1] // 3, im.shape[0] // 3), interpolation=cv2.INTER_AREA))
    json.dump({'site': name, 'face': face, 'seam_median_abs_depth_vox': seams, 'canvas_origin_z_lat': [int(z_min), int(l_min)]},
              open(os.path.join(base, 'summary.json'), 'w'), indent=1)
    print('seams (median |depth diff| vox, small = same sheet):', seams, flush=True)


if __name__ == '__main__':
    main()
