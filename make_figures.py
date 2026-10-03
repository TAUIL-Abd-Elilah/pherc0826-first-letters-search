"""Figures for the README, from results/ plus the campaign directory (maps, ct.png, mosaics).
usage: CAMPAIGN_DIR=... ROUTE24_CACHE=... python make_figures.py"""
import json, os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'search'))
C = os.environ.get('CAMPAIGN_DIR', 'campaign_0826')
R = os.path.join(HERE, 'results')
F = os.path.join(HERE, 'figures')
os.makedirs(F, exist_ok=True)
PX = 9.362e-3                                                    # mm per voxel, PHerc0826 9.362 um scan
LEADS = [('z3900_x0_d1', 'fwd'), ('z2600_x0_d1', 'fwd'), ('z5200_x0_d0', 'rev'), ('z3900_x1_d0', 'rev'),
         ('z9100_x2_d0', 'rev'), ('z14300_x2_d2', 'fwd'), ('z13000_x0_d0', 'fwd')]


def site_map():
    import zfetch as zf
    from dense0826 import VOL
    secs = {s['z']: s for s in json.load(open(os.path.join(R, 'campaign', 'sections.json')))}
    plan = json.load(open(os.path.join(R, 'campaign', 'sites_v2.json')))
    lead_names = {n for n, _ in LEADS}
    m = zf.meta(VOL, 3)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, zc in zip(axs, (2600, 6500, 10400, 14300)):
        sl = zf.read_box(VOL, 3, (zc // 8, 0, 0), (zc // 8 + 1, m['shape'][1], m['shape'][2]))[0]
        ax.imshow(sl, cmap='gray', extent=(0, sl.shape[1] * 8 * PX, sl.shape[0] * 8 * PX, 0))
        for s in plan:
            if s['z'] != zc:
                continue
            col = 'red' if s['name'] in lead_names else 'deepskyblue'
            ax.add_patch(Rectangle(((s['lat'] - 800) * PX, (s['dep'] - 250) * PX), 1600 * PX, 500 * PX, fill=False, ec=col, lw=1.4))
        ax.set_title(f'z = {zc} ({zc * PX:.0f} mm)', fontsize=10)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ys = [ (s['dep']) * PX for s in plan if s['z'] == zc]
        if ys:
            ax.set_ylim(max(ys) + 15, min(ys) - 15); ax.set_xlim(secs[zc]['cx'] * PX - 30, secs[zc]['cx'] * PX + 30)
    fig.suptitle('PHerc0826: every grid site (12 x 15 mm of one tracked sheet each, depth axis y). Red = leads sent to the controls.', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(F, '01_site_map.png'), dpi=110); plt.close(fig)


def crop_at_peak(d, face, size=640):
    ct = cv2.imread(os.path.join(d, 'ct.png'), 0)
    e = cv2.imread(os.path.join(d, f'ens_{face}.png'), 0)
    v = np.load(os.path.join(d, 'sheet_valid.npy'))
    g = cv2.GaussianBlur(e.astype(np.float32) * v, (0, 0), 20)
    y, x = np.unravel_index(np.argmax(g), g.shape)
    H, W = ct.shape
    y0, x0 = int(np.clip(y - size // 2, 0, H - size)), int(np.clip(x - size // 2, 0, W - size))
    return ct[y0:y0 + size, x0:x0 + size], e[y0:y0 + size, x0:x0 + size]


def leads():
    fig, axs = plt.subplots(2, len(LEADS), figsize=(2.3 * len(LEADS), 5))
    for i, (n, face) in enumerate(LEADS):
        ct, e = crop_at_peak(os.path.join(C, n), face)
        axs[0, i].imshow(ct, cmap='gray'); axs[1, i].imshow(e, cmap='magma', vmin=0, vmax=255)
        axs[0, i].set_title(f'{n}\n{face}', fontsize=8)
        for a in axs[:, i]:
            a.set_xticks([]); a.set_yticks([])
        axs[1, i].plot([20, 20 + 1 / PX], [610, 610], color='w', lw=2)
    axs[0, 0].set_ylabel('CT (sheet surface)', fontsize=9); axs[1, 0].set_ylabel('Reader v2 + d9v2', fontsize=9)
    fig.suptitle('The seven leads: 6 x 6 mm around each strongest response (white bar = 1 mm). CT above, model map below.', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(F, '02_leads.png'), dpi=110); plt.close(fig)


def mosaic():
    b = os.path.join(C, '_ext', 'z13000_x0_d0')
    ct = cv2.imread(os.path.join(b, 'mosaic_ct.png'), 0); e = cv2.imread(os.path.join(b, 'mosaic_ens_fwd.png'), 0)
    px3 = 3 * PX
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    ext = (0, ct.shape[1] * px3, ct.shape[0] * px3, 0)
    axs[0].imshow(ct, cmap='gray', extent=ext); axs[1].imshow(e, cmap='magma', extent=ext, vmin=0, vmax=255)
    axs[0].set_title('CT of one tracked sheet (8 tiles + site, 7/8 on the same sheet)', fontsize=9)
    axs[1].set_title('Reader v2 + d9v2, front face: the two site marks (centre) stay isolated', fontsize=9)
    for a in axs:
        a.set_xlabel('lateral (mm)'); a.set_ylabel('z (mm)')
    fig.tight_layout(); fig.savefig(os.path.join(F, '03_context_z13000_x0_d0.png'), dpi=110); plt.close(fig)


def controls():
    nb = json.load(open(os.path.join(R, 'campaign', 'neighbour_control.json')))
    names = [n for n, _ in LEADS if n in nb] + ['z11700_x1_d1']
    site_l, best_l, lab = [], [], []
    for n in names:
        s = nb[n]; ref = s['site_ref']
        site_l.append(ref['in_blobs'] - ref['around'])
        lifts = [r['in_blobs'] - r['around'] for r in s['neighbours'] if r.get('in_blobs') is not None and r.get('around') is not None]
        best_l.append(max(lifts))
        lab.append(n + (' (negative control)' if n == 'z11700_x1_d1' else ''))
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(y - 0.2, site_l, 0.4, label='site sheet (k = 0)', color='tab:red')
    ax.barh(y + 0.2, best_l, 0.4, label='strongest of 6 neighbouring windings (k = -3..+3, both faces)', color='tab:gray')
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('response lift under the site blobs (mean inside - mean in a 0.3 mm ring around them)')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False); ax.set_title('Neighbour-winding control', fontsize=10)
    fig.set_size_inches(8.6, 4.4); fig.tight_layout(); fig.savefig(os.path.join(F, '04_neighbour_control.png'), dpi=110); plt.close(fig)


def periodicity():
    rows = json.load(open(os.path.join(R, 'campaign', 'row_period.json')))
    def vals(sel):
        return [r['line_frac'] for r in rows if sel(r) and r.get('line_frac') is not None]
    groups = [('known text, model maps, along z', vals(lambda r: r['set'] == 'known' and r['item'].endswith('|z') and '|human|' not in r['item'])),
              ('known text, model maps, across', vals(lambda r: r['set'] == 'known' and r['item'].endswith('|lateral') and '|human|' not in r['item'])),
              ('known text, human labels, along z', vals(lambda r: r['set'] == 'known' and r['item'].endswith('|z') and '|human|' in r['item'])),
              ('PHerc0826 sites, along z (12 mm windows)', vals(lambda r: r['set'] == '0826')),
              ('PHerc0826 sites, across', [r['lateral_frac'] for r in rows if r['set'] == '0826' and r.get('lateral_frac') is not None])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(groups):
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=12, alpha=0.7)
        if v:
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color='k', lw=2)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('share of profile power at 3.5-8 mm periods (text-line pitch)')
    ax.set_title('Line-pitch test: known text (PHerc0841, PHerc0139) vs PHerc0826', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(F, '05_line_pitch.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    for fn in (leads, mosaic, controls, periodicity, site_map):
        try:
            fn(); print('ok', fn.__name__, flush=True)
        except Exception as ex:
            print('FAILED', fn.__name__, repr(ex)[:300], flush=True)
