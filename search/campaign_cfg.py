"""Campaign config for the control scripts (neighbors.py, extend_site.py, reread_v8in.py).
A campaign dir is either the PHerc0826 dense campaign (sites_v2.json, depth axis y everywhere) or a dense_any.py
campaign (campaign.json + sites.json with a per-site axis). Sites are returned as (z, lateral, depth, axis): for
axis y lateral = x and depth = y; for axis x lateral = y and depth = x (bigsheet_v2 --seed z lateral depth)."""
import json, os


def load(cdir):
    if os.path.exists(os.path.join(cdir, 'campaign.json')):
        c = json.load(open(os.path.join(cdir, 'campaign.json')))
        sites = {}
        for s in json.load(open(os.path.join(cdir, 'sites.json'))):
            lat, dep = (s['cx'], s['cy']) if s['axis'] == 'y' else (s['cy'], s['cx'])
            sites[s['name']] = (s['z'], lat, dep, s['axis'])
        return c['vol'], c['pred'], sites
    from dense0826 import VOL, PRED
    sites = {s['name']: (s['z'], s['lat'], s['dep'], 'y') for s in json.load(open(os.path.join(cdir, 'sites_v2.json')))}
    return VOL, PRED, sites


def d9(cdir, default):
    """Checkpoint in the campaign's 'd9v2' reader slot (dense_any.py logs it as d9v2_slot; d9v2 otherwise), so the
    controls read with the same models as the campaign."""
    p = os.path.join(cdir, 'campaign.json')
    return json.load(open(p)).get('d9v2_slot', default) if os.path.exists(p) else default


def box_for(zc, lat, dep, axis, N, half_lat=800, half_dep=250):
    """z0 z1 y0 y1 x0 x1 for bigsheet_v2 (N = level-0 (Z, Y, X))."""
    if axis == 'y':
        return [zc - 650, zc + 650, max(0, dep - half_dep), min(N[1], dep + half_dep), max(0, lat - half_lat), min(N[2], lat + half_lat)]
    return [zc - 650, zc + 650, max(0, lat - half_lat), min(N[1], lat + half_lat), max(0, dep - half_dep), min(N[2], dep + half_dep)]
