"""d9v3 vs everything on the realigned held-out labels. Single models from results_eval_v3.json, plus
ensembles (mean of the kept prediction TIFFs, scored with eval9v2.score). Human-label AUC per test segment,
mean over all 8 and over PHerc0841 (the unseen scroll). -> results_v3_table.json
usage: compare_v3.py"""
import json, os
import numpy as np
import tifffile
from eval9v2 import score, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = [l.strip().replace('/', '__') for l in open(os.path.join(HERE, 'test.txt')) if l.strip()]


def load(seg, name):
    a = tifffile.imread(os.path.join(OUT, seg, f'pred_{name}.tif')).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def main():
    res = json.load(open(os.path.join(HERE, 'results_eval_v3.json')))
    models = sorted({k.split('|')[0] for k in res})
    table = {m: {s: res.get(f'{m}|{s}', {}).get('human_auc') for s in TEST} for m in models}
    ens = {'rv2+d9v2': ('reader_v2', 'd9v2_12k'), 'rv2+d9v3': ('reader_v2', 'd9v3_12k'),
           'rv2+sch+d9v3': ('reader_v2', 'scheirer_ft', 'd9v3_12k'), 'd9v3_8k+12k': ('d9v3_8k', 'd9v3_12k')}
    for name, parts in ens.items():
        table[name] = {}
        for s in TEST:
            try:
                ps = [load(s, p) for p in parts]
                H = min(p.shape[0] for p in ps); W = min(p.shape[1] for p in ps)
                table[name][s] = score(s, np.mean([p[:H, :W] for p in ps], 0)).get('human_auc')
            except FileNotFoundError:
                table[name][s] = None
    short = {s: s.split('__')[0][5:] + ':' + s.split('__')[1][:10] for s in TEST}
    print(f"{'model':16s} " + ' '.join(f'{short[s]:>16s}' for s in TEST) + '   all-8   0841')
    rows = {}
    for m, v in table.items():
        vals = [v[s] for s in TEST]
        a8 = np.mean([x for x in vals if x is not None]) if all(x is not None for x in vals) else None
        a41 = np.mean([v[s] for s in TEST if s.startswith('PHerc0841') and v[s] is not None])
        rows[m] = {'per_segment': v, 'mean_all8': a8, 'mean_0841': a41}
        print(f'{m:16s} ' + ' '.join(f'{x:16.3f}' if x is not None else f"{'-':>16s}" for x in vals)
              + (f'   {a8:.3f}' if a8 is not None else '      - ') + f'   {a41:.3f}')
    json.dump(rows, open(os.path.join(HERE, 'results_v3_table.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
