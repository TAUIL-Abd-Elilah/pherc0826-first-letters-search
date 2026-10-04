"""stdin = dense_any campaign.log lines (any scroll). Print: sites worth a look (strong >= 0.5, |one-sided| >= 0.3,
slab <= 0.6 on the strong face), failures/errors, and a progress line every 20 sites. argv[1] = campaign dir."""
import ast, os, re, sys
OUT = sys.argv[1]
sys.argv = ['x', OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_review import slab_fraction  # noqa: E402
n = 0
for line in sys.stdin:
    s = line.strip()
    if re.search(r'track failed|empty sheet|Traceback|Error|LOW DISK', s):
        print(os.path.basename(OUT), s[:200], flush=True); continue
    m = re.match(r'(\S+ \S+ )?(z\d+_y\d+_x\d+) ([xy]) valid \S+ (\{.*\}) one-sided (\S+)', s)
    if not m:
        continue
    n += 1
    if n % 20 == 0:
        print(os.path.basename(OUT), 'sites read this run:', n, flush=True)
    name, e, one = m.group(2), ast.literal_eval(m.group(4)), m.group(5)
    if not e or one == 'None':
        continue
    face = 'fwd' if e['ens_fwd'][0] >= e['ens_rev'][0] else 'rev'
    if e[f'ens_{face}'][0] < 0.5 or abs(float(one)) < 0.3:
        continue
    d = os.path.join(OUT, name)
    slab = slab_fraction(f'{d}/ens_{face}.png', f'{d}/ct.png')
    if slab <= 0.6:
        print(os.path.basename(OUT), name, m.group(3), face, 'best', e[f'ens_{face}'][0], 'band', e[f'ens_{face}'][1], 'one-sided', one, 'slab', slab, flush=True)
