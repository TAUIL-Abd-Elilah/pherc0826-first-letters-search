"""stdin = campaign_v2.log lines; print only sites worth a look: strong (>= 0.5), one-sided (>= 0.3), not one slab
(slab <= 0.6 on the strong face), plus failures/errors. Helper for a live log monitor."""
import ast, os, re, sys
sys.argv = ['x', os.environ.get('CAMPAIGN_DIR', 'campaign_0826')]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_review import slab_fraction  # noqa: E402
for line in sys.stdin:
    if re.search(r'failed|empty|Traceback|Error', line):
        print(line.strip(), flush=True); continue
    m = re.match(r'(\S+ )?(z\d+_x\d_d\d) valid \S+ (\{.*\}) one-sided (\S+)', line.strip())
    if not m:
        continue
    name, e, one = m.group(2), ast.literal_eval(m.group(3)), m.group(4)
    if not e or one == 'None':
        continue
    face = 'fwd' if e['ens_fwd'][0] >= e['ens_rev'][0] else 'rev'
    if e[f'ens_{face}'][0] < 0.5 or abs(float(one)) < 0.3:
        continue
    d = os.path.join(os.environ.get('CAMPAIGN_DIR', 'campaign_0826'), name)
    slab = slab_fraction(f'{d}/ens_{face}.png', f'{d}/ct.png')
    if slab <= 0.6:
        print(f'{name} {face} best {e[f"ens_{face}"][0]} band {e[f"ens_{face}"][1]} one-sided {one} slab {slab}', flush=True)
