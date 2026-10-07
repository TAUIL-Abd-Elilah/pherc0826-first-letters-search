"""Block until a dense_any campaign logs a site worth a look, then print it and exit (for one-shot background waits;
a `tail -F | watch_any | head -1` pipe never exits when no further flag arrives). Polls <dir>/campaign.log every
60 s from the current end of the file; flags = watch_any.py lines on the faces given (default fwd only) plus
failures/errors; also exits when every planned site has a result.
usage: wait_flag.py <campaign_dir> [--faces fwd,rev]"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
d = sys.argv[1]
faces = sys.argv[sys.argv.index('--faces') + 1].split(',') if '--faces' in sys.argv else ['fwd']
log = os.path.join(d, 'campaign.log')
n_sites = len(json.load(open(os.path.join(d, 'sites.json'))))
pos = os.path.getsize(log) if os.path.exists(log) else 0
while True:
    time.sleep(60)
    if not os.path.exists(log):
        continue
    with open(log, encoding='utf-8', errors='ignore') as f:
        f.seek(pos); new = f.read(); pos = f.tell()
    if new.strip():
        out = subprocess.run([sys.executable, os.path.join(HERE, 'watch_any.py'), d], input=new, capture_output=True,
                             text=True, encoding='utf-8').stdout
        hits = [l for l in out.splitlines() if 'sites read' not in l and
                (any(f' {fc} best' in l for fc in faces) or re.search(r'track failed|Traceback|Error|LOW DISK', l))]
        if hits:
            print('\n'.join(hits), flush=True); break
    done = len(re.findall(r' valid | track failed| empty sheet', open(log, encoding='utf-8', errors='ignore').read()))
    if done >= n_sites:
        print(os.path.basename(d), 'CAMPAIGN COMPLETE', done, 'results', flush=True); break
