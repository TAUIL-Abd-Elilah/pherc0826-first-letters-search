"""Fine-tune the public ink_9um model on NATIVE ~9 um surface volumes of six scrolls (PHerc0009B, 0139, 0343P,
0500P2, 0814 at 8.64 / 9.362 um), supervised by the organisers' ~2.4 um predictions carried onto the ~9 um grid
(pack9.py + align9.py). Copy of cross-scan-ink-transfer/src/dist9/ft9.py with: 21-layer stacks (the inference
window is layers 2..18), valid mask from the middle layer, and sampling balanced per scroll.

Keeps the released model, its robust-MAD normalization and its z convention: inference centre-
crops 17 of the 28 native layers (start 6); training jitters that window by +-2 like the recipe's
17-of-21 jitter. Loss = BCE on 0.25 + 0.5 * target (the recipe's label smoothing 0.5, so outputs stay
on the released calibration) + 0.5 * soft Dice. In-plane flips / 90-degree rotations.

Writes checkpoints in the released format (weights under 'ema_model'), so the stock inference CLI
evaluates them unchanged.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import zarr

sys.path.insert(0, os.environ.get('INK9UM_SRC', 'villa/vesuvius/src'))
from vesuvius.ink_detection.config import InkConfig  # noqa: E402
from vesuvius.ink_detection.inference.inference_runtime import TargetModel  # noqa: E402
from vesuvius.ink_detection.inference.infer import (flat_preprocessing_from_config,  # noqa: E402
                                                    normalize_flat_patch)
from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights  # noqa: E402
from vesuvius.ink_detection.models.model import make_model  # noqa: E402

DATA = os.environ.get('D9_DATA', 'corpus_9um')


class Pairs:
    def __init__(self, segs, patch=128, depth=17, start=2, jitter=2, min_valid=0.9, label='canon_al.npy'):
        self.items = []
        for s in segs:
            d = os.path.join(DATA, s)
            sv = zarr.open_array(os.path.join(d, 'sv.zarr'), mode='r')
            lab = np.load(os.path.join(d, label)).astype(np.float32)
            H, W = min(sv.shape[1], lab.shape[0]), min(sv.shape[2], lab.shape[1])
            valid = np.isfinite(lab[:H, :W]) & (np.asarray(sv[sv.shape[0] // 2, :H, :W]) > 0)
            # candidate patch corners on a 32 px lattice with >= min_valid coverage
            ys, xs = np.meshgrid(np.arange(0, H - patch, 32), np.arange(0, W - patch, 32), indexing='ij')
            cs = np.cumsum(np.cumsum(np.pad(valid, ((1, 0), (1, 0))), 0), 1)
            cov = (cs[ys + patch, xs + patch] - cs[ys, xs + patch] - cs[ys + patch, xs] + cs[ys, xs]) / patch ** 2
            ok = cov >= min_valid
            if ok.sum() == 0:
                print(f'{s}: no patch positions, skipped', flush=True)
                continue
            self.items.append({'seg': s, 'sv': sv, 'lab': np.nan_to_num(lab[:H, :W]), 'yx': np.stack([ys[ok], xs[ok]], 1)})
            print(f'{s}: {ok.sum()} patch positions', flush=True)
        self.patch, self.depth, self.start, self.jitter = patch, depth, start, jitter
        scroll = [it['seg'].split('__')[0] for it in self.items]
        n = {k: scroll.count(k) for k in set(scroll)}
        w = np.array([1.0 / n[k] for k in scroll]); self.w = w / w.sum()
        print('per-scroll segments:', n, flush=True)

    def sample(self, rng):
        it = self.items[rng.choice(len(self.items), p=self.w)]
        y, x = it['yx'][rng.integers(len(it['yx']))]
        z0 = self.start + int(rng.integers(-self.jitter, self.jitter + 1))
        img = np.asarray(it['sv'][z0:z0 + self.depth, y:y + self.patch, x:x + self.patch])
        lab = it['lab'][y:y + self.patch, x:x + self.patch]
        k = int(rng.integers(4))
        img, lab = np.rot90(img, k, (1, 2)), np.rot90(lab, k)
        if rng.random() < 0.5:
            img, lab = img[:, :, ::-1], lab[:, ::-1]
        return np.ascontiguousarray(img), np.ascontiguousarray(lab)


def build(ckpt):
    payload = load_checkpoint(ckpt)
    cfg = InkConfig.from_mapping(payload['config'])
    _, state = select_inference_weights(payload, source=ckpt)
    base = make_model(cfg)
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    base.load_state_dict(state, strict=False)
    return payload, cfg, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--init', default=os.path.join(os.environ.get('INK9UM_CKPTS', 'models/ink_9um'), 'hybrid_3d2d-seed42', 'step-075000.pth'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='canon_al.npy')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    payload, cfg, base = build(a.init)
    pre = flat_preprocessing_from_config(cfg.data.normalization)
    dev = torch.device('cuda')
    model = TargetModel(base, input_pad_depth_to=cfg.model.input_pad_depth_to).to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    scaler = torch.amp.GradScaler('cuda')
    data = Pairs(a.train, label=a.label)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    log = open(os.path.join(a.out, 'train.jsonl'), 'a')
    t0 = time.time()
    for step in range(1, a.steps + 1):
        imgs, labs = zip(*(data.sample(rng) for _ in range(a.batch)))
        x = torch.from_numpy(np.stack([normalize_flat_patch(i, pre) for i in imgs]))[:, None].to(dev)
        t = torch.from_numpy(np.stack(labs))[:, None].to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
        logits = F.interpolate(logits.float(), size=t.shape[-2:], mode='bilinear', align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, 0.25 + 0.5 * t)
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
        loss = bce + 0.5 * dice.mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0:
            rec = {'step': step, 'loss': float(loss), 'bce': float(bce), 'dice': float(dice.mean()),
                   'lr': sched.get_last_lr()[0], 's': round(time.time() - t0)}
            log.write(json.dumps(rec) + '\n'); log.flush()
            if step % 250 == 0:
                print(rec, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            out = dict(payload)
            for k in ('state_dict', 'model_state_dict', 'model'):
                out.pop(k, None)
            out['ema_model'] = {k: v.detach().cpu() for k, v in base.state_dict().items()}
            out['finetune'] = {'init': a.init, 'train_segments': a.train, 'step': step,
                               'labels': 'organisers ~2.4um predictions on the native ~9um grid, six scrolls'}
            torch.save(out, os.path.join(a.out, f'ft-{step:06d}.pth'))
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
