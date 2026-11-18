#!/usr/bin/env python3
"""Stage 2: the Foveated Selector on intervention-based supervision (Sec. 3.3).

Trained only after the merger is frozen (App. A.1). The merger and the whole
vision-language backbone stay frozen; the only trainable module is the
selector, and the only objective is the tie-aware set likelihood of Eq. 4 over
the winner sets measured by `label` (`data.selector_labels`).

No language-model forward pass is needed: the selector reads native tokens,
compressed tokens and question embeddings, all of which come from one frozen
vision-encoder pass and an embedding lookup. One encoder pass is shared by all
questions of an image.

Train and validation partitions are formed at the image-group level, and the
validation figure is winner membership: the fraction of held-out examples whose
argmax cell is in the winner set W (App. B).
"""
import os
import random

import torch

import config
from data.benchmarks import group_items
from data.selector_labels import split_image_groups
from models import foveated_merger as fm
from models import foveated_selector as fs
from models.foveated_compression import FoveatedEncoding
from models.question_embedding import QuestionEmbedder
from models.token_injection import TokenInjector
from training.losses import tie_aware_selector_loss
from training.optim import adamw_with_warmup


def save_checkpoint(path, merger, selector, epoch, val_membership, cfg):
    """The selector, together with the exact merger its labels were measured on."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"merger": fm.state(merger), "selector": fs.state(selector),
                "epoch": int(epoch), "val_winner_membership": float(val_membership),
                "train_config": dict(cfg)}, tmp)
    os.replace(tmp, path)
    return path


def _by_group(items, indices):
    """Regroup `indices` by shared image, preserving item order inside a group."""
    out, cur, last = [], [], None
    for i in indices:
        im = items[i][1]
        if last is not None and im is not last:
            out.append(cur)
            cur = []
        cur.append(i)
        last = im
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def winner_membership(mw, merger, selector, injector, qemb, items, groups, labels):
    """Fraction of items whose argmax cell is a measured winner."""
    selector.eval()
    hits, n = 0, 0
    for g in groups:
        enc = FoveatedEncoding(mw, injector, merger, items[g[0]][1])
        for i in g:
            name, _im, question, _a = items[i]
            q = qemb.tokens(question) if qemb is not None else None
            hits += int(int(enc.cell_scores(selector, q).argmax()) in labels[name])
            n += 1
    selector.train()
    return hits / max(1, n)


def train_selector(mw, merger, items, labels, out_path, cfg=config.SELECTOR_TRAIN,
                   use_question=True, seed=config.SEED, log_every=100):
    """Train the Foveated Selector with Table A1's selector settings."""
    dev = mw.model.device
    merger = merger.to(dev).eval()
    for p in merger.parameters():
        p.requires_grad_(False)
    selector = fs.FoveatedSelector(use_question=use_question).to(dev).train()
    injector = TokenInjector(mw.model.model.visual, capture_patch=True)
    qemb = QuestionEmbedder(mw.model, mw.tokenizer) if use_question else None
    opt, sched = adamw_with_warmup(selector.parameters(), cfg["lr"], cfg["weight_decay"],
                                   cfg["warmup_steps"])

    items = [it for it in items if it[0] in labels]
    names = [it[0] for it in items]
    train_idx, val_idx = split_image_groups(group_items(items), names, cfg["val_frac"], seed)
    train_groups, val_groups = _by_group(items, train_idx), _by_group(items, val_idx)
    print(f"selector: {sum(p.numel() for p in selector.parameters()) / 1e6:.2f}M params, "
          f"{len(train_idx)} train / {len(val_idx)} val items", flush=True)

    rng = random.Random(seed)
    step, losses, membership = 0, [], float("nan")
    for epoch in range(cfg["epochs"]):
        rng.shuffle(train_groups)
        for g in train_groups:
            enc = FoveatedEncoding(mw, injector, merger, items[g[0]][1])
            for i in g:
                name, _im, question, _a = items[i]
                q = qemb.tokens(question) if qemb is not None else None
                loss = tie_aware_selector_loss(enc.cell_scores(selector, q), labels[name])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
                step += 1
                if step % log_every == 0:
                    print(f"epoch {epoch + 1} step {step:>6}  "
                          f"loss {sum(losses[-log_every:]) / log_every:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if val_groups:
            membership = winner_membership(mw, merger, selector, injector, qemb, items,
                                           val_groups, labels)
            print(f"epoch {epoch + 1}: val winner membership {membership * 100:.2f}%",
                  flush=True)
        save_checkpoint(out_path, merger, selector, epoch + 1, membership, cfg)
    print(f"saved {out_path} after {cfg['epochs']} epochs")
    return selector
