#!/usr/bin/env python3
"""Intervention-based selector supervision (Sec. 3.3, App. B).

For each training example all nine deterministic K=1 actions are executed and
scored with the official metric (`evaluation.interventions`). With r_c the
score obtained by preserving cell c, the winner set is

    W = {c : r_c = max_j r_j}.

All-nine ties are discarded -- they carry no spatial preference -- while
partial ties are retained as sets rather than collapsed to an arbitrary
one-hot target (Eq. 4).

Labels are merger-specific by construction: a cell wins only relative to what
the rest of the image looks like once compressed, so labels measured against a
different merger are void, not merely stale.
"""
import glob
import json
import zlib

import numpy as np


def winner_set(scores):
    """Winner set W of one 9-vector of K=1 scores, or None for an all-nine tie."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0 or float(np.ptp(s)) <= 0:
        return None
    return set(np.flatnonzero(s >= s.max()).tolist())


def load_winner_sets(patterns, single_winner=False):
    """Label files -> {name: winner set W}, all-nine ties dropped.

    `single_winner=True` keeps only the lowest-index winner of every set: the
    "Single-Winner Supervision" ablation of Table 2.
    """
    labels = {}
    for pattern in ([patterns] if isinstance(patterns, str) else patterns):
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for rec in json.load(f):
                    w = winner_set(rec["cell_scores"])
                    if w is None:
                        continue
                    labels[str(rec["name"])] = {min(w)} if single_winner else w
    return labels


def best_fixed_cell(labels, n_cells=9):
    """The single cell with the highest winner membership on TRAINING labels.

    This is the "Fixed K=1" baseline's cell (Table 1): chosen once from
    training data, then applied to every evaluation image.
    """
    counts = np.zeros(n_cells, dtype=np.int64)
    for w in labels.values():
        for c in w:
            counts[c] += 1
    return int(counts.argmax())


def split_image_groups(groups, names, val_frac, seed=0):
    """Image-group-level train/val split (App. A.1).

    `groups` are lists of item indices sharing one image; `names` maps index
    -> item name. Assignment is a deterministic hash of the group's first
    name, so the split is identical across shards and reruns and no image
    lands on both sides.
    """
    train, val = [], []
    threshold = int(val_frac * 2 ** 32)
    for g in groups:
        h = zlib.crc32(f"{seed}:{names[g[0]]}".encode())
        (val if h < threshold else train).extend(g)
    return train, val
