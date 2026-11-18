#!/usr/bin/env python3
"""Exhaustive, budget-matched K=1 interventions (Sec. 3.3, 4.2).

Every one of the nine cells is preserved in turn, the other eight stay
compressed, and the frozen model answers under the shared generation protocol.
All nine actions cost exactly 17/81 = 20.99% of native's visual tokens, so they
differ only in WHERE native fidelity is spent.

The same measurement serves two purposes:

  label    on TRAIN splits, the per-cell official scores r_c whose argmax set
           is the selector's winner set W (`data.selector_labels`).
  oracle   on EVAL splits, the test-time maximum over the nine actions -- the
           non-deployable K=1 Region-Choice Oracle of Table 1.
"""
import json
import os

import config
from models.qwen2_5_vl import generate_mixed
from evaluation.metrics import official_score, task_of


def k1_outcomes(mw, enc, name, question, answer, extra=None):
    """-> (official score r_c per cell [9], answer per cell [9])."""
    task = task_of(name)
    scores, answers = [], []
    for cell in range(config.N_CELLS):
        lay, tokens = enc.preserve([cell])
        pred = generate_mixed(mw, enc.frame_entry(lay.n_tokens), question, task, tokens, lay)
        answers.append(pred)
        scores.append(float(official_score(name, task, pred, answer, extra)))
    return scores, answers


def label_cells(mw, injector, merger, items, out_json, checkpoint_every=25,
                max_visual_tokens=16384):
    """Measure K=1 outcomes for every item and append them to `out_json`. Resumable."""
    from models.foveated_compression import FoveatedEncoding

    records, done = [], set()
    if os.path.exists(out_json):
        with open(out_json) as f:
            records = json.load(f)
        done = {r["name"] for r in records}

    for i, (name, image, question, answer) in enumerate(items):
        if name in done or not str(answer).strip():
            continue
        enc = FoveatedEncoding(mw, injector, merger, image, max_visual_tokens)
        scores, answers = k1_outcomes(mw, enc, name, question, answer)
        records.append({"name": name, "task": task_of(name),
                        "cell_scores": scores, "cell_answers": answers})
        if (i + 1) % checkpoint_every == 0:
            atomic_json(out_json, records)
    atomic_json(out_json, records)
    return records


def atomic_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
