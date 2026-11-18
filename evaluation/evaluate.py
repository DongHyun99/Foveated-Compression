#!/usr/bin/env python3
"""Benchmark Foveated Compression against the conditions of Table 1.

Every condition starts from the same 252-aligned source image (App. A.2), and
every reported cost is the number of visual tokens the LLM actually consumed.

  native                     the uncompressed image. 100%.
  uniform                    Uniform Foveated Compression: every unit compressed
                             by the Foveated Merger. Exactly 1/9 = 11.11%.
  iso_token_downsample       bicubic resize to the uniform token count. 11.11%.
  random_k1                  a deterministic pseudo-random cell, seeded from the
                             item name. 20.99%.
  fixed_k1                   one cell chosen on TRAINING labels and used for every
                             image (`data.selector_labels.best_fixed_cell`). 20.99%.
  selector                   the Foveated Selector's argmax cell. 20.99%.
  matched_budget_downsample  bicubic resize targeting 17/81; 19.82% in practice
                             because sizes are quantised to 28 px.
  downsample_2x              both sides halved; 24.57% in practice.
  oracle_k1                  all nine K=1 actions run for real, official-metric
                             MAXIMUM recorded. Non-deployable upper bound.

K=1 sequence lengths are ASSERTED, not assumed (`FoveatedEncoding.preserve`),
so a geometry change can never be reported as an accuracy result at a silently
different cost.

MACRO is the unweighted mean over benchmarks with MME's raw 0-2800 score
divided by 28 (Table 1); MMStar is averaged within its l2 categories first.
"""
import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

import config
from data.benchmarks import load_extra_fields
from models import foveated_merger as fm
from models import foveated_selector as fs
from models.foveated_compression import FoveatedEncoding
from models.qwen2_5_vl import generate_answer, generate_mixed
from models.token_injection import TokenInjector
from evaluation.interventions import atomic_json, k1_outcomes
from evaluation.metrics import mme_aggregate, mmstar_aggregate, official_score, row_index, task_of

CONDITIONS = ("native", "uniform", "iso_token_downsample", "random_k1", "fixed_k1",
              "selector", "matched_budget_downsample", "downsample_2x", "oracle_k1")
DEFAULT_CONDITIONS = ("native", "uniform", "iso_token_downsample", "random_k1",
                      "selector", "matched_budget_downsample", "downsample_2x")

# Metadata the benchmark-level aggregation needs, kept on each record.
_META_FIELDS = {"mme": ("question_id", "category"), "mmstar": ("l2_category",)}


def downsample_entry(enc, target_tokens):
    """The aligned image bicubic-resized toward `target_tokens` visual tokens.

    Linear scale is sqrt(target / native) so the AREA ratio -- and therefore
    the token count -- matches, up to 28 px quantisation. -> (entry, actual tokens)
    """
    scale = (target_tokens / enc.n_native) ** 0.5
    unit = enc.ms * enc.patch_size
    w = max(unit, int(round(enc.image.width * scale / unit) * unit))
    h = max(unit, int(round(enc.image.height * scale / unit) * unit))
    entry = {"type": "image", "image": enc.image.resize((w, h), Image.BICUBIC),
             "min_pixels": config.PROCESSOR_MIN_PIXELS, "max_pixels": 10 ** 9}
    return entry, (w // unit) * (h // unit)


def deterministic_cell(name, n_cells=config.N_CELLS):
    """A reproducible pseudo-random cell, seeded by the item name.

    Seeded from the name rather than a global RNG so the random control is
    identical across shards, machines and reruns.
    """
    return int(zlib.crc32(name.encode())) % n_cells


@torch.no_grad()
def evaluate_item(mw, injector, merger, selector, qemb, name, image, question, answer,
                  extra=None, conditions=DEFAULT_CONDITIONS, fixed_cell=None,
                  max_visual_tokens=16384):
    """All requested conditions for one item. -> {condition: {score, tokens, ratio, answer}}."""
    task = task_of(name)
    enc = FoveatedEncoding(mw, injector, merger, image, max_visual_tokens)
    out = {}

    def record(key, pred, tokens):
        out[key] = {"score": float(official_score(name, task, pred, answer, extra)),
                    "tokens": int(tokens), "ratio": float(tokens) / enc.n_native,
                    "answer": pred}

    def mixed(key, lay, tokens):
        record(key, generate_mixed(mw, enc.frame_entry(lay.n_tokens), question, task,
                                   tokens, lay), lay.n_tokens)

    def resized(key, target):
        entry, n = downsample_entry(enc, target)
        record(key, generate_answer(mw, [entry], question, task), n)

    if "native" in conditions:
        record("native", generate_answer(mw, [enc.native_entry], question, task), enc.n_native)
    if "uniform" in conditions:
        mixed("uniform", *enc.uniform())
    if "iso_token_downsample" in conditions:
        resized("iso_token_downsample", enc.n_native * config.TOKEN_BUDGET[0])
    if "random_k1" in conditions:
        mixed("random_k1", *enc.preserve([deterministic_cell(name)]))
    if "fixed_k1" in conditions:
        if fixed_cell is None:
            raise ValueError("fixed_k1 needs a cell chosen on training labels")
        mixed("fixed_k1", *enc.preserve([fixed_cell]))
    if "selector" in conditions:
        if selector is None:
            raise ValueError("the selector condition needs a selector checkpoint")
        q = qemb.tokens(question) if qemb is not None else None
        cell = int(enc.cell_scores(selector, q).argmax())
        mixed("selector", *enc.preserve([cell]))
        out["selector"]["cell"] = cell
    if "matched_budget_downsample" in conditions:
        resized("matched_budget_downsample", enc.n_native * config.TOKEN_BUDGET[1])
    if "downsample_2x" in conditions:
        resized("downsample_2x", enc.n_native / 4)
    if "oracle_k1" in conditions:
        scores, answers = k1_outcomes(mw, enc, name, question, answer, extra)
        best = int(np.argmax(scores))
        out["oracle_k1"] = {"score": scores[best],
                            "tokens": int(enc.n_native * config.TOKEN_BUDGET[1]),
                            "ratio": config.TOKEN_BUDGET[1], "answer": answers[best],
                            "cell": best, "cell_scores": scores}
    return out


def _task_value(task, rows, cond):
    """Benchmark-level score in [0, 1] under the benchmark's own aggregation."""
    if task == "mme":
        raw = mme_aggregate((r["meta"]["category"], r["meta"]["question_id"],
                             r[cond]["score"] > 0) for r in rows)
        return raw / 2800.0, raw
    if task == "mmstar":
        return mmstar_aggregate((r["meta"]["l2_category"], r[cond]["score"]) for r in rows), None
    return float(np.mean([r[cond]["score"] for r in rows])), None


def aggregate(records, conditions):
    """Per-benchmark score, macro average (MME / 28), and mean token ratio per condition."""
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    out = {}
    for cond in conditions:
        per_task, raw, ratios = {}, {}, {}
        for task, rows in by_task.items():
            rows = [r for r in rows if cond in r]
            if not rows:
                continue
            per_task[task], r = _task_value(task, rows, cond)
            if r is not None:
                raw[task] = r
            ratios[task] = float(np.mean([x[cond]["ratio"] for x in rows]))
        if not per_task:
            continue
        out[cond] = {
            "macro": float(np.mean(list(per_task.values()))),
            "token_ratio": float(np.mean(list(ratios.values()))),
            "by_task": per_task,
            "raw": raw,
            "n": sum(1 for r in records if cond in r),
        }
    return out


def report(summary):
    """One table: tokens, per-benchmark scores and macro, like Table 1."""
    tasks = [t for t in config.EVAL_TASKS if any(t in v["by_task"] for v in summary.values())]
    tasks += sorted({t for v in summary.values() for t in v["by_task"]} - set(tasks))
    head = f"{'condition':>26} {'tokens':>8} " + " ".join(f"{t[:10]:>10}" for t in tasks)
    lines = [head + f" {'macro':>8}"]
    for cond, v in summary.items():
        cells = []
        for t in tasks:
            if t not in v["by_task"]:
                cells.append(f"{'-':>10}")
            elif t in v["raw"]:
                cells.append(f"{v['raw'][t]:>10.1f}")
            else:
                cells.append(f"{v['by_task'][t] * 100:>10.2f}")
        lines.append(f"{cond:>26} {v['token_ratio'] * 100:>7.2f}% " + " ".join(cells)
                     + f" {v['macro'] * 100:>8.2f}")
    return "\n".join(lines)


def load_merger_and_selector(merger_path, selector_path, device):
    """Load both checkpoints and verify they belong together.

    A selector is trained against ONE merger: its K=1 labels and its features
    are both specific to what the rest of the image looks like once compressed
    by that exact merger, so pairing it with another one is refused.
    """
    merger = fm.load(merger_path).to(device).eval()
    for p in merger.parameters():
        p.requires_grad_(False)
    if not selector_path:
        return merger, None

    payload = torch.load(selector_path, map_location="cpu", weights_only=False)
    embedded = payload["merger"]["state_dict"]
    own = fm.state(merger)["state_dict"]
    if set(embedded) != set(own) or any(not torch.equal(embedded[k], own[k]) for k in own):
        raise ValueError("the selector checkpoint was trained against a different merger")
    selector = fs.from_state(payload["selector"]).to(device).eval()
    for p in selector.parameters():
        p.requires_grad_(False)
    return merger, selector


def run(mw, items, merger, selector, qemb, conditions, out_json=None, fixed_cell=None,
        max_visual_tokens=16384, checkpoint_every=25):
    """Evaluate a list of (name, image, question, answer). Resumable."""
    injector = TokenInjector(mw.model.model.visual, capture_patch=True)
    extras = load_extra_fields({task_of(n) for n, *_ in items})
    records, done = [], set()
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            records = json.load(f)
        done = {r["name"] for r in records}

    for i, (name, image, question, answer) in enumerate(items):
        if name in done or not str(answer).strip():
            continue
        task = task_of(name)
        extra = extras.get(task, {}).get(row_index(name, task))
        row = evaluate_item(mw, injector, merger, selector, qemb, name, image, question,
                            answer, extra=extra, conditions=conditions,
                            fixed_cell=fixed_cell, max_visual_tokens=max_visual_tokens)
        row["name"] = name
        row["task"] = task
        if task in _META_FIELDS:
            row["meta"] = {k: extra[k] for k in _META_FIELDS[task]}
        records.append(row)
        if out_json and (i + 1) % checkpoint_every == 0:
            atomic_json(out_json, records)
    if out_json:
        atomic_json(out_json, records)
    return records
