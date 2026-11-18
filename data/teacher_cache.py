#!/usr/bin/env python3
"""The native model's cached answers: the Foveated Merger's only supervision.

Behavioural self-distillation (Sec. 3.2) needs, per training example, the
frozen native model's greedy answer y_{1:L} and its next-token distribution at
every answer position. `dump` produces both once, as npz shards that
`TeacherCache` reads:

    names    [n]          item names "{task}_{row}"
    tasks    [n]          task of each item
    answers  [n]          decoded teacher answers
    lens     [n]          answer length L per item
    n_visual [n]          native visual tokens of the cached view
    ids      [sum L]      teacher token ids, concatenated
    topk_ids [sum L, K]   top-K token ids per position, sorted descending
    topk_lp  [sum L, K]   their log-probs, fp16 (K = 256, App. A.1)
    mass     [sum L]      retained top-K probability mass per position

WHY CACHE. p_T depends on (image, question, answer) only; the merger never
enters it. Caching removes a full teacher forward from every training step at
no cost in accuracy, and greedy decoding already computes p_T at every answer
position, so the dump is a single generate() per item.

TOP-K IS ENOUGH, AND IT IS MEASURED. KL's terms are weighted by the teacher's
probability. The retained mass is stored per position and `report` prints it;
the measured top-256 mass is 0.99936, so the discarded tail is worth 6.4e-04
nats against a 0.148-nat training signal. Run `dump-teacher --report` before
training on a new dump.

THE CACHED VIEW IS THE STUDENT'S VIEW. `dump` resizes with the same
`utils.geometry.training_view`, factor and per-task visual-token cap as
`training.train_merger.merger_step`, and prompts through the same
`models.qwen2_5_vl` input path as `forced_answer_logits`.

THE STUDENT IS FORCED ON THE TEACHER'S OWN TOKEN IDS, not on a re-tokenisation
of the answer string. Byte-pair merging occasionally absorbs an answer into the
prompt's final token; forcing on the cached ids guarantees the cached
distribution and the scored span cannot drift apart.
"""
import glob
import os
import sys
import time

import numpy as np
import torch


def shard_paths(pattern):
    """Sorted complete shards. `*.npz` also matches an in-flight `x.npz.tmp.npz`."""
    return [p for p in sorted(glob.glob(pattern)) if not p.endswith(".tmp.npz")]


class TeacherCache:
    """Cached teacher answers keyed by item name."""

    def __init__(self, pattern):
        self.ids, self.topk_ids, self.topk_lp, self.answers = {}, {}, {}, {}
        for p in shard_paths(pattern):
            z = np.load(p, allow_pickle=True)
            off = np.concatenate([[0], np.cumsum(z["lens"])])
            ids, ti, tl = z["ids"], z["topk_ids"], z["topk_lp"]
            for i, nm in enumerate(z["names"]):
                a, b = int(off[i]), int(off[i + 1])
                if b <= a:
                    continue
                nm = str(nm)
                self.ids[nm] = ids[a:b]
                self.topk_ids[nm] = ti[a:b]
                self.topk_lp[nm] = tl[a:b]
                self.answers[nm] = str(z["answers"][i])

    def __contains__(self, name):
        return name in self.ids

    def __len__(self):
        return len(self.ids)

    def get(self, name, dev):
        """-> (answer ids [L], top-k ids [L, K], top-k log-probs [L, K])."""
        return (torch.as_tensor(self.ids[name], device=dev, dtype=torch.long),
                torch.as_tensor(self.topk_ids[name].astype(np.int64), device=dev),
                torch.as_tensor(self.topk_lp[name].astype(np.float32), device=dev))


# --------------------------------------------------------------------------
# producer
# --------------------------------------------------------------------------
def save(path, rec):
    """Write a shard. Never in place: an interrupted write would otherwise
    leave a dump that loads fine and is silently short."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"

    def cat(key, shape, dtype):
        return np.concatenate([r[key] for r in rec]) if rec else np.zeros(shape, dtype)

    np.savez(tmp,
             names=np.array([r["name"] for r in rec], object),
             tasks=np.array([r["task"] for r in rec], object),
             answers=np.array([r["answer"] for r in rec], object),
             lens=np.array([len(r["ids"]) for r in rec], np.int32),
             n_visual=np.array([r["n_visual"] for r in rec], np.int32),
             ids=cat("ids", 0, np.int32),
             topk_ids=cat("topk_ids", (0, 0), np.int32),
             topk_lp=cat("topk_lp", (0, 0), np.float16),
             mass=cat("mass", 0, np.float32))
    os.replace(tmp, path)


def load_records(path):
    """Read a shard back into the record form `save` writes."""
    z = np.load(path, allow_pickle=True)
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    out = []
    for i, nm in enumerate(z["names"]):
        a, b = int(off[i]), int(off[i + 1])
        out.append({"name": str(nm), "task": str(z["tasks"][i]),
                    "answer": str(z["answers"][i]), "n_visual": int(z["n_visual"][i]),
                    "ids": z["ids"][a:b], "topk_ids": z["topk_ids"][a:b],
                    "topk_lp": z["topk_lp"][a:b], "mass": z["mass"][a:b]})
    return out


def report(pattern):
    """Coverage and top-k truncation of an existing dump."""
    paths = shard_paths(pattern)
    if not paths:
        raise FileNotFoundError(f"no teacher shards match {pattern!r}")
    zs = [np.load(p, allow_pickle=True) for p in paths]
    n = sum(len(z["names"]) for z in zs)
    if n == 0:
        raise ValueError(f"{len(paths)} shard(s) match {pattern!r} but hold no items")
    lens = np.concatenate([z["lens"] for z in zs])
    n_visual = np.concatenate([z["n_visual"] for z in zs])
    mass = np.concatenate([z["mass"] for z in zs]).astype(np.float64)
    tasks = {}
    for z in zs:
        for t in z["tasks"]:
            tasks[str(t)] = tasks.get(str(t), 0) + 1
    lost = 1.0 - mass
    print(f"{len(paths)} shards, {n:,} items, {lens.sum():,} answer tokens "
          f"(mean {lens.mean():.2f})")
    print(f"visual tokens per item: mean {n_visual.mean():.0f}, max {n_visual.max()}")
    print("per task:", {k: v for k, v in sorted(tasks.items())})
    print("retained top-k mass: " + "  ".join(
        f"p{q} {np.percentile(mass, q):.6f}" for q in (0, 1, 5, 25, 50))
        + f"  mean {mass.mean():.6f}")
    print(f"discarded mass: mean {lost.mean():.2e}, max {lost.max():.2e} "
          f"(an upper bound on the per-position KL truncation error, in nats)")
    return {"items": n, "mass_mean": float(mass.mean()), "mass_min": float(mass.min())}


def dump(mw, items, out, topk, max_visual_tokens, merge, resume_from="",
         checkpoint_every=500, limit=-1):
    """Cache the teacher for `items` into `out`, skipping items in `resume_from`.

    `resume_from` must be a different path from `out`. `save` rewrites `out`
    in full from the records held in memory, so keeping the earlier shard at
    its own path means no failure of this run can lose it.

    `merge` and `max_visual_tokens` must be the values `train-merger` uses
    (`config.MERGER["merge"]`, `config.MERGER_TRAIN["max_visual_tokens"]`);
    any other value caches the teacher on a differently sized image.
    """
    import config
    from evaluation.metrics import task_of
    from models.qwen2_5_vl import teacher_topk
    from utils.geometry import training_view

    if resume_from and os.path.abspath(resume_from) == os.path.abspath(out):
        raise ValueError("--resume-from must differ from --out")
    visual = mw.model.model.visual
    factor = merge * visual.spatial_merge_size * visual.patch_size

    rec = load_records(resume_from) if resume_from else []
    done = {r["name"] for r in rec}
    todo = [it for it in items if it[0] not in done]
    if limit > 0:
        todo = todo[:limit]
    print(f"teacher dump: {len(todo):,} to do, {len(done):,} resumed -> {out}", flush=True)

    t0, n_skip = time.time(), 0
    for j, (name, image, question, _answer) in enumerate(todo):
        task = task_of(name)
        try:
            view = training_view(image, task, factor, max_visual_tokens)
            r = teacher_topk(mw, config.image_entry(view, "native"), question, task, topk)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = None
            print(f"  [skip] {name}: out of memory", file=sys.stderr, flush=True)
        if r is None:
            n_skip += 1
        else:
            ids, ti, tl, mass, n_visual = r
            rec.append({"name": name, "task": task, "n_visual": n_visual,
                        "answer": mw.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        "ids": ids, "topk_ids": ti, "topk_lp": tl, "mass": mass})
        if (j + 1) % checkpoint_every == 0:
            save(out, rec)
            el = time.time() - t0
            print(f"  {j + 1:,}/{len(todo):,}  skipped {n_skip}  "
                  f"{el / (j + 1):.2f} s/item  eta {(len(todo) - j - 1) * el / (j + 1) / 3600:.2f} h",
                  flush=True)
    save(out, rec)
    print(f"saved {len(rec):,} items -> {out} ({n_skip} skipped, "
          f"{(time.time() - t0) / 3600:.2f} h)", flush=True)
    return rec
