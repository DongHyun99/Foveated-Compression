#!/usr/bin/env python3
"""Measured region utility for crop foveation (not part of the paper).

The cache-based crop selector (`models.crop_selector`) is calibrated against
one measured quantity:

    gain(R) = log p(answer | base view + R) - log p(answer | base view)

It is the model's own answer likelihood, so it needs no hand-designed notion
of saliency and it cannot disagree with the model about what the model finds
useful. The alternatives all can: query attention, Laplacian energy, object
detectors and CLIP similarity each encode somebody's theory of relevance, and
this repository's own measurements found that attention-derived region scores
rank regions WORSE than random on several tasks.

Two derived labels come from the same measurement.

  correct[j]   whether base + candidate j actually produces the right answer
               under the benchmark's official metric. Binary, and what the
               deployed policy is judged on.

  gain[j]      the continuous version above. Ranks candidates even when
               several of them flip the answer identically, which is why the
               selector uses it as an auxiliary target.

A NOTE ON COST. Labelling one image at a 3x3 grid is ten forward passes: one
baseline and nine candidates. The greedy multi-region search is quadratic on
top of that. This is why the selector is trained from a feature cache rather
than online, and why the label set is built once and frozen.

THE PROMPT IS PART OF THE MEASUREMENT. Every call routes through
`config.build_question(task, q)`, so a gain is measured under exactly the
prompt the policy is deployed with.
"""
import torch

import config
from models.qwen2_5_vl import answer_logprob, generate_answer


def base_entries(image, scale=None):
    """The downscaled global view every foveated input starts from."""
    s = config.BASE_SCALE if scale is None else scale
    w = max(1, int(image.width * s))
    h = max(1, int(image.height * s))
    return [config.image_entry(image.resize((w, h)), "base", scale=s)]


def candidate_entries(image, boxes):
    """One high-resolution crop entry per candidate pixel box."""
    return [config.image_entry(image.crop(b), "crop") for b in boxes]


@torch.inference_mode()
def measure_region_gains(mw, image, boxes, question, answer, task, base_scale=None):
    """Per-candidate gain and correctness against the base view.

    -> {"baseline_lp", "baseline_answer", "gains" [n], "answers" [n]}

    One baseline pass plus one pass per candidate for the likelihood, and the
    same again for the generations the official metric needs. Nothing here is
    approximated: the whole point of this function is to be the ground truth
    everything cheaper is validated against.
    """
    base = base_entries(image, base_scale)
    baseline_lp = answer_logprob(mw, base, question, answer, task)
    baseline_answer = generate_answer(mw, base, question, task)

    gains, answers = [], []
    for entry in candidate_entries(image, boxes):
        entries = base + [entry]
        lp = answer_logprob(mw, entries, question, answer, task)
        gains.append(lp - baseline_lp)
        answers.append(generate_answer(mw, entries, question, task))
    return {"baseline_lp": baseline_lp, "baseline_answer": baseline_answer,
            "gains": gains, "answers": answers}


@torch.inference_mode()
def greedy_multi_region(mw, image, boxes, question, answer, task, k_max=2,
                        base_scale=None):
    """Greedy forward selection of up to k_max regions by measured gain.

    -> (round-1 gains over every candidate, [{"idx", "gain"} per rank])

    The round-1 gains are returned separately because they are the single-cell
    labels, and re-deriving them from the greedy trace would quietly change
    what "the gain of cell j" means after the first pick.

    This is what validates the cheap approximation the K-selector relies on --
    that the reward of showing several cells together is close to the MAX of
    their individual rewards. Greedy search measures the real thing.
    """
    base = base_entries(image, base_scale)
    baseline_lp = answer_logprob(mw, base, question, answer, task)
    entries = candidate_entries(image, boxes)

    n = len(entries)
    chosen, chosen_entries, ranks = [], [], []
    round1 = None
    for rank in range(k_max):
        gains = [None] * n
        for j in range(n):
            if j in chosen:
                continue
            lp = answer_logprob(mw, base + chosen_entries + [entries[j]],
                                        question, answer, task)
            gains[j] = lp - baseline_lp
        if rank == 0:
            round1 = list(gains)
        valid = [(j, g) for j, g in enumerate(gains) if g is not None]
        if not valid:
            break
        best_idx, best_gain = max(valid, key=lambda x: x[1])
        chosen.append(best_idx)
        chosen_entries.append(entries[best_idx])
        ranks.append({"idx": best_idx, "gain": best_gain})
        torch.cuda.empty_cache()
    return round1, ranks
