#!/usr/bin/env python3
"""Deployment: one image, one question, one mixed-resolution sequence, one generation.

THE ACTION SPACE IS THE FIXED 3x3 COARSE-CELL GRID, NOT INDIVIDUAL UNITS. The
Foveated Selector emits one logit per compressed unit because that is the
resolution its features have, but the deployed action is always a WHOLE cell:
the unit logits are mean-pooled into nine cell scores, ranked, and the top-K
cells are preserved at native resolution (K = 1 in the paper).

NO LLM PREFILL FOR THE SELECTION ITSELF. Everything selector-side runs off one
native vision-encoder pass -- the pass the model needs to see the image at all
-- plus the question's input embeddings, an embedding-table lookup.

THE TOKEN ACCOUNTING CONTRACT. The only accepted cost is the number of
image_pad positions `model.generate` actually consumed. `MixedFrame` raises at
prefill if the processor's placeholder count and the layout's length disagree,
so `n_vis_actual == layout.n_tokens` is not reported on trust -- anything else
would already have raised. The sidecar log records that number verbatim,
paired with the native count and the cell action for the same image.
"""
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

import config
from models import foveated_merger as fm
from models import foveated_selector as fs
from models import mixed_resolution as mr
from models.foveated_compression import FoveatedEncoding
from models.question_embedding import QuestionEmbedder
from models.token_injection import TokenInjector


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# pure decision logic -- no torch, no model state, directly unit-testable
# --------------------------------------------------------------------------
def validate_single_image_request(visuals):
    """Batch-1, exactly one image, no video, no text-only. Fails explicitly."""
    if visuals is None or len(visuals) != 1 or not isinstance(visuals[0], Image.Image):
        raise ValueError("Foveated Compression requires exactly one image per request "
                         f"(batch_size=1, no video/multi-image/text-only); got {visuals!r}")
    return visuals[0]


def solve_cell_count(requested_ratio, n_native, n_units, ranked_sizes, expansion=8):
    """The number of top-ranked cells whose ACTUAL token count is closest to
    `requested_ratio` of native.

        total(K) = n_units + expansion * sum(ranked_sizes[:K])

    which is the packed count when the preserved units are exactly the units of
    the top-K cells: a compressed unit costs one token, a preserved one costs
    merge**2 = expansion + 1. Ties break toward the SMALLER K.
    """
    if n_native <= 0:
        raise ValueError(f"n_native must be > 0, got {n_native}")
    best_k, best_diff, cum = 0, None, 0
    for k in range(len(ranked_sizes) + 1):
        if k > 0:
            cum += int(ranked_sizes[k - 1])
        diff = abs((n_units + expansion * cum) / n_native - float(requested_ratio))
        if best_diff is None or diff < best_diff:
            best_diff, best_k = diff, k
    return best_k


def resolve_budget(token_ratio, cell_k):
    """Exactly one of (token_ratio, cell_k) chooses how many cells to preserve."""
    if (token_ratio is None) == (cell_k is None):
        raise ValueError(f"exactly one of token_ratio/cell_k must be given, got "
                         f"token_ratio={token_ratio!r} cell_k={cell_k!r}")
    if cell_k is not None and not 0 <= int(cell_k) <= config.N_CELLS:
        raise ValueError(f"cell_k must be in [0, {config.N_CELLS}], got {cell_k!r}")
    if token_ratio is not None and not 0.0 < float(token_ratio) <= 1.0:
        raise ValueError(f"token_ratio must be in (0, 1], got {token_ratio!r}")
    return (float(token_ratio) if token_ratio is not None else None,
            int(cell_k) if cell_k is not None else None)


# --------------------------------------------------------------------------
# checkpoint + one resolved request
# --------------------------------------------------------------------------
class FoveatedCheckpoint:
    """A selector checkpoint (merger embedded), frozen and in eval mode.

    The file's SHA-256 is recorded on load and written into every token-log
    record, so a reported number can always be traced to the exact weights.
    """

    def __init__(self, path, device):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("merger", "selector"):
            if key not in payload:
                raise ValueError(f"{path} is not a selector checkpoint: missing {key!r}")
        self.merger = fm.from_state(payload["merger"]).to(device).eval()
        self.selector = fs.from_state(payload["selector"]).to(device).eval()
        for p in list(self.merger.parameters()) + list(self.selector.parameters()):
            p.requires_grad_(False)
        self.path = path
        self.sha256 = sha256_file(path)


class FoveatedResult:
    __slots__ = ("lay", "tokens", "n_vis_actual", "n_vis_native", "ratio",
                 "requested_ratio", "cell_k", "selected_cells", "cell_scores")

    def __init__(self, lay, tokens, n_vis_actual, n_vis_native, ratio,
                 requested_ratio, cell_k, selected_cells, cell_scores):
        self.lay = lay
        self.tokens = tokens
        self.n_vis_actual = n_vis_actual
        self.n_vis_native = n_vis_native
        self.ratio = ratio
        self.requested_ratio = requested_ratio
        self.cell_k = cell_k
        self.selected_cells = selected_cells
        self.cell_scores = cell_scores


class FoveatedPipeline:
    """Frozen backbone + Foveated Merger + Foveated Selector, hooked exactly once.

    PyTorch forward hooks accumulate, so the injector is installed here and
    never per request -- re-hooking per image silently multiplies the per-pass
    cost until the run stalls.
    """

    def __init__(self, mw, ckpt_path, device=None):
        self.mw = mw
        self.ckpt = FoveatedCheckpoint(ckpt_path, device or mw.model.device)
        self.injector = TokenInjector(mw.model.model.visual, capture_patch=True)
        self.qemb = (QuestionEmbedder(mw.model, mw.tokenizer)
                     if self.ckpt.selector.use_question else None)

    @torch.inference_mode()
    def layout(self, image, question_text, token_ratio=None, cell_k=None):
        """One native vision-encoder pass -> cell ranking -> mixed-resolution tokens.

        With neither budget given, the paper's K = 1 is used.
        """
        if token_ratio is None and cell_k is None:
            cell_k = config.K_PRESERVED
        token_ratio, cell_k = resolve_budget(token_ratio, cell_k)
        enc = FoveatedEncoding(self.mw, self.injector, self.ckpt.merger, image)
        q = self.qemb.tokens(question_text) if self.qemb is not None else None
        scores = enc.cell_scores(self.ckpt.selector, q).float().cpu().numpy()
        order = np.argsort(-scores, kind="stable")

        if cell_k is None:
            sizes = np.bincount(enc.cell_of_unit, minlength=config.N_CELLS)
            cell_k = solve_cell_count(token_ratio, enc.n_native, len(enc.cell_of_unit),
                                      sizes[order], expansion=enc.merge ** 2 - 1)
        selected = sorted(int(c) for c in order[:cell_k])
        lay, tokens = enc.preserve(selected)
        return FoveatedResult(lay=lay, tokens=tokens.to(dtype=self.mw.model.dtype),
                              n_vis_actual=lay.n_tokens, n_vis_native=enc.n_native,
                              ratio=lay.n_tokens / enc.n_native, requested_ratio=token_ratio,
                              cell_k=cell_k, selected_cells=selected,
                              cell_scores=scores.tolist())

    @torch.inference_mode()
    def generate(self, image, question_text, task, token_ratio=None, cell_k=None):
        """Selector -> mixed-resolution sequence -> greedy answer. -> (text, FoveatedResult)."""
        from models.qwen2_5_vl import generate_mixed

        result = self.layout(image, question_text, token_ratio, cell_k)
        visual = self.mw.model.model.visual
        fw, fh = mr.frame_size(result.lay.n_tokens, visual.spatial_merge_size,
                               visual.patch_size)
        frame = {"type": "image", "image": Image.new("RGB", (fw, fh)),
                 "min_pixels": config.PROCESSOR_MIN_PIXELS, "max_pixels": 10 ** 9}
        text = generate_mixed(self.mw, frame, question_text, task, result.tokens, result.lay)
        return text, result


# --------------------------------------------------------------------------
# token sidecar
# --------------------------------------------------------------------------
def token_log_record(task, split, doc_id, result, checkpoint_sha256):
    return {
        "task": task, "split": split, "doc_id": doc_id,
        "n_vis_actual": int(result.n_vis_actual),
        "n_vis_native": int(result.n_vis_native),
        "ratio": float(result.ratio),
        "requested_ratio": (float(result.requested_ratio)
                            if result.requested_ratio is not None else None),
        "checkpoint_sha256": checkpoint_sha256,
        "cell_k": int(result.cell_k),
        "selected_cells": [int(c) for c in result.selected_cells],
    }


def append_token_log(path, task, split, doc_id, result, checkpoint_sha256):
    """Append one JSONL record, in the evaluation's own sequential call order,
    so a later analysis can join it against per-sample outputs by
    (task, split, doc_id)."""
    record = token_log_record(task, split, doc_id, result, checkpoint_sha256)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
