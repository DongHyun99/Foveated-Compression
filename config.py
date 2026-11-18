#!/usr/bin/env python3
"""Every setting of the paper, defined once and enforced everywhere.

Foveated Compression: Selective High-Resolution Preservation for Token-Efficient
VLMs (arXiv:2610.07729). Section / table references below point into the paper.

Every stage builds its prompt, sizes its images and loads its model through this
module, so a run's protocol is a property of the repository rather than of
whichever command launched it: a prompt suffix or a pixel bound that differs
between labelling and evaluation silently measures prompt formatting instead of
the method.

  BACKBONE         the frozen Qwen2.5-VL-7B-Instruct and its processor bounds.
  MERGER           Foveated Merger architecture (Sec. 3.2).
  SELECTOR         Foveated Selector architecture (Sec. 3.3).
  MERGER_TRAIN     Table A1, "Merger" column (App. A.1).
  SELECTOR_TRAIN   Table A1, "Selector" column (App. A.1).
  GENERATION       the shared evaluation protocol (Sec. 4.1, App. A.2).
  EVAL_TASKS       the eight reported benchmarks (Sec. 4.1, Table 1).
  TRAIN_TASKS      the six merger training sources (Sec. 4.1).

`image_entry` assigns pixel bounds per VIEW ROLE, scaled by AREA. A view at
linear scale s must cost s**2 of native's visual-token budget, and that only
holds if its min/max pixel bounds are scaled too. Pinning a downscaled view's
floor at 28*28 instead breaks the budget on every image small enough that
*native* was upscaled to the 200704 floor while the downscaled view was not --
measured as 85% of MMStar and 14% of GQA.
"""
import os
import random

BACKBONE = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "min_pixels": 200704,           # 256 * 28 * 28
    "max_pixels": 12845056,
    "system_prompt": "You are a helpful assistant.",
}

SEED = 42

# --------------------------------------------------------------------------
# geometry (App. A.3)
# --------------------------------------------------------------------------
# Image dimensions are multiples of 252 = 14 * 2 * 3 * 3: the 14 px vision
# patch, Qwen's native 2x2 visual merge, the Foveated Merger's 3x3 reduction
# and the 3x3 selection grid. Compression blocks and selection-cell boundaries
# therefore align exactly on every image.
ALIGNMENT_PX = 252

# The 3 x 3 coarse-cell action space, K = 1 preserved cell (Sec. 3.3, 3.4).
SELECTION_GRID = 3
N_CELLS = SELECTION_GRID * SELECTION_GRID
K_PRESERVED = 1

# Exact visual-token budget of preserving K of 9 cells (Eq. 5, Table A2).
TOKEN_BUDGET = {0: 1 / 9, 1: 17 / 81, 2: 25 / 81, 3: 33 / 81}

# --------------------------------------------------------------------------
# modules (Sec. 3.2, 3.3)
# --------------------------------------------------------------------------
MERGER = {
    "dim": 1280,         # C, the vision-encoder patch width
    "merge": 3,          # 3x3 native-token block -> 1 compressed token
    "hidden": 2560,      # W_fc: R^{9C} -> R^2560   (32.77M trainable parameters)
}

SELECTOR = {
    "d_token": 3584,     # LLM-side visual-token width
    "hidden": 512,       # visual-descriptor / question width   (10.8M parameters)
    "heads": 8,          # question cross-attention heads
    "max_question_tokens": 48,
}

# --------------------------------------------------------------------------
# training (Table A1)
# --------------------------------------------------------------------------
MERGER_TRAIN = {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "epochs": 1,
    "warmup_steps": 200,          # linear warmup, then constant
    "grad_accum": 4,
    "grad_clip": 1.0,
    "max_visual_tokens": 1600,    # GQA is capped lower, see TRAIN_TASKS
    "teacher_topk": 256,
}

SELECTOR_TRAIN = {
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "epochs": 4,
    "warmup_steps": 100,          # linear warmup, then constant
    "val_frac": 0.1,              # held-out image groups for winner-membership
}

# --------------------------------------------------------------------------
# evaluation protocol (Sec. 4.1, App. A.2)
# --------------------------------------------------------------------------
# One instruction, appended to every benchmark question, for every condition.
INSTRUCTION = "\nAnswer the question using a single word or phrase."

GENERATION = {
    "max_new_tokens": 16,
    "do_sample": False,
    "num_beams": 1,
}


def _task(path, split, cap=False, **extra):
    d = {"path": path, "split": split, "capitalize_question": cap}
    d.update(extra)
    return d


# EVAL splits. Full official splits -- nothing here is subsampled.
TASKS = {
    "gqa":            _task("lmms-lab/GQA", "testdev"),
    "mmbench_en":     _task("lmms-lab/MMBench", "dev"),
    "mmbench_cn":     _task("lmms-lab/MMBench", "dev"),
    "mme":            _task("lmms-lab/MME", "test"),
    "pope":           _task("lmms-lab/POPE", "test"),
    "mmstar":         _task("Lin-Chen/MMStar", "val"),
    "ocrbench":       _task("echo840/OCRBench", "test"),
    "chartqa":        _task("lmms-lab/ChartQA", "test"),
    # Supported but not part of the paper's reported set.
    "docvqa_val":     _task("lmms-lab/DocVQA", "validation"),
    "textvqa":        _task("lmms-lab/textvqa", "validation", cap=True),
    "scienceqa_img":  _task("lmms-lab/ScienceQA", "test"),
}

# The eight benchmarks of Table 1, in table order.
EVAL_TASKS = ("gqa", "mmbench_en", "mmbench_cn", "mme", "pope", "mmstar",
              "ocrbench", "chartqa")

# TRAIN-split sources of the Foveated Merger (Sec. 4.1) -- never used to report
# a benchmark number. Training on the published train splits makes every eval
# split held out by construction; carving a training set out of an eval split
# leaks (GQA testdev_balanced is 12,578 questions over just 398 images).
#
# The `train_` prefix is load-bearing: task_of matches name.startswith(task+"_"),
# so a bare "docvqa" key would swallow the "docvqa_val_*" names.
#
# `max_visual_tokens` overrides MERGER_TRAIN["max_visual_tokens"] (App. A.1:
# 1,600 visual tokens, reduced to 600 for GQA).
TRAIN_TASKS = {
    "train_gqa":       _task("lmms-lab/GQA", "train", max_visual_tokens=600),
    "train_textvqa":   _task("lmms-lab/textvqa", "train", cap=True),
    "train_docvqa":    _task("lmms-lab/DocVQA", "train"),
    "train_infovqa":   _task("lmms-lab/DocVQA", "train"),
    "train_chartqa":   _task("lmms-lab/ChartQA", "train"),
    "train_scienceqa": _task("lmms-lab/ScienceQA", "train"),
}
TASKS.update(TRAIN_TASKS)

# --------------------------------------------------------------------------
# crop foveation (not part of the paper; see models/crop_selector.py)
# --------------------------------------------------------------------------
# A crop-foveated input is a global view at linear scale BASE_SCALE plus one
# cell of a SELECTION_GRID x SELECTION_GRID tiling at linear scale CROP_SCALE:
# 0.5**2 + (1/3)**2 = 36% of native's token budget on every image.
BASE_SCALE = 0.5
CROP_SCALE = 1.0 / SELECTION_GRID

# Attention-capture passes run on an image already resized to the model's
# selection cap and are not part of the token budget, so they keep a floor low
# enough never to upscale it. The processor's own floor must stay tight for the
# same reason: it re-resizes on top of whatever the vision preprocessor already
# did, and a 200704 floor there silently upscales every crop.
CROP_MIN_PIXELS = 28 * 28
PROCESSOR_MIN_PIXELS = 28 * 28


def train_max_visual_tokens(task, default):
    """Per-task visual-token cap for merger training; never above `default`."""
    v = TASKS.get(task, {}).get("max_visual_tokens")
    return default if v is None else min(v, default)


def build_question(task, question):
    """(Optionally capitalised) benchmark question + the common instruction."""
    q = question.capitalize() if TASKS[task]["capitalize_question"] else question
    return f"{q}{INSTRUCTION}"


def generation_kwargs(tokenizer):
    """Greedy, single beam, at most 16 new tokens -- identical for every condition."""
    return {**GENERATION, "use_cache": True, "pad_token_id": tokenizer.pad_token_id}


def scaled_bounds(scale):
    """The backbone's (min_pixels, max_pixels) scaled by AREA."""
    area = float(scale) ** 2
    return int(BACKBONE["min_pixels"] * area), int(BACKBONE["max_pixels"] * area)


def image_entry(image, role, scale=None):
    """A vision-processor image entry with this protocol's bounds for `role`.

      native     the whole image at full resolution -- the backbone bounds verbatim
      base       the downscaled global view -- bounds scaled by BASE_SCALE**2
      crop       one high-res cell    -- bounds scaled by CROP_SCALE**2
      selection  attention-capture pass, outside the token budget
    """
    if role == "native":
        mn, mx = BACKBONE["min_pixels"], BACKBONE["max_pixels"]
    elif role == "base":
        mn, mx = scaled_bounds(BASE_SCALE if scale is None else scale)
    elif role == "crop":
        mn, mx = scaled_bounds(CROP_SCALE if scale is None else scale)
    elif role == "selection":
        mn, mx = CROP_MIN_PIXELS, BACKBONE["max_pixels"]
    else:
        raise ValueError(f"unknown view role {role!r}")
    return {"type": "image", "image": image, "min_pixels": mn, "max_pixels": mx}


def model_kwargs(needs_attention=False):
    """Kwargs for the Qwen2.5-VL wrapper.

    needs_attention forces the eager attention implementation: flash-attention
    never materialises the attention matrix, so a query-attention ROI probe
    silently receives None instead of failing.
    """
    return {
        "pretrained": BACKBONE["model_id"],
        "torch_dtype": BACKBONE["dtype"],
        "min_pixels": BACKBONE["min_pixels"],
        "max_pixels": BACKBONE["max_pixels"],
        "system_prompt": BACKBONE["system_prompt"],
        "attn_implementation": "eager" if needs_attention else BACKBONE["attn_implementation"],
    }


def seed_everything(seed=None):
    """Only matters for temperature>0 probes; greedy decoding is deterministic."""
    import numpy as np
    import torch

    s = SEED if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    return s


def assert_matches(model_wrapper, needs_attention=False):
    """Fail loudly if a loaded model does not match this protocol.

    A wrapper loaded with different pixel bounds produces a different number of
    visual tokens for the same image, which makes every cost number in a run
    incomparable to every other.
    """
    mw = model_wrapper
    want_attn = "eager" if needs_attention else BACKBONE["attn_implementation"]
    got = getattr(mw.model.config, "_attn_implementation", want_attn)
    problems = []
    if int(getattr(mw, "min_pixels", BACKBONE["min_pixels"])) != BACKBONE["min_pixels"]:
        problems.append(f"min_pixels {mw.min_pixels} != {BACKBONE['min_pixels']}")
    if int(getattr(mw, "max_pixels", BACKBONE["max_pixels"])) != BACKBONE["max_pixels"]:
        problems.append(f"max_pixels {mw.max_pixels} != {BACKBONE['max_pixels']}")
    if got != want_attn:
        problems.append(f"attn_implementation {got!r} != {want_attn!r}")
    if problems:
        raise RuntimeError("model does not match the protocol: " + "; ".join(problems))
    return True
