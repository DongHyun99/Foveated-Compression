#!/usr/bin/env python3
"""Benchmark and training-split pools, in the one tuple shape everything uses.

EVERY task becomes `(name, image, question, answer)`. That uniformity is why
the labelling, training and evaluation loops have no per-benchmark branches:
the differences between benchmarks live in exactly two places -- the split
definition in `config.TASKS`, and the metric in `evaluation.metrics.official_score`.

Three kinds of benchmark have to be flattened into that shape:

  plain          question and answer are already usable strings (TextVQA,
                 ChartQA, DocVQA, OCRBench, POPE, MME, MMStar -- MMStar's
                 question column already carries its options inline).
  joined         GQA keeps questions and images in separate configs, joined
                 here on imageId.
  option columns MMBench keeps its four options in separate columns and its
                 answer as a letter, so the question text is assembled here.
  option index   ScienceQA keeps options as a list and the answer as an
                 integer index, so the index is converted to the matching
                 letter here.

Options are rendered as "(A) text" consistently, which is also why scoring
uses a robust extractor rather than a benchmark's own parser -- see
`evaluation.metrics`' module docstring for that deviation and its justification.

NAMES ARE "{task}_{row}", and that is load-bearing. `metrics.task_of` recovers
the task from the name, `metrics.row_index` recovers the dataset row so the
multi-answer fields a single answer string cannot carry can be re-fetched, and
the evaluation's resume logic keys on the same name. A name that does not
round-trip to its dataset row is a silent data-integrity failure.

LAZY IMAGES. An image is decoded once and shared across every question that
uses it, and object identity is what `group_items` groups on. Holding
one image object per question would multiply decode cost by the questions-per-
image factor, which is 32x on GQA.
"""
from collections import defaultdict

import config

MC_LETTERS = "ABCDEFGH"

# task -> (hf path, config, split, image column, question column, answer column)
PLAIN_SPECS = {
    "chartqa":    ("lmms-lab/ChartQA", None, "test", "image", "question", "answer"),
    "docvqa_val": ("lmms-lab/DocVQA", "DocVQA", "validation", "image", "question", "answers"),
    "textvqa":    ("lmms-lab/textvqa", None, "validation", "image", "question", "answers"),
    "ocrbench":   ("echo840/OCRBench", None, "test", "image", "question", "answer"),
    "pope":       ("lmms-lab/POPE", None, "test", "image", "question", "answer"),
    "mme":        ("lmms-lab/MME", None, "test", "image", "question", "answer"),
    "mmstar":     ("Lin-Chen/MMStar", None, "val", "image", "question", "answer"),
}

# Tasks whose official metric needs fields a single answer string cannot hold:
# reference-answer lists, OCRBench's source dataset, MME's (category,
# question_id) pairing, MMStar's l2 category.
EXTRA_FIELD_SPECS = {
    "docvqa_val": ("answers",),
    "textvqa": ("answers",),
    "ocrbench": ("answers", "dataset"),
    "mme": ("question_id", "category"),
    "mmstar": ("l2_category",),
}


class LazyImage:
    """One decoded RGB image, shared by every question that references it.

    `__getattr__` forwards to the decoded image so callers can treat this as a
    PIL image, while identity stays stable across an image's questions -- see
    the module docstring for why that identity matters.
    """

    __slots__ = ("_row", "_column", "_image")

    def __init__(self, row, column):
        self._row = row
        self._column = column
        self._image = None

    @property
    def image(self):
        if self._image is None:
            self._image = self._row[self._column].convert("RGB")
            self._row = None
        return self._image

    def __getattr__(self, item):
        return getattr(self.image, item)

    def convert(self, mode):
        return self.image.convert(mode)

    def resize(self, *a, **kw):
        return self.image.resize(*a, **kw)

    def crop(self, box):
        return self.image.crop(box)


def _row_indices(n_rows, cap, shard_idx, num_shards):
    """Evenly spaced rows, then this shard's stride.

    Evenly spaced rather than the first `cap`: many of these datasets are
    ordered by source document or by category, so a prefix is a biased
    subsample of exactly the dimension the benchmark varies along.
    """
    n = n_rows if cap is None else min(cap, n_rows)
    step = max(1, n_rows // max(1, n))
    return list(range(0, n_rows, step))[:n][shard_idx::num_shards]


def _sample_plain(task, cap, shard_idx, num_shards):
    from datasets import load_dataset

    path, subset, split, img_col, q_col, a_col = PLAIN_SPECS[task]
    ds = load_dataset(path, subset, split=split) if subset else load_dataset(path, split=split)
    out = []
    for i in _row_indices(len(ds), cap, shard_idx, num_shards):
        row = ds[i]
        if row.get(img_col) is None:
            continue
        answer = row.get(a_col)
        if isinstance(answer, (list, tuple)):
            answer = list(answer)
            if not answer:
                continue
        elif not str(answer or "").strip():
            continue
        out.append((f"{task}_{i}", LazyImage(row, img_col), row.get(q_col, ""), answer))
    return out


class _RowRef:
    """A dataset row fetched only when its image is first decoded."""

    __slots__ = ("_ds", "_idx")

    def __init__(self, ds, idx):
        self._ds, self._idx = ds, idx

    def __getitem__(self, column):
        return self._ds[self._idx][column]


def _sample_gqa(task, split, cap, shard_idx, num_shards):
    """GQA balanced questions joined to their images.

    lmms-lab/GQA keeps questions ("*_balanced_instructions") and images
    ("*_balanced_images") in separate configs, joined on imageId. One
    LazyImage per image id, so every question of an image shares it.
    """
    from datasets import load_dataset

    questions = load_dataset("lmms-lab/GQA", f"{split}_balanced_instructions", split=split)
    images = load_dataset("lmms-lab/GQA", f"{split}_balanced_images", split=split)
    row_of = {img_id: j for j, img_id in enumerate(images["id"])}
    lazy = {}
    out = []
    for i in _row_indices(len(questions), cap, shard_idx, num_shards):
        row = questions[i]
        j = row_of.get(row["imageId"])
        if j is None or not str(row.get("answer") or "").strip():
            continue
        if j not in lazy:
            lazy[j] = LazyImage(_RowRef(images, j), "image")
        out.append((f"{task}_{i}", lazy[j], row["question"], row["answer"]))
    return out


def _sample_mmbench(cap, shard_idx, num_shards, lang="en"):
    """MMBench dev. Options live in separate A-D columns; assemble them here."""
    from datasets import load_dataset

    task = f"mmbench_{lang}"
    ds = load_dataset("lmms-lab/MMBench", lang, split="dev")
    out = []
    for i in _row_indices(len(ds), cap, shard_idx, num_shards):
        row = ds[i]
        if row.get("image") is None:
            continue
        options = [(letter, row.get(letter)) for letter in "ABCD"]
        options = [(letter, opt) for letter, opt in options
                   if opt and str(opt).strip().lower() != "nan"]
        if len(options) < 2:
            continue
        answer = row.get("answer")
        if answer not in dict(options):
            continue
        q = row.get("question", "")
        hint = row.get("hint")
        if hint and str(hint).strip().lower() != "nan":
            q = f"{hint}\n{q}"
        q += "\n" + "\n".join(f"({letter}) {opt}" for letter, opt in options)
        out.append((f"{task}_{i}", LazyImage(row, "image"), q, answer))
    return out


def _sample_scienceqa(cap, shard_idx, num_shards):
    """ScienceQA-IMG. Answer is an integer index; convert it to its letter."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")
    out = []
    for i in _row_indices(len(ds), cap, shard_idx, num_shards):
        row = ds[i]
        choices = row.get("choices") or []
        answer_idx = row.get("answer")
        if row.get("image") is None or not choices or answer_idx is None:
            continue
        if answer_idx >= len(choices):
            continue
        q = row.get("question", "")
        hint = row.get("hint")
        if hint and str(hint).strip():
            q = f"{hint}\n{q}"
        q += "\n" + "\n".join(f"({MC_LETTERS[j]}) {c}" for j, c in enumerate(choices))
        out.append((f"scienceqa_img_{i}", LazyImage(row, "image"), q,
                    MC_LETTERS[answer_idx]))
    return out


def _sample_train_split(task, cap, shard_idx, num_shards):
    """A published TRAIN split, for the distillation loop.

    Training on train splits rather than carving a slice out of an eval split
    is what makes every evaluation split held out by construction -- see
    `config.TRAIN_TASKS` for the leak this avoids.
    """
    from datasets import load_dataset

    spec = config.TASKS[task]
    ds = load_dataset(spec["path"], split=spec["split"])
    out = []
    for i in _row_indices(len(ds), cap, shard_idx, num_shards):
        row = ds[i]
        img = row.get("image")
        if img is None:
            continue
        answer = row.get("answers") or row.get("answer")
        if isinstance(answer, (list, tuple)):
            answer = list(answer)
            if not answer:
                continue
        elif not str(answer or "").strip():
            continue
        q = row.get("question", "")
        choices = row.get("choices")
        if choices:
            idx = row.get("answer")
            if not isinstance(idx, int) or idx >= len(choices):
                continue
            q += "\n" + "\n".join(f"({MC_LETTERS[j]}) {c}" for j, c in enumerate(choices))
            answer = MC_LETTERS[idx]
        out.append((f"{task}_{i}", LazyImage(row, "image"), q, answer))
    return out


def sample_task_pool(tasks, cap_per_task=None, shard_idx=0, num_shards=1):
    """-> [(name, image, question, answer)] across `tasks`.

    cap_per_task=None takes each task's FULL official split. The per-task
    numbers this repository reports are computed over full splits, not
    subsamples, and the cap exists for development runs and for the
    distillation pool -- not for anything that gets published.
    """
    items = []
    for task in tasks:
        if task == "gqa":
            got = _sample_gqa(task, "testdev", cap_per_task, shard_idx, num_shards)
        elif task == "train_gqa":
            got = _sample_gqa(task, "train", cap_per_task, shard_idx, num_shards)
        elif task in PLAIN_SPECS:
            got = _sample_plain(task, cap_per_task, shard_idx, num_shards)
        elif task.startswith("mmbench_"):
            got = _sample_mmbench(cap_per_task, shard_idx, num_shards,
                                  lang=task.split("_", 1)[1])
        elif task == "scienceqa_img":
            got = _sample_scienceqa(cap_per_task, shard_idx, num_shards)
        elif task.startswith("train_"):
            got = _sample_train_split(task, cap_per_task, shard_idx, num_shards)
        else:
            raise ValueError(f"unknown task {task!r}")
        print(f"[ok] {task}: {len(got)} items (shard {shard_idx}/{num_shards})")
        items.extend(got)
    return items


def load_extra_fields(tasks=None):
    """{task: {row index: {field: value}}} for the metrics that need them.

    Restricting the load to the tasks actually being scored matters offline:
    requiring an unrelated dataset makes an otherwise fully cached run fail
    before the first sample.
    """
    from datasets import load_dataset

    wanted = set(tasks) if tasks is not None else set(EXTRA_FIELD_SPECS)
    out = defaultdict(dict)
    for task, fields in EXTRA_FIELD_SPECS.items():
        if task not in wanted:
            continue
        path, subset, split = PLAIN_SPECS[task][:3]
        ds = load_dataset(path, subset, split=split) if subset else load_dataset(path, split=split)
        ds = ds.select_columns(list(fields))      # never decode the images
        for i, row in enumerate(ds):
            out[task][i] = {f: row.get(f) for f in fields}
    return dict(out)


def group_items(items):
    """Items that share an image -> [[i, ...], ...], in first-appearance order.

    The sampler hands back one lazily loaded image object per image and
    repeats it across that image's questions, so object identity is the
    grouping key. Image groups are the unit of the selector's train/val split
    (App. A.1), so no image appears on both sides.
    """
    groups = {}
    for i, (_name, im, _q, _a) in enumerate(items):
        groups.setdefault(id(im), []).append(i)
    return list(groups.values())
