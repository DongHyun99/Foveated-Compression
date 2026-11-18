#!/usr/bin/env python3
"""Crop-selector features and their shard format -- NOT part of the paper.

FEATURES. A candidate region is described by three kinds of thing, which fail
in different ways and are therefore all kept:

  pixel        laplacian variance and saturation spread, computed with no
               model at all. The single strongest hand-made feature measured
               (+6.1% on the informative subset) is laplacian variance, and
               the reason is direct: almost every large-gain case is
               "illegible text becomes legible", and Laplacian variance is the
               classical blur detector. It costs no GPU.

  ViT content  sub-cell-resolved mean vectors from the projector output and
               from intermediate tower layers. The MEAN says what is there.

  ViT texture  a randomly projected per-sub-cell standard deviation. The mean
               cannot express texture, and texture is exactly what a 2x
               downscale destroys -- so the variance across patches, not their
               average, is the part that predicts whether resolution will
               help. This is the learned counterpart of laplacian variance.

Everything is resolved to sub-cells rather than pooled to one vector per
candidate, because a question about a sign in the corner of a cell is not
answered by that cell's average.

THE SHARD FORMAT. fp16 npz, column-oriented. JSON costs about 12 bytes per
float, which puts a 45k-image pool with this feature set at roughly 1.6 TB;
fp16 npz puts it at ~68 GB. The format, not the disk, was the binding
constraint on how rich the input representation could be.

Two failure modes are guarded here because both actually happened:

  a key absent from STACKED/SCALARS/RAGGED is silently dropped by save_shard,
  which is how laplacian variance -- the strongest single feature -- was lost
  on the first extraction run. save_shard now raises on any unclassified key.

  an NpzFile re-inflates a whole column on every `z[k]`, so a per-record
  `z[k][i]` read costs n_records x n_keys full-column inflations. On an 8 GB
  shard that is ~140,000 of them. `load_shard_arrays` inflates each column
  exactly once, which is the difference between minutes and days.

This module is pure numpy plus OpenCV -- no torch, no VLM -- so the training
side can import it without CUDA.
"""
import glob
import os

import numpy as np

# Per-record arrays stacked across images into one [N, ...] array.
STACKED = (
    # oracle labels
    "cand_boxes", "gains", "correct", "qattn_scores", "div_scores", "cos_sims",
    # sub-cell-resolved ViT content
    "subcell_merger", "subcell_layers", "layer_std_proj",
    "vit_mean", "vit_std_proj", "vit_max_proj",
    # scalar per-candidate detail features
    "mean_c_std", "mean_c_max_minus_mean", "dissim", "token_norm_std",
    "laplacian_var", "laplacian_delta", "sat_std",
    "question_hidden",
    # multi-cell sweep: one row per evaluated cell subset
    "k", "cell_mask", "score", "gain",
    # occlusion counterfactual: per-cell signals at grid 2/3/4
    "cell_score2", "cell_score3", "cell_score4",
    "cell_logprob2", "cell_logprob3", "cell_logprob4",
    "cell_entropy2", "cell_entropy3", "cell_entropy4",
)
SCALARS = ("resize_ok", "native_ok", "baseline_lp", "baseline_score",
           "baseline_logprob", "baseline_entropy",
           "score2", "score3", "score4",
           "best_region2", "best_region3", "best_region4")
RAGGED = ("name", "task", "answer", "resize_ans", "native_ans", "cand_ans",
          "question_token_hidden", "qattn_grid", "gen", "baseline_gen")


# --------------------------------------------------------------------------
# pixel features -- no model, no GPU
# --------------------------------------------------------------------------
def cell_pixel_features(image, px_box):
    """(laplacian variance, saturation std) of one candidate crop.

    Laplacian variance is the classical blur/sharpness detector used by camera
    autofocus. Saturation spread is a different notion of visual complexity --
    colour diversity rather than edge energy -- and the two disagree often
    enough to both be worth carrying.
    """
    import cv2

    arr = np.array(image.crop(px_box).convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_std = float(hsv[:, :, 1].astype(np.float64).std())
    return laplacian_var, sat_std


def laplacian_delta(image, px_box, base_scale=0.5):
    """Sharpness LOST by viewing this region through the downscaled base view.

    The absolute sharpness of a region says how detailed it is; the difference
    against the same region as the base view renders it says how much of that
    detail the cheap view already has. Only the second quantity is a reason to
    spend a crop.
    """
    import cv2

    crop = image.crop(px_box).convert("RGB")
    full = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    w = max(1, int(crop.width * base_scale))
    h = max(1, int(crop.height * base_scale))
    small = np.array(crop.resize((w, h))).astype(np.uint8)
    small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(small, (crop.width, crop.height), interpolation=cv2.INTER_CUBIC)
    return float(cv2.Laplacian(full, cv2.CV_64F).var() - cv2.Laplacian(small, cv2.CV_64F).var())


# --------------------------------------------------------------------------
# sub-cell pooling of a spatially coherent feature grid
# --------------------------------------------------------------------------
def _clamp_slice(ys, xs, i, j, H, W):
    """Sub-cell bounds that are always non-empty AND inside the grid.

    A cell can be only ~3 merged tokens tall, so linspace-derived sub-cell
    edges collide after rounding. Without the clamp the last sub-cell gets an
    EMPTY slice whose mean is NaN, and that NaN propagates into the PCA fit and
    kills the entire cache build several hours in.
    """
    a = min(int(ys[i]), H - 1)
    b = max(a + 1, min(int(ys[i + 1]), H))
    c = min(int(xs[j]), W - 1)
    d = max(c + 1, min(int(xs[j + 1]), W))
    return a, b, c, d


def _grid_window(grid, box, img_size):
    H, W = grid.shape[:2]
    img_w, img_h = img_size
    x0, y0, x1, y1 = box
    gx0 = max(0, min(W - 1, int(x0 / img_w * W)))
    gx1 = max(gx0 + 1, min(W, int(np.ceil(x1 / img_w * W))))
    gy0 = max(0, min(H - 1, int(y0 / img_h * H)))
    gy1 = max(gy0 + 1, min(H, int(np.ceil(y1 / img_h * H))))
    return gy0, gy1, gx0, gx1


def pool_subcells(grid, box, img_size, sub=3):
    """Mean-pool a [H, W, D] feature grid over the sub x sub regions of `box`.

    `grid` MUST already be un-permuted into row-major order -- see
    `models.token_injection.unpermute_vit_seq`. Feeding a raw in-tower capture here
    reproduces the window-order scrambling that made every ViT-derived feature
    useless while the pixel features still worked.
    """
    H, W = grid.shape[:2]
    gy0, gy1, gx0, gx1 = _grid_window(grid, box, img_size)
    ys = np.linspace(gy0, gy1, sub + 1).round().astype(int)
    xs = np.linspace(gx0, gx1, sub + 1).round().astype(int)
    out = np.empty((sub * sub, grid.shape[-1]), dtype=np.float32)
    for i in range(sub):
        for j in range(sub):
            a, b, c, d = _clamp_slice(ys, xs, i, j, H, W)
            out[i * sub + j] = grid[a:b, c:d].reshape(-1, grid.shape[-1]).mean(axis=0)
    return out


def pool_subcells_mean_std(grid, box, img_size, proj, sub=3):
    """As `pool_subcells`, plus a randomly projected per-sub-cell std.

    `proj` is a fixed [D, d] random projection; the std vector is full-width
    and storing it raw would multiply the cache size for a signal that is
    diffuse by nature.
    """
    H, W = grid.shape[:2]
    gy0, gy1, gx0, gx1 = _grid_window(grid, box, img_size)
    ys = np.linspace(gy0, gy1, sub + 1).round().astype(int)
    xs = np.linspace(gx0, gx1, sub + 1).round().astype(int)
    means = np.empty((sub * sub, grid.shape[-1]), dtype=np.float32)
    stds = np.empty((sub * sub, proj.shape[1]), dtype=np.float32)
    for i in range(sub):
        for j in range(sub):
            a, b, c, d = _clamp_slice(ys, xs, i, j, H, W)
            p = grid[a:b, c:d].reshape(-1, grid.shape[-1])
            means[i * sub + j] = p.mean(axis=0)
            stds[i * sub + j] = (p.std(axis=0) if p.shape[0] > 1
                                 else np.zeros(grid.shape[-1], np.float32)) @ proj
    return means, stds


def candidate_detail_features(grid, box, img_size):
    """Scalar per-candidate descriptors of a [H, W, D] feature grid.

    mean_c_std              spread across channels of the region mean
    mean_c_max_minus_mean   peakedness of that mean
    dissim                  1 - mean cosine similarity between the region's
                            tokens and the whole image's mean token: how much
                            this region stands OUT, which is the model-side
                            analogue of saliency
    token_norm_std          spread of token norms, a crude texture measure
    """
    H, W = grid.shape[:2]
    gy0, gy1, gx0, gx1 = _grid_window(grid, box, img_size)
    p = grid[gy0:gy1, gx0:gx1].reshape(-1, grid.shape[-1]).astype(np.float32)
    m = p.mean(axis=0)
    glob = grid.reshape(-1, grid.shape[-1]).astype(np.float32).mean(axis=0)
    denom = (np.linalg.norm(p, axis=1) * np.linalg.norm(glob) + 1e-8)
    cos = (p @ glob) / denom
    return {
        "mean_c_std": float(m.std()),
        "mean_c_max_minus_mean": float(m.max() - m.mean()),
        "dissim": float(1.0 - cos.mean()),
        "token_norm_std": float(np.linalg.norm(p, axis=1).std()),
    }


def zscore(a, axis=0):
    return (a - a.mean(axis=axis, keepdims=True)) / (a.std(axis=axis, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------
# shard IO
# --------------------------------------------------------------------------
def save_shard(path, records, meta=None):
    """Atomic fp16 npz write.

    Raises on any record key that is not classified, so a newly added feature
    can never be silently dropped -- which is exactly how the strongest single
    feature in this repository was lost once already.
    """
    if not records:
        return
    known = set(STACKED) | set(SCALARS) | set(RAGGED)
    unknown = sorted(set(records[0]) - known)
    if unknown:
        raise KeyError(f"unclassified record keys would be dropped: {unknown}. "
                       f"Add them to STACKED, SCALARS or RAGGED in data/crop_features.py")
    out = {}
    for k in STACKED:
        if k in records[0]:
            out[k] = np.stack([np.asarray(r[k], dtype=np.float32)
                               for r in records]).astype(np.float16)
    for k in SCALARS:
        if k in records[0]:
            out[k] = np.asarray([r[k] for r in records], dtype=np.float32)
    for k in RAGGED:
        if k in records[0]:
            out[k] = np.array([r[k] for r in records], dtype=object)
    if meta:
        out["_meta"] = np.array([meta], dtype=object)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def load_shard_arrays(path, keys=None):
    """Column-oriented shard read: {key: [N, ...]}, each column inflated once.

    Floats stay fp16 as stored -- cast at the point of use. Pass `keys` to skip
    inflating columns the caller does not need.
    """
    z = np.load(path, allow_pickle=True)
    avail = [k for k in z.files if k != "_meta"]
    want = avail if keys is None else [k for k in keys if k in avail]
    return {k: z[k] for k in want}


def load_shard(path):
    """Per-image dicts with floats restored to float32.

    Convenient, not cheap: float32 per-record copies run about 5x the on-disk
    size. Stream with `load_shard_arrays` when the pool is large.
    """
    cols = load_shard_arrays(path)
    if not cols:
        return []
    n = len(next(iter(cols.values())))
    recs = []
    for i in range(n):
        r = {}
        for k, col in cols.items():
            v = col[i]
            r[k] = v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float16 else v
        recs.append(r)
    return recs


def iter_shards(pattern):
    """Sorted complete shard paths, excluding save_shard's in-flight temp file.

    `shard_*.npz` also matches `shard_3.npz.tmp.npz` -- the `*` swallows
    `3.npz.tmp` -- and reading that mid-write raises BadZipFile. Cost one
    killed analysis process while an extraction was checkpointing beside it.
    """
    return [p for p in sorted(glob.glob(pattern))
            if not os.path.basename(p).endswith(".tmp.npz")]


def load_shard_names(path):
    """Cheap resume check: reads only the name column."""
    if not os.path.exists(path):
        return set()
    try:
        z = np.load(path, allow_pickle=True)
        return set(z["name"].tolist())
    except Exception:
        return set()


def fit_pca(x, d_out, n_fit=200_000, seed=0):
    """PCA basis fitted on at most n_fit rows, always from TRAIN rows only.

    Fitting on the full population would leak test-set structure into the
    input representation itself, which no downstream split can undo.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    if x.shape[0] > n_fit:
        x = x[rng.choice(x.shape[0], n_fit, replace=False)]
    p = PCA(n_components=d_out, random_state=seed)
    p.fit(x.astype(np.float32))
    return p


def pca_transform(p, x, chunk=100_000):
    """PCA.transform in row chunks, so one shard never materialises twice."""
    out = np.empty((x.shape[0], p.n_components_), dtype=np.float32)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = p.transform(x[i:i + chunk].astype(np.float32))
    return out
