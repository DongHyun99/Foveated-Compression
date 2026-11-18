#!/usr/bin/env python3
"""Image sizing and index maps shared by every stage (App. A.3).

Three invariants everything else depends on.

  1. Every reported condition, the native reference included, resizes to a
     multiple of 252 px, because 252 = 14 * 2 * 3 * 3 is the smallest pitch at
     which a 3x3 selection-cell boundary always coincides with a
     compressed-token boundary. That is what makes the token budget exact on
     every image rather than approximate.

  2. Merger training uses the merger's own block pitch (84 px = 14 * 2 * 3)
     under a per-task visual-token cap; it never needs cell alignment.

  3. Native visual tokens are addressed ROW-MAJOR everywhere. See
     `models.token_injection` for where that is established and why the
     alternative fails silently -- the vision tower hands back window-permuted
     tokens from every capture point inside it, and reshaping those to
     (gh, gw) scrambles space without error.
"""
import numpy as np


def factor_aligned_size(w, h, factor, min_px, max_px):
    """Resize target with both sides a multiple of `factor`, within pixel bounds."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    hh, ww = smart_resize(h, w, factor=factor, min_pixels=min_px, max_pixels=max_px)
    return ww, hh


def training_size(w, h, task, factor, max_visual_tokens):
    """Merger-training resize under the per-task visual-token cap (App. A.1).

    Images over the cap are DOWNSCALED, never skipped: the merger is applied
    per block with shared weights, so a block learned at a lower resolution
    still transfers, while a block never seen at all does not.
    """
    import config

    cap = config.train_max_visual_tokens(task, max_visual_tokens)
    return factor_aligned_size(
        w, h, factor, config.BACKBONE["min_pixels"],
        min(config.BACKBONE["max_pixels"], cap * 28 * 28),
    )


def training_view(image, task, factor, max_visual_tokens):
    """The resized RGB image merger training sees, and the teacher is cached on.

    One function for both sides: if the teacher dump and the student were
    resized separately, the cached distribution would belong to a different
    image than the one the student is trained against, without any error.
    """
    from PIL import Image

    W, H = training_size(image.width, image.height, task, factor, max_visual_tokens)
    return image.convert("RGB").resize((W, H), Image.BICUBIC)


def aligned_size(w, h, max_visual_tokens=16384):
    """Resize target for every reported condition: both sides a multiple of 252 px."""
    import config

    return factor_aligned_size(w, h, config.ALIGNMENT_PX, config.BACKBONE["min_pixels"],
                               min(config.BACKBONE["max_pixels"], max_visual_tokens * 28 * 28))


def block_index(Rh, Rw, ms=3):
    """Row-major indices of each ms x ms block of native visual tokens.

    -> ([M, ms*ms] int array, (Gh, Gw)) with M = Gh*Gw. Row u gathers the
    native tokens N(u) of compressed unit u; slot order inside a row is
    (row-major block, row-major offset), the order the frozen 2x2 visual
    merger's concatenation expects when ms=2.
    """
    Gh, Gw = Rh // ms, Rw // ms
    a, b = np.divmod(np.arange(Gh * Gw), Gw)
    return np.stack([(ms * a + dr) * Rw + (ms * b + dc)
                     for dr in range(ms) for dc in range(ms)], 1), (Gh, Gw)


def coarse_cells(Gh, Gw, grid=3):
    """Partition the compressed-unit grid into `grid` x `grid` coarse cells.

    -> (cell_of_unit [M], sizes [grid*grid]). Rows and columns are split into
    near-equal chunks, so the cells are NOT all the same size when the grid
    does not divide evenly. Anything that prices a configuration in tokens must
    read `sizes`, not M / grid**2.
    """
    re_ = np.linspace(0, Gh, grid + 1).round().astype(int)
    ce_ = np.linspace(0, Gw, grid + 1).round().astype(int)
    a, b = np.divmod(np.arange(Gh * Gw), Gw)
    cell = np.zeros(Gh * Gw, np.int64)
    for i in range(grid):
        for j in range(grid):
            m = (a >= re_[i]) & (a < re_[i + 1]) & (b >= ce_[j]) & (b < ce_[j + 1])
            cell[m] = i * grid + j
    return cell, np.bincount(cell, minlength=grid * grid)


def grid_candidates(width, height, n_h, n_w):
    """Systematic, signal-agnostic n_h x n_w tiling -> [(x0, y0, x1, y1), ...].

    Signal-agnostic on purpose: the candidate pool must not be produced by the
    same signal a selector is later evaluated against, or the evaluation only
    measures how well that signal agrees with itself.
    """
    xs = np.linspace(0, width, n_w + 1).round().astype(int)
    ys = np.linspace(0, height, n_h + 1).round().astype(int)
    return [(int(xs[j]), int(ys[i]), int(xs[j + 1]), int(ys[i + 1]))
            for i in range(n_h) for j in range(n_w)]


def px_box_to_grid_box(px_box, orig_size, base_gh, base_gw):
    """Original-image pixel box -> (row, col, height, width) on the RoPE grid.

    The inverse direction of what a region-of-interest selector produces: the
    selector reasons in grid coordinates because that is what M-RoPE indexes,
    but every crop is taken in pixels. Both ends round OUTWARD so a box never
    loses a partially covered grid cell -- a half-covered cell containing the
    answer is worth more than an exact area match.
    """
    x0, y0, x1, y1 = px_box
    W, H = orig_size
    top_r = int(np.floor(y0 / H * base_gh))
    top_c = int(np.floor(x0 / W * base_gw))
    bottom = int(np.ceil(y1 / H * base_gh))
    right = int(np.ceil(x1 / W * base_gw))
    top_r = max(0, min(top_r, base_gh - 1))
    top_c = max(0, min(top_c, base_gw - 1))
    bottom = max(top_r + 1, min(bottom, base_gh))
    right = max(top_c + 1, min(right, base_gw))
    return (top_r, top_c, bottom - top_r, right - top_c)


def cell_pixel_box(cell, width, height, grid=3):
    """Pixel box of coarse cell `cell` under a `grid` x `grid` tiling."""
    a, b = divmod(int(cell), grid)
    xs = np.linspace(0, width, grid + 1).round().astype(int)
    ys = np.linspace(0, height, grid + 1).round().astype(int)
    return int(xs[b]), int(ys[a]), int(xs[b + 1]), int(ys[a + 1])


def region_signal_mean(grid, grid_hw, image_size, px_box):
    """Mean of a [gh, gw] signal grid inside an original-image pixel box.

    Used to score a candidate region against any dense per-grid-cell signal
    (query attention, saliency, token norm) on the same footing as a measured
    gain, so the two are directly comparable.
    """
    gh, gw = grid_hw
    r, c, dh, dw = px_box_to_grid_box(px_box, image_size, gh, gw)
    return float(np.asarray(grid).reshape(gh, gw)[r:r + dh, c:c + dw].mean())


def token_budget_table(grid=3):
    """Exact visual-token budget of preserving K of grid**2 cells (Eq. 5).

    (K * native_cell + (n - K) * compressed_cell) / total, which under the
    252-multiple resize is independent of the image: 11.11 / 20.99 / 30.86 /
    40.74 % for K = 0..3 on a 3x3 grid.
    """
    n = grid * grid
    per_cell = 1.0 / n
    return {k: k * per_cell + (n - k) * per_cell / 9.0 for k in range(n + 1)}
