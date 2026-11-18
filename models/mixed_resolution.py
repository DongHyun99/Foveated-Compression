#!/usr/bin/env python3
"""Mixed-resolution construction (Sec. 3.4): one sequence, two resolutions.

The selected cell's compressed units are replaced in place by their native
3x3 token blocks; every other unit stays compressed. Native and compressed
tokens share the native visual-grid coordinate system for M-RoPE: native
tokens use unit stride, compressed tokens the top-left coordinate of their
block.

There are two ways to implement "keep one cell at native resolution, compress
the rest", and only one of them is a compression method.

  replication      Sequence length stays at the native grid N. Each compressed
                   vector is written into all of its native slots, so M-RoPE is
                   byte-identical to native and nothing has to be built.
                   Accuracy is faithful -- verified at 64.95% against 64.79%,
                   +0.16pp [-0.78, +1.10] -- but the LLM still prefills N
                   tokens, so it measures accuracy and nothing about cost.

  mixed sequence   THIS MODULE. Sequence length is the real one. Kept tokens
                   sit at a 28 px pitch and compressed ones at 84 px, so there
                   is no single regular grid for the model's own
                   `get_rope_index` to read coordinates off -- which is exactly
                   what replication was avoiding. This module supplies those
                   coordinates directly.

Measured on one A5000 at batch 1, per visual token: CPU preprocess 0.0134 ms,
ViT 0.251 ms, LLM prefill 0.204 ms. Prefill is 45% of the per-token cost, so a
genuinely shorter sequence removes what replication cannot.

HOW IT FITS TOGETHER. Three things have to agree on one token order:

    1. the token tensor                         build_tokens()
    2. the M-RoPE coordinates                   layout.hw, installed by MixedFrame
    3. the number of image_pad slots emitted    frame_size()

Each `layout_*` function produces all three from one scan, so they cannot
drift apart, and MixedFrame raises at prefill if the processor's placeholder
count disagrees with the layout's length. That check is why a reported token
count is the number the LLM actually consumed rather than a number a script
computed alongside it.

THE FRAME. The processor decides how many image_pad slots to emit from
`image_grid_thw` alone, so any grid whose merged count equals L works.
`frame_size` returns the squarest such image. Its pixels are never read:
MixedFrame replaces `visual.forward` outright, so the frame costs one CPU
preprocess and no ViT pass at all.

    lay, _ = layout_cells(Rh, Rw, keep_cells=[4])
    tok = build_tokens(hout, comp, lay)
    W, H = frame_size(lay.n_tokens)
    with MixedFrame(mw, tok, lay):
        answer = generate(mw, [{"type": "image", "image": Image.new("RGB", (W, H))}], ...)
"""
import types
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Layout:
    """One mixed-resolution token order.

    src      [L]      row to take from the native tokens (is_comp False) or
                      from the compressed tokens (True)
    is_comp  [L]      which array `src` indexes
    hw       [L, 2]   (h, w) in the NATIVE grid, for M-RoPE
    n_tokens int      L
    grid     (Rh, Rw) native grid this layout was built for
    """
    src: np.ndarray
    is_comp: np.ndarray
    hw: np.ndarray
    n_tokens: int
    grid: tuple

    @property
    def compression(self):
        """L / N -- the fraction of native's visual tokens this sequence costs."""
        Rh, Rw = self.grid
        return self.n_tokens / (Rh * Rw)


def _finish(src, is_comp, hw, Rh, Rw):
    """Sort one scan into causal order and wrap it in a Layout.

    M-RoPE coordinates do NOT replace causal sequence order. Sorting by the
    native raster coordinates makes an all-kept layout exactly the model's own
    native token order, and stops a mixed layout from giving a later image row
    visibility ahead of an earlier-row token.
    """
    src = np.asarray(src, np.int64)
    is_comp = np.asarray(is_comp, bool)
    hw = np.asarray(hw, np.int64)
    order = np.lexsort((hw[:, 1], hw[:, 0]))
    return Layout(src[order], is_comp[order], hw[order], len(src), (Rh, Rw))


def layout_mask(Rh, Rw, keep_units, ms=2):
    """Packed sequence for an arbitrary per-unit keep mask at the native pitch.

    `keep_units` has one bool per complete ms x ms block of native tokens. Kept
    blocks emit their native tokens; compressed blocks emit one token each. A
    ragged final row or column that cannot form a complete block is emitted at
    native resolution, which lets this path use the model's exact factor-28
    native image grid instead of distorting the image to make the arithmetic
    even.

    Coordinates are the NATIVE grid's, so a kept token gets exactly the (h, w)
    it would have had under native and a compressed token gets its block's
    top-left. Adjacent compressed tokens are therefore `ms` apart while
    adjacent kept tokens are 1 apart: the coordinate spacing ENCODES the scale
    difference rather than hiding it.
    """
    if Rh <= 0 or Rw <= 0 or ms <= 0:
        raise ValueError(f"invalid grid {Rh}x{Rw}, merge={ms}")
    Gh, Gw = Rh // ms, Rw // ms
    M = Gh * Gw
    keep = np.asarray(keep_units, dtype=bool).reshape(-1)
    if keep.size != M:
        raise ValueError(f"keep_units has {keep.size} entries, unit grid is {Gh}x{Gw}={M}")

    src, is_comp, hw = [], [], []
    for a in range(Gh):
        for b in range(Gw):
            u = a * Gw + b
            if keep[u]:
                for dr in range(ms):
                    for dc in range(ms):
                        r, c = ms * a + dr, ms * b + dc
                        src.append(r * Rw + c)
                        is_comp.append(False)
                        hw.append((r, c))
            else:
                src.append(u)
                is_comp.append(True)
                hw.append((ms * a, ms * b))
        # ragged right edge, kept native so the source image is never rescaled
        for dr in range(ms):
            r = ms * a + dr
            for c in range(Gw * ms, Rw):
                src.append(r * Rw + c)
                is_comp.append(False)
                hw.append((r, c))
    # ragged bottom edge, including the bottom-right corner
    for r in range(Gh * ms, Rh):
        for c in range(Rw):
            src.append(r * Rw + c)
            is_comp.append(False)
            hw.append((r, c))

    return _finish(src, is_comp, hw, Rh, Rw)


def layout_cells(Rh, Rw, keep_cells, merge=3, grid=3):
    """Packed sequence for the Foveated Merger: preserved cells native, rest 9:1.

    The 252-multiple resize makes Rh and Rw divisible by merge*grid, so a 3x3
    cell boundary always falls on a compressed-token boundary -- no cell is
    ever split across a compressed token and there is no ragged remainder.
    That is the whole point of the geometry: cost becomes exactly
    (K * cell_native + (9 - K) * cell_comp) / (Rh * Rw), i.e. 11.1 / 21.0 /
    30.9 / 40.7 % for K = 0..3 on every image regardless of its size.

    -> (Layout, comp_index), where comp_index[j] is the row-major position of
    emitted compressed token j in the (Rh/merge, Rw/merge) compressed grid.
    The caller uses it to gather exactly the compressed tokens this layout
    emits, instead of computing all of them and throwing most away.
    """
    if Rh % (merge * grid) or Rw % (merge * grid):
        raise ValueError(f"grid {Rh}x{Rw} is not divisible by merge*grid={merge * grid}; "
                         f"resize to a multiple of 14*2*{merge * grid} px first")
    Ch, Cw = Rh // merge, Rw // merge        # compressed grid
    ch, cw = Rh // grid, Rw // grid          # native rows/cols per cell
    kch, kcw = Ch // grid, Cw // grid        # compressed rows/cols per cell
    keep = {int(k) for k in keep_cells}
    if any(not 0 <= k < grid * grid for k in keep):
        raise ValueError(f"cell ids must be in [0, {grid * grid})")

    src, is_comp, hw, comp_index = [], [], [], []
    for cell in range(grid * grid):
        a, b = divmod(cell, grid)
        if cell in keep:
            for r in range(a * ch, (a + 1) * ch):
                for c in range(b * cw, (b + 1) * cw):
                    src.append(r * Rw + c)
                    is_comp.append(False)
                    hw.append((r, c))
        else:
            for r in range(a * kch, (a + 1) * kch):
                for c in range(b * kcw, (b + 1) * kcw):
                    src.append(len(comp_index))
                    comp_index.append(r * Cw + c)
                    is_comp.append(True)
                    hw.append((r * merge, c * merge))

    return _finish(src, is_comp, hw, Rh, Rw), comp_index


def build_tokens(native, compressed, lay):
    """[L, D] in `lay`'s order. `native` is row-major, `compressed` is src-indexed."""
    if native.shape[-1] != compressed.shape[-1]:
        raise ValueError(f"width mismatch: native {tuple(native.shape)} vs "
                         f"compressed {tuple(compressed.shape)}")
    idx = torch.as_tensor(lay.src, device=native.device)
    m = torch.as_tensor(lay.is_comp, device=native.device)
    out = torch.empty(lay.n_tokens, native.shape[-1],
                      device=native.device, dtype=native.dtype)
    out[~m] = native.index_select(0, idx[~m])
    out[m] = compressed.to(native.dtype).index_select(0, idx[m])
    return out


def frame_size(n_tokens, ms=2, patch=14):
    """(W, H) px of the squarest dummy image whose merged-token count is n_tokens.

    The pixels are never read -- MixedFrame short-circuits the tower -- so this
    only has to make the processor emit exactly n_tokens image_pad slots. A
    prime n_tokens degenerates to a 1 x n_tokens strip, which is correct for
    the same reason, just marginally worse for cache behaviour.
    """
    unit = patch * ms
    p = max(d for d in range(1, int(n_tokens ** 0.5) + 1) if n_tokens % d == 0)
    return (n_tokens // p) * unit, p * unit


class MixedFrame:
    """Context manager: hand `tokens` to the LLM with `lay`'s coordinates.

    Patches two things on the way in and restores both on the way out.

      visual.forward    returns `tokens` without running the tower, which saves
                        the dummy frame's ViT pass entirely.
      get_rope_index    assigns (t, h, w) per image token from lay.hw instead
                        of reading a regular grid off image_grid_thw.

    `get_rope_index` is only called at prefill -- the model caches rope_deltas
    for decode -- so the patch does not have to survive generation, but it must
    be installed before `generate()` is entered.

    Batch 1, one image, no video. A mixed sequence has no meaning for a padded
    batch and the coordinate scan assumes a single contiguous placeholder run.
    """

    def __init__(self, mw, tokens, lay):
        self.mw = mw
        self.tokens = tokens
        self.lay = lay
        self._saved = None

    def __enter__(self):
        mw = self.mw
        visual = mw.model.model.visual
        inner = mw.model.model
        tokens, lay = self.tokens, self.lay
        image_token_id = mw.model.config.image_token_id

        def fake_forward(pixel_values, grid_thw=None, **kw):
            n = int(grid_thw.prod(-1).sum() // visual.spatial_merge_size ** 2)
            if n != lay.n_tokens:
                raise ValueError(f"frame gives {n} pads, layout has {lay.n_tokens}")
            return tokens.to(device=pixel_values.device, dtype=pixel_values.dtype)

        hw_t = torch.as_tensor(lay.hw, dtype=torch.long)

        def rope_index(self_inner, input_ids=None, image_grid_thw=None,
                       video_grid_thw=None, second_per_grid_ts=None,
                       attention_mask=None):
            if video_grid_thw is not None:
                raise ValueError("MixedFrame does not support videos")
            if input_ids.shape[0] != 1:
                raise ValueError("MixedFrame is batch-1 only")
            if attention_mask is not None and not bool(attention_mask.bool().all()):
                raise ValueError("MixedFrame does not support padded batches")
            ids = input_ids[0]
            length = ids.shape[0]
            img = (ids == image_token_id).nonzero(as_tuple=True)[0]
            if img.numel() != lay.n_tokens:
                raise ValueError(f"{img.numel()} image pads, layout has {lay.n_tokens}")
            s, e = int(img[0]), int(img[-1]) + 1
            if e - s != lay.n_tokens:
                raise ValueError("image pads are not contiguous")

            pos = torch.zeros(3, length, dtype=torch.long)
            pos[:, :s] = torch.arange(s).unsqueeze(0).expand(3, -1)
            # the image run: t is constant (single frame), h/w come from the
            # native grid so both scales share one coordinate space
            pos[0, s:e] = s
            pos[1, s:e] = hw_t[:, 0] + s
            pos[2, s:e] = hw_t[:, 1] + s
            # Text resumes past the image's full NATIVE extent, the same rule
            # the unpatched model uses. Resuming from the emitted coordinates'
            # maximum instead shifts the all-compressed case one position
            # earlier than every other rate, which makes rates incomparable.
            Rh, Rw = lay.grid
            nxt = s + max(Rh, Rw)
            if e < length:
                pos[:, e:] = torch.arange(nxt, nxt + length - e).unsqueeze(0).expand(3, -1)

            position_ids = pos.unsqueeze(1).to(input_ids.device)
            deltas = (position_ids.max() + 1 - length).view(1, 1)
            return position_ids, deltas

        # A prime token count produces a 1 x L dummy frame. The generic image
        # helper rejects aspect ratios above 200 even though this frame's
        # pixels are never consumed, so that one guard is bypassed while exact
        # factor alignment and pixel bounds stay enforced.
        import qwen_vl_utils.vision_process as qvp
        from transformers.models.qwen2_vl import image_processing_qwen2_vl as qslow
        from transformers.models.qwen2_vl import image_processing_qwen2_vl_fast as qfast

        resize_saved = []
        for mod in (qvp, qslow, qfast):
            original = mod.smart_resize

            def unrestricted(height, width, factor=28, min_pixels=None, max_pixels=None,
                             _original=original):
                if max(height, width) / min(height, width) <= 200:
                    return _original(height, width, factor, min_pixels, max_pixels)
                if height % factor or width % factor:
                    raise ValueError(f"thin dummy frame {height}x{width} is not "
                                     f"factor-{factor} aligned")
                lo = factor * factor if min_pixels is None else min_pixels
                hi = 10 ** 18 if max_pixels is None else max_pixels
                if not lo <= height * width <= hi:
                    raise ValueError(f"thin dummy frame area {height * width} "
                                     f"outside [{lo}, {hi}]")
                return height, width

            resize_saved.append((mod, original))
            mod.smart_resize = unrestricted

        self._saved = (visual.forward, inner.get_rope_index, inner.rope_deltas, resize_saved)
        visual.forward = fake_forward
        inner.get_rope_index = types.MethodType(rope_index, inner)
        inner.rope_deltas = None            # force a recompute at prefill
        return self

    def __exit__(self, *exc):
        visual = self.mw.model.model.visual
        inner = self.mw.model.model
        fwd, rope, deltas, resize_saved = self._saved
        # visual.forward was an instance attribute added above; drop it so the
        # class method takes over again rather than leaving a bound copy behind
        try:
            del visual.forward
        except AttributeError:
            visual.forward = fwd
        inner.get_rope_index = rope
        inner.rope_deltas = deltas
        for mod, resize in resize_saved:
            mod.smart_resize = resize
        self._saved = None
        return False
