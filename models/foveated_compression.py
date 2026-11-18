#!/usr/bin/env python3
"""Foveated Compression end to end (Fig. 2): encode once, compress, select, replace.

    image --(252-aligned resize)--> frozen Qwen vision encoder   one pass
          native tokens [N, 3584] ------------------------------+
          patch grid [1, 1280, Ph, Pw] --> Foveated Merger      |
              --> frozen 2x2 visual merger --> compressed [N/9] |
          Foveated Selector(native, compressed, question) --> 9 cell scores
          mixed-resolution sequence: argmax cell native, rest compressed

`FoveatedEncoding` holds one image's native and compressed tokens so every
condition, label and selector step reuses the same single encoder pass.
Re-running it per condition would also let two conditions disagree about the
very features they are compared on.
"""
import numpy as np
import torch
from PIL import Image

import config
from utils.geometry import aligned_size, block_index, coarse_cells
from models import foveated_selector as fs
from models import mixed_resolution as mr


def encode_native(mw, injector, image):
    """One frozen vision-encoder pass over an already-aligned image.

    -> (native tokens [Rh*Rw, D] row-major, patch grid [1, C, 2Rh, 2Rw], (Rh, Rw))
    `injector` must be a TokenInjector built with capture_patch=True.
    """
    visual = mw.model.model.visual
    ms = visual.spatial_merge_size
    dev = mw.model.device
    enc = mw.processor.image_processor(images=image, min_pixels=config.PROCESSOR_MIN_PIXELS,
                                       max_pixels=10 ** 9, return_tensors="pt")
    thw = enc["image_grid_thw"]
    Rh, Rw = int(thw[0][1]) // ms, int(thw[0][2]) // ms
    injector.repl = None
    with torch.no_grad():
        visual(enc["pixel_values"].to(device=dev, dtype=mw.model.dtype), grid_thw=thw.to(dev))
    native = injector.out
    if native is None or native.shape[0] != Rh * Rw:
        raise RuntimeError(f"vision encoder returned "
                           f"{None if native is None else native.shape[0]} tokens, "
                           f"grid says {Rh * Rw}")
    return native, injector.patch_grid(Rh, Rw, ms), (Rh, Rw)


def compress(mw, merger, patch_grid, native_grid):
    """Foveated Merger, then the frozen 2x2 visual merger, applied once.

    -> compressed tokens [(Rh/merge) * (Rw/merge), D], row-major over the
    compressed-unit grid. Gradients flow into `merger` when it is training.
    """
    visual = mw.model.model.visual
    ms = visual.spatial_merge_size
    merge = merger.merge
    Rh, Rw = native_grid
    if Rh % merge or Rw % merge:
        raise ValueError(f"{Rh}x{Rw} native grid is not divisible by merge={merge}")
    dev = mw.model.device
    merged = merger(patch_grid.to(dev).float())                 # [(2Rh/m)(2Rw/m), C]
    pairs = torch.as_tensor(block_index(Rh * ms // merge, Rw * ms // merge, ms)[0], device=dev)
    d_in = visual.merger.hidden_size // (ms * ms)
    return visual.merger(merged[pairs].reshape(-1, d_in).to(dtype=mw.model.dtype))


class FoveatedEncoding:
    """One image at the shared 252-aligned geometry, encoded once."""

    def __init__(self, mw, injector, merger, image, max_visual_tokens=16384):
        W, H = aligned_size(image.width, image.height, max_visual_tokens)
        self.image = image.convert("RGB").resize((W, H), Image.BICUBIC)
        self.native, patch_grid, self.grid = encode_native(mw, injector, self.image)
        with torch.no_grad():
            self.compressed = compress(mw, merger, patch_grid, self.grid)
        Rh, Rw = self.grid
        self.merge = merger.merge
        self.ms = mw.model.model.visual.spatial_merge_size
        self.patch_size = mw.model.model.visual.patch_size
        self.n_native = Rh * Rw
        self.unit_grid = (Rh // self.merge, Rw // self.merge)
        self.blocks = torch.as_tensor(block_index(Rh, Rw, self.merge)[0],
                                      device=self.native.device)
        self.cell_of_unit = coarse_cells(*self.unit_grid, grid=config.SELECTION_GRID)[0]
        self.native_entry = config.image_entry(self.image, "native")

    def cell_scores(self, selector, q_tokens=None):
        """Foveated Selector -> [9] cell scores s_c."""
        unit_logits = selector(self.native[self.blocks], self.compressed, self.unit_grid,
                               q_tokens)
        cell_of_unit = torch.as_tensor(self.cell_of_unit, device=unit_logits.device)
        return fs.cell_scores(unit_logits, cell_of_unit, config.N_CELLS)

    def uniform(self):
        """Uniform Foveated Compression: every unit compressed (K = 0, 1/9 of native)."""
        Ch, Cw = self.unit_grid
        n = Ch * Cw
        lay = mr.Layout(src=np.arange(n, dtype=np.int64), is_comp=np.ones(n, dtype=bool),
                        hw=np.stack(np.divmod(np.arange(n), Cw), 1) * self.merge,
                        n_tokens=n, grid=self.grid)
        return lay, mr.build_tokens(self.native, self.compressed, lay)

    def preserve(self, cells):
        """Mixed-resolution sequence with `cells` native. Asserts the exact Eq. 5 budget."""
        Rh, Rw = self.grid
        lay, comp_index = mr.layout_cells(Rh, Rw, cells, self.merge, config.SELECTION_GRID)
        k = len(set(int(c) for c in cells))
        expect = config.TOKEN_BUDGET.get(k)
        if expect is not None and abs(lay.n_tokens - expect * self.n_native) > 1e-6:
            raise RuntimeError(f"K={k} sequence length {lay.n_tokens} violates the exact "
                               f"{expect:.4f} budget against {self.n_native} native tokens")
        idx = torch.as_tensor(comp_index, device=self.compressed.device, dtype=torch.long)
        return lay, mr.build_tokens(self.native, self.compressed[idx], lay)

    def frame_entry(self, n_tokens):
        """Placeholder image whose image_pad count is `n_tokens` (pixels never read)."""
        fw, fh = mr.frame_size(n_tokens, self.ms, self.patch_size)
        return {"type": "image", "image": Image.new("RGB", (fw, fh)),
                "min_pixels": config.PROCESSOR_MIN_PIXELS, "max_pixels": 10 ** 9}
