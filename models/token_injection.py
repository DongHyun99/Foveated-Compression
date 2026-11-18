#!/usr/bin/env python3
"""Read the vision tower's merged tokens, and put different ones back.

This is the seam the whole method hangs on. The Foveated Merger has to see
the tower's real patch features, and the LLM has to receive a mixed-resolution
sequence in place of the native tokens, without either side knowing anything
happened.

ORDERING, which is easy to get backwards and fails silently. Qwen2.5-VL's
vision tower permutes tokens into window order before its blocks and restores
row-major itself, right after the projector:

    hidden_states = self.merger(hidden_states)
    hidden_states = hidden_states[torch.argsort(window_index), :]   # returned

So the two capture points are in DIFFERENT orders:

  merger forward pre-hook  -> WINDOW order    (needs un-permuting)
  tower forward hook       -> ROW-MAJOR       (already correct, both directions)

Treating the tower's output as window order -- un-permuting the capture, or
handing back `repl[window_index]` -- makes the block gather in
`utils.geometry.block_index` pick spatially unrelated tokens. Nothing raises. It costs
1.8-3.8pp of accuracy, and it is the reason an earlier generation of this
repository's ViT-derived per-cell features carried no signal at all while the
pixel-space ones did.

Gradients flow through `repl`, so the same hook serves training and evaluation.
Freeze the backbone and the tower builds no graph of its own.
"""
import torch


def unpermute_vit_seq(x, window_index, spatial_merge_unit):
    """Undo the ViT window permutation on a [seq, D] tensor captured INSIDE the tower.

    Handles both capture points, which have different lengths:

      projector OUTPUT   one row per merged token, so seq == len(window_index)
                         and the permutation applies directly -- literally what
                         the model itself does.
      block output       one row per RAW patch, so
                         seq == len(window_index) * spatial_merge_unit and the
                         permutation applies to GROUPS of spatial_merge_unit
                         rows, matching the model's own
                         reshape(seq // smu, smu, -1)[window_index].
    """
    seq, d = x.shape
    n_groups = int(window_index.numel())
    rev = torch.argsort(window_index)
    if seq == n_groups:
        return x[rev]
    if seq == n_groups * spatial_merge_unit:
        return x.reshape(n_groups, spatial_merge_unit, d)[rev].reshape(seq, d)
    raise ValueError(f"cannot un-permute {seq} rows against {n_groups} windows "
                     f"at merge unit {spatial_merge_unit}")


class TokenInjector:
    """Hooks one Qwen2.5-VL vision tower.

    After a forward pass:
      out    [n_merged, D]        merged tokens, ROW-MAJOR (the tower's return)
      patch  [n_merged, smu, D']  the projector's INPUT, un-permuted to row-major
                                  (only when capture_patch=True)
      thw                         grid_thw of the last call

    Set `repl` (row-major, same length as `out`) to replace the tower's output
    on the next forward. It is NOT cleared automatically: a caller that
    generates must set it before and clear it after, because generation runs
    the tower exactly once, at prefill.
    """

    def __init__(self, visual, capture_patch=False):
        self.visual = visual
        self.capture_patch = capture_patch
        self.repl = None
        self.out = None
        self.patch = None
        self.thw = None
        self._raw_in = None
        visual.register_forward_pre_hook(self._pre, with_kwargs=True)
        if capture_patch:
            visual.merger.register_forward_pre_hook(self._grab)
        visual.register_forward_hook(self._post)

    def _pre(self, m, a, kw):
        self.thw = kw["grid_thw"] if "grid_thw" in kw else a[1]
        # A forward that never reaches the projector (MixedFrame short-circuits
        # the tower) must not pair this call's grid with a stale capture.
        self._raw_in = None
        self.patch = None

    def _grab(self, m, inp):
        self._raw_in = inp[0].detach()

    def _post(self, m, i, out):
        if self.capture_patch and self._raw_in is not None:
            smu = self.visual.spatial_merge_size ** 2
            win, _ = self.visual.get_window_index(self.thw.cpu())
            p = unpermute_vit_seq(self._raw_in.float().cpu(), win.cpu(), smu)
            self.patch = p.reshape(-1, smu, p.shape[-1])
        if self.repl is None:
            self.out = out.detach()
            return out
        return self.repl.to(device=out.device, dtype=out.dtype)

    def merged_grid(self, ms=2):
        """(Rh, Rw) of the last forward's merged-token grid."""
        return int(self.thw[0][1]) // ms, int(self.thw[0][2]) // ms

    def patch_grid(self, Rh, Rw, ms=2):
        """Captured projector input -> [1, C, Rh*ms, Rw*ms] patch grid.

        The Foveated Merger is a 2-D module over raw patches, so the captured
        [n_merged, smu, C] has to be unfolded back into the spatial layout the
        patches came from. Getting this permutation wrong is the same silent
        failure as getting the window order wrong, one level down.
        """
        if self.patch is None:
            raise RuntimeError("TokenInjector was built with capture_patch=False")
        pt = self.patch.float()
        return (pt.reshape(Rh, Rw, ms, ms, -1)
                  .permute(4, 0, 2, 1, 3)
                  .reshape(1, -1, Rh * ms, Rw * ms)
                  .contiguous())


@torch.inference_mode()
def encode(mw, img, image_processor):
    """One tower pass with no substitution.

    -> (projector_input [n_merged, smu, D] row-major,
        projector_output [n_merged, Dout] row-major,
        (merged_h, merged_w))

    Both capture points come off the PROJECTOR, which emits window order, so
    both are un-permuted here. Reading the tower's return instead would already
    be row-major -- see the module docstring.
    """
    visual = mw.model.model.visual
    enc = image_processor(images=img, return_tensors="pt")
    pv = enc["pixel_values"].to(device=mw.model.device, dtype=mw.model.dtype)
    thw = enc["image_grid_thw"].to(mw.model.device)
    grabbed = {}
    h1 = visual.merger.register_forward_pre_hook(
        lambda m, i: grabbed.__setitem__("in", i[0].detach()))
    h2 = visual.merger.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("out", o.detach()))
    try:
        visual(pv, grid_thw=thw)
    finally:
        h1.remove()
        h2.remove()
    ms = visual.spatial_merge_size
    smu = ms * ms
    win, _ = visual.get_window_index(thw.cpu())
    win = win.cpu()
    mi = unpermute_vit_seq(grabbed["in"].float().cpu(), win, smu)
    mo = unpermute_vit_seq(grabbed["out"].float().cpu(), win, smu)
    gh, gw = int(thw[0][1]), int(thw[0][2])
    return mi.reshape(-1, smu, mi.shape[-1]), mo, (gh // ms, gw // ms)
