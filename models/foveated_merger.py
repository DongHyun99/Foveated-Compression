#!/usr/bin/env python3
"""Foveated Merger (Sec. 3.2): 3x3 raw ViT patches -> 1 patch, before Qwen's projector.

WHERE IT SITS, AND WHY THERE.

    ViT blocks
       |   patches [P, 1280]                    row-major, un-permuted
    FoveatedMerger              3x3 x 1280 -> 1280        TRAINED
       |   [P/9, 1280]
    Qwen visual merger          2x2 x 1280 -> 3584        FROZEN, applied ONCE
       |   [P/36, 3584]   compressed tokens c_u

For each 3x3 neighbourhood (Eq. 1):

    x~ = 1/9 sum_i x_i + W_head GELU(W_fc [RMSNorm(x_1); ...; RMSNorm(x_9)] + b_fc) + b_head

with W_fc: R^{9C} -> R^2560 randomly initialised and W_head: R^2560 -> R^C
ZERO-initialised, so step 0 is exactly mean pooling and every point of movement
from there is attributable to behavioural self-distillation. 32.77M trainable
parameters. Each compressed token corresponds spatially to one 3x3 block of
native visual tokens and shares their 3584-d interface, which is what makes
in-place native-token replacement possible (Sec. 3.4).

Qwen's own visual merger outputs 3584, so nothing can be merged after it
without re-applying it. Keeping this layer at 1280 -> 1280 means the space it
reads is the space it writes, and placing it BEFORE the frozen merger is also
where the information is: per final token, 36 patches carry 36 * 1280 numbers,
while anything after the frozen merger sees only 4 * 3584 of them.

merge=2 VARIANT (not in the paper). At merge=2 the input is 4 x 1280, byte-for-
byte the frozen merger's input space, so `load_pretrained` can copy its RMSNorm
and first Linear. At merge=3 nothing can be inherited.

QuestionConditionedMerger (not in the paper) adds a depthwise context
convolution and a question-conditioned softmax over the merge slots. The
question channel is deliberately narrow; widening it takes this model class
straight to answer priors: loss fell 1.40 -> 0.86 while accuracy fell
58% -> 45% and teacher agreement collapsed 86% -> 48%.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Qwen2RMSNorm, restated so this module needs no import from modeling code."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * v).to(x.dtype)


def _blocks(grid, merge):
    """[1, C, Ph, Pw] -> [M, merge*merge, C], row-major over the merged grid.

    Slot k of row m is the patch at offset (k // merge, k % merge) inside
    merged cell m -- the (row-major cell, row-major offset) convention Qwen's
    patch embedding uses, so slot order matches what the frozen projector's
    concatenation expects. A transposed permutation here produces a perfectly
    plausible loss curve and a silently wrong model.
    """
    _b, C, Ph, Pw = grid.shape
    m = merge
    x = grid.reshape(C, Ph // m, m, Pw // m, m)      # C, a, dr, b, dc
    x = x.permute(1, 3, 2, 4, 0)                     # a, b, dr, dc, C
    return x.reshape(-1, m * m, C)                   # (a*b), (dr*dc), C


class FoveatedMerger(nn.Module):
    """[1, dim, Ph, Pw] patch grid -> [Ph*Pw/merge^2, dim]. Question-blind.

    Same space in and out. The deployed 9:1 merger applies merge=3 once and
    then the frozen visual merger once, which is a different (and measurably
    better) object than applying a merge=2 layer twice.
    """

    def __init__(self, dim=1280, merge=3, hidden=2560):
        super().__init__()
        self.dim, self.merge = dim, merge
        self.smu = merge * merge
        self.hidden_in = dim * self.smu              # 5120 at merge=2, 11520 at merge=3
        # hidden=None ties the width to the input (required for merge=2 to copy
        # fc1 from the frozen merger). merge=3 inherits nothing, so the paper's
        # 2560 bottleneck lands the module at 32.77M instead of a 132.7M fc1.
        self.hidden = self.hidden_in if hidden is None else int(hidden)

        self.ln = RMSNorm(dim)                              # RMSNorm
        self.fc1 = nn.Linear(self.hidden_in, self.hidden)   # W_fc, b_fc
        self.act = nn.GELU()
        self.head = nn.Linear(self.hidden, dim)             # W_head, b_head (zero-init)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.level = nn.Parameter(torch.zeros(dim))         # recursion tag

    def load_pretrained(self, projector):
        """Copy the frozen projector's norm and first Linear.

        Shapes are asserted rather than reshaped: a silent mismatch here looks
        like a bad initialisation, not like a bug.
        """
        if projector.ln_q.weight.shape != self.ln.weight.shape:
            raise ValueError(f"ln_q {tuple(projector.ln_q.weight.shape)} vs "
                             f"{tuple(self.ln.weight.shape)}")
        if projector.mlp[0].weight.shape != self.fc1.weight.shape:
            raise ValueError(
                f"cannot inherit the projector: mlp[0] is "
                f"{tuple(projector.mlp[0].weight.shape)} but fc1 is "
                f"{tuple(self.fc1.weight.shape)}. Only merge=2 at the default "
                f"hidden width matches; merge=3 must be trained from scratch.")
        with torch.no_grad():
            self.ln.weight.copy_(projector.ln_q.weight.float())
            self.fc1.weight.copy_(projector.mlp[0].weight.float())
            self.fc1.bias.copy_(projector.mlp[0].bias.float())
        return self

    def blocks(self, grid):
        return _blocks(grid, self.merge)

    def forward(self, grid, q_emb=None, level=0, return_weights=False):
        """q_emb is accepted and IGNORED -- this architecture is question-blind."""
        if level:
            grid = grid + level * self.level.view(1, -1, 1, 1)
        x = self.blocks(grid)                                     # [M, smu, C]
        out = x.mean(1) + self.head(self.act(self.fc1(
            self.ln(x).reshape(-1, self.hidden_in))))
        if not return_weights:
            return out
        # One column per architecture in the training log; with no pooling term
        # every patch carries 1/smu by construction.
        return out, out.new_full((x.shape[0], self.smu), 1.0 / self.smu)


class QuestionConditionedMerger(nn.Module):
    """[1, dim, Ph, Pw] (+ question) -> [Ph*Pw/merge^2, dim]. Three additive terms.

      ctx    depthwise 3x3 over the patch grid. Context, not compression: the
             resolution is unchanged. Without it the merge sees only its own
             patches and cannot tell a text line from a flat region. Widening
             the MLP's input to a 3x3 block neighbourhood instead would make
             its first Linear 236M; this is 11.5K for the same receptive field.
      pool   question-conditioned softmax over the merge slots.
      mlp    the merge itself, mirroring the projector's shape.

    ctx and the MLP's output projection are zeroed, so step 0 is mean pooling
    over the block. The slot projection is initialised SMALL rather than zero:
    zeroing it too makes the softmax uniform regardless of the question, which
    starts the entire question path at exactly zero gradient -- measured, the
    slot weights then never moved and the question projection's gradient was
    five orders of magnitude below the slot head's.
    """

    def __init__(self, dim=1280, merge=3, hidden=None, dq=32, d_q_in=3584,
                 use_question=True, slot_norm=True, tau_init=1.0):
        super().__init__()
        self.dim, self.merge = dim, merge
        self.smu = merge * merge
        self.hidden_in = dim * self.smu
        h = hidden or self.hidden_in
        self.use_question = use_question

        # (1) context -- residual, zero-init, resolution preserved
        self.ctx = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        nn.init.zeros_(self.ctx.weight)
        self.gc = nn.Parameter(torch.ones(1))

        # (2) the merge -- residual over the pooled slots, zero-init output
        self.ln = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_in, h), nn.GELU(),
                                 nn.Linear(h, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.g = nn.Parameter(torch.ones(1))

        # (3) question -> narrow code -> slot weights
        self.q_proj = nn.Sequential(nn.Linear(d_q_in, dq), nn.LayerNorm(dq))
        self.w_h = nn.Linear(dim, 64, bias=False)
        self.w_q = nn.Linear(dq, 64, bias=False)
        self.slot = nn.Linear(64, self.smu)
        nn.init.normal_(self.slot.weight, std=0.05)
        nn.init.zeros_(self.slot.bias)
        # The slot logits need normalising before the temperature can mean
        # anything. On real-scale inputs the raw logits have std 0.006, so any
        # usable temperature would be ~256, at which point its own gradient is
        # not usable. A LayerNorm across the slots (no affine, so it cannot
        # undo itself) takes the logit std to 0.84 and hands tau a real range.
        self.slot_norm = slot_norm
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        # The question-blind control arm reads this instead of the question
        # code, so it stays slot-adaptive while being question-blind. Not zero:
        # w_q has no bias, so w_q(0) would collapse every slot logit.
        self.q_const = nn.Parameter(torch.randn(dq))
        self.level = nn.Parameter(torch.zeros(dim))

    def blocks(self, grid):
        return _blocks(grid, self.merge)

    def slot_weights(self, x, q_emb=None):
        """-> [M, smu], a softmax over the patches of each merged block."""
        z = (self.q_proj(q_emb) if (self.use_question and q_emb is not None)
             else self.q_const)
        a = F.normalize(self.w_h(x.mean(1)), dim=-1)
        b = F.normalize(self.w_q(z), dim=-1)
        lg = self.slot(a * b)
        if self.slot_norm:
            lg = F.layer_norm(lg, (self.smu,))
        return F.softmax(self.tau * lg, dim=-1)

    def forward(self, grid, q_emb=None, level=0, return_weights=False):
        grid = grid + self.gc * self.ctx(grid)                       # (1)
        if level:
            grid = grid + level * self.level.view(1, -1, 1, 1)
        x = self.blocks(grid)                                         # [M, smu, C]
        w = self.slot_weights(x, q_emb)                              # (3)
        out = (x * w.unsqueeze(-1)).sum(1)
        out = out + self.g * self.mlp(self.ln(x).reshape(-1, self.hidden_in))  # (2)
        return (out, w) if return_weights else out


# --------------------------------------------------------------------------
# checkpoint plumbing -- one tagged format for both architectures
# --------------------------------------------------------------------------

_ARCH = {"foveated_merger": FoveatedMerger,
         "question_conditioned_merger": QuestionConditionedMerger}


def build(arch="foveated_merger", dim=1280, merge=3, hidden=2560, projector=None,
          use_question=False, dq=32, d_q_in=3584, slot_norm=True, tau_init=1.0):
    """Fresh merger. `projector` (the live frozen visual merger) is copied in
    only for the merge=2 FoveatedMerger variant; see `load_pretrained`."""
    if arch == "foveated_merger":
        if use_question:
            raise ValueError("FoveatedMerger has no question channel -- use "
                             "arch='question_conditioned_merger'.")
        f = FoveatedMerger(dim, merge=merge, hidden=hidden)
        if projector is not None:
            f.load_pretrained(projector)
        return f
    if arch == "question_conditioned_merger":
        return QuestionConditionedMerger(dim, merge=merge, hidden=hidden, dq=dq,
                                         d_q_in=d_q_in, use_question=use_question,
                                         slot_norm=slot_norm, tau_init=tau_init)
    raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(_ARCH)}")


def from_state(st):
    """Rebuild a merger from the dict `state` produced."""
    arch = st["arch"]
    if arch == "foveated_merger":
        f = FoveatedMerger(st["dim"], merge=st["merge"], hidden=st["hidden"])
    elif arch == "question_conditioned_merger":
        f = QuestionConditionedMerger(st["dim"], merge=st["merge"], hidden=st["hidden"],
                                      dq=st["dq"], use_question=st["use_question"],
                                      slot_norm=st["slot_norm"], tau_init=st["tau"])
    else:
        raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(_ARCH)}")
    f.load_state_dict(st["state_dict"])
    return f


def load(path):
    """The merger stored under "merger" in a training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "merger" not in payload:
        raise ValueError(f"{path} has no 'merger' entry")
    return from_state(payload["merger"])


def state(f):
    """The dict `from_state` reads back, tagged with its architecture."""
    sd = {k: v.cpu() for k, v in f.state_dict().items()}
    if isinstance(f, FoveatedMerger):
        return {"arch": "foveated_merger", "dim": f.dim, "merge": f.merge,
                "hidden": f.hidden, "state_dict": sd}
    return {"arch": "question_conditioned_merger", "dim": f.dim, "merge": f.merge,
            "hidden": f.mlp[0].out_features, "dq": f.q_const.numel(),
            "use_question": f.use_question, "slot_norm": f.slot_norm,
            "tau": float(f.tau.detach()), "state_dict": sd}
