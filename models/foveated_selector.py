#!/usr/bin/env python3
"""Foveated Selector (Sec. 3.3): which of the nine coarse cells to keep native.

A compressed unit u is one 3x3 block N(u) of native visual tokens that the
Foveated Merger collapses into a single compressed token c_u. The selector
scores every unit, pools the unit logits into nine cell scores, and the
highest-scoring cell is preserved at native resolution (K = 1).

Everything it reads is available BEFORE the LLM prefill:

    native tokens of each unit      [M, 9, D]       from the vision encoder
    compressed token of each unit   [M, D]          from the Foveated Merger
    question token embeddings       [T <= 48, D]    an embedding-table lookup

so it costs no additional language-model forward pass and no additional
high-resolution vision-encoder pass.

VISUAL DESCRIPTOR (Eq. 3), 512-d per unit:

    v_u = P_n LN(h_u) + P_c LN(c_u) + P_d (LN(h_u) - LN(c_u)) + P_xy(p_u)

with h_u the mean of the unit's native tokens and p_u its normalised 2-D
coordinate. The four terms encode native content, compressed content,
compression-induced discrepancy, and location.

QUESTION PATH. Question tokens are projected to 512-d and encoded by one
self-attention/FFN block; each visual descriptor then attends to them through
eight-head cross-attention, with visual units as queries. A depthwise 3x3
convolution adds local spatial context and a scoring MLP emits one logit l_u
per unit. 10.8M trainable parameters.

`use_question=False` removes the question path AND the spatial input P_xy --
the "Question and Spatial Inputs Removed" ablation of Table 2.
"""
import torch
import torch.nn as nn

import config


class FoveatedSelector(nn.Module):
    """[M, S, D] native + [M, D] compressed (+ [T, D] question) -> one logit per unit."""

    def __init__(self, d_token=config.SELECTOR["d_token"], hidden=config.SELECTOR["hidden"],
                 heads=config.SELECTOR["heads"],
                 max_question_tokens=config.SELECTOR["max_question_tokens"],
                 use_question=True):
        super().__init__()
        self.use_question = use_question

        self.norm_native = nn.LayerNorm(d_token)
        self.norm_compressed = nn.LayerNorm(d_token)
        self.proj_native = nn.Linear(d_token, hidden)        # P_n
        self.proj_compressed = nn.Linear(d_token, hidden)    # P_c
        # The DIFFERENCE term is a direct estimate of what compressing this
        # unit would destroy -- the quantity the selection is about. The two
        # absolute terms only say what is there, not what would be lost.
        self.proj_diff = nn.Linear(d_token, hidden)          # P_d

        if use_question:
            self.proj_xy = nn.Linear(2, hidden)              # P_xy
            self.norm_q = nn.LayerNorm(d_token)
            self.proj_q = nn.Linear(d_token, hidden)
            self.q_pos = nn.Parameter(torch.zeros(max_question_tokens, hidden))
            self.q_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.q_norm = nn.LayerNorm(hidden)
            self.q_ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                      nn.Linear(hidden * 2, hidden))
            self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden)

        self.ctx = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        # Zero read-out: step 0 scores every cell equally, so the tie-aware
        # objective starts from a uniform distribution over the nine cells.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, native, compressed, unit_grid, q_tokens=None):
        """Score compressed units.

        native      [M, S, D]  native tokens N(u) per unit (S = 9 at merge=3)
        compressed  [M, D]     compressed tokens c_u
        unit_grid   (Gh, Gw)   M = Gh*Gw, row-major
        q_tokens    [T, D]     question token embeddings, or None
        """
        h = native.float().mean(1)
        c = compressed.float()
        hn, cn = self.norm_native(h), self.norm_compressed(c)
        x = self.proj_native(hn) + self.proj_compressed(cn) + self.proj_diff(hn - cn)
        Gh, Gw = unit_grid

        if self.use_question:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, Gh, device=x.device, dtype=x.dtype),
                torch.linspace(-1, 1, Gw, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            x = x + self.proj_xy(torch.stack((xx, yy), -1).reshape(-1, 2))
            if q_tokens is not None:
                if q_tokens.ndim != 2:
                    raise ValueError("q_tokens must be [tokens, d_token]")
                q = self.proj_q(self.norm_q(q_tokens.float()))
                q = q + self.q_pos[:q.shape[0]]
                q = self.q_norm(q + self.q_self(q[None], q[None], q[None],
                                                need_weights=False)[0][0])
                q = self.q_norm(q + self.q_ff(q))
                x = self.cross_norm(x + self.cross(x[None], q[None], q[None],
                                                   need_weights=False)[0][0])

        x = x + self.ctx(x.t().reshape(1, -1, Gh, Gw)).reshape(-1, Gh * Gw).t()
        return self.mlp(x).squeeze(-1)


def cell_scores(unit_logits, cell_of_unit, n_cells=config.N_CELLS):
    """s_c = mean of l_u over the units U_c of cell c.

    Whole-cell selection matches the intervention granularity and keeps every
    candidate action at identical token cost. Mean, not sum, so a cell is never
    preferred merely for containing more units.
    """
    if unit_logits.shape[0] != cell_of_unit.shape[0]:
        raise ValueError(f"{unit_logits.shape[0]} unit logits against "
                         f"{cell_of_unit.shape[0]} unit->cell entries")
    out = unit_logits.new_zeros(n_cells)
    for c in range(n_cells):
        m = cell_of_unit == c
        if not bool(m.any()):
            raise ValueError(f"cell {c} has no units; grid is too small")
        out[c] = unit_logits[m].mean()
    return out


def state(selector):
    return {"arch": "foveated_selector", "use_question": selector.use_question,
            "hidden": selector.proj_native.out_features,
            "state_dict": {k: v.cpu() for k, v in selector.state_dict().items()}}


def from_state(st):
    selector = FoveatedSelector(hidden=st["hidden"], use_question=st["use_question"])
    selector.load_state_dict(st["state_dict"])
    return selector


def load(path):
    """The selector stored under "selector" in a selector checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "selector" not in payload:
        raise ValueError(f"{path} has no 'selector' entry")
    return from_state(payload["selector"])
