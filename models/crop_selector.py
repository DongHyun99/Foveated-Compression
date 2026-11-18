#!/usr/bin/env python3
"""Crop selector for pixel-space crop foveation -- NOT part of the paper.

A separate, cache-based line of the same idea as the Foveated Selector. Where
`foveated_selector` picks one coarse cell inside a single mixed-resolution
sequence, this head scores the ten actions of a base+crop input: a 0.5x global
view plus one of nine 3x3 crops re-encoded at high resolution, or no crop at
all. This is the "pixel-space selective magnification" route the paper
contrasts against (Sec. 1, 2.2). It is trained offline on a feature cache, which
is what makes the full ablation ladder and the multi-seed protocol affordable
-- a VLM forward pass per configuration would not be.

THE ACTION SPACE AND WHY NO-CROP IS IN IT. A foveated input costs
0.5**2 + (1/3)**2 = 36% of native's tokens. Dropping the crop costs 25%. The
no-crop action is not a fallback; it is the correct answer whenever a crop
would mislead, and there are measurably many such images -- the selector has to
be able to abstain or it will hallucinate relevance.

THE LOSS. `expected_reward_loss` is -sum_i softmax(score_i / tau) * reward_i
over the ten actions. Annealing tau from 1.0 to 0.05 turns a soft expectation
into a near-argmax, which is what the deployed policy actually does.

The loss's gradient is EXACTLY ZERO when all ten rewards are equal, and that
fact defines what data is worth training on. See `CATEGORIES` below: images
where every action succeeds, or every action fails, teach the cell choice
nothing. Only the categories where the choice changes the outcome carry
signal, and only those are kept.

THE GAIN AUXILIARY. Reward is binary and therefore coarse. The measured
per-region gain,

    gain(R) = log p(answer | base + R) - log p(answer | base)

is continuous and ranks regions even when several of them flip the answer the
same way. It enters as a listwise cross-entropy against a temperature-softmaxed
z-scored gain target, with a weight that decays to zero over the first half of
training: it is a shaping term for the early ranking, not the objective.

THE SUFFICIENCY/SELECTION SPLIT. Splitting the ten-way head into a lightweight
BaseSufficiencyHead ("is the base view alone already correct?", from base
features only) and a nine-way selection head (invoked only when a crop is
needed) is motivated by exactly the zero-gradient fact above: the tied-reward
categories carry no gradient for the CELL choice, but they carry a perfectly
well-posed binary classification gradient for the sufficiency target. Splitting recovers training
signal that the joint model structurally cannot use.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CROP, N_ACTION, NOCROP = 9, 10, 9

D_CELL = 64      # PCA dim for projected sub-cell vectors
D_LAYER = 32     # PCA dim per intermediate ViT layer sub-cell vector
D_Q = 64         # PCA dim for question hiddens (last-token and per-token alike)
MAX_Q_TOK = 24   # question tokens kept; covers the vast majority of questions

# The ablation ladder. name -> (q_tokens, sub_resolved, constant_question,
# use_midlayers). Each rung removes exactly one thing, so a difference between
# two rungs is attributable to one variable.
STAGES = {
    "control": (False, False, True, True),    # question blanked entirely
    "a":       (False, False, False, True),   # one pooled question vector
    "b":       (False, True, False, True),    # sub-cell-resolved candidates
    "ab":      (True, True, False, True),     # + per-token question
    "ab_nomid": (True, True, False, False),   # ab, intermediate ViT layers zeroed
}


# --------------------------------------------------------------------------
# the six sample categories -- defined once so every consumer agrees
# --------------------------------------------------------------------------
# Cut on three measured facts per image:
#   L  the downscaled base view alone is correct
#   K  how many of the nine base+crop pairs are correct
#   H  native full resolution is correct
#
# Exhaustive and disjoint: L=1 splits into 1|2 on K=9; L=0 splits into 5 (K=9),
# 6 (K=1..8), and 3|4 (K=0, split on H).
#
# K=0 and K=9 are the two ends where the choice of crop cannot matter --
# nothing works, or everything works. K=1..8 is the only band where picking
# differently changes the answer, which is the only place a selector has a
# reason to exist.
#
# H separates 3 from 4 because they demand different responses even though a
# selector can do nothing about either: 3 is "the pixels were there and the
# tiling lost them", 4 is "the model cannot answer this at any resolution".
# H is NOT one of the selector's actions, so it never enters 1, 2, 5 or 6.
CATEGORIES = [
    ("1", lambda L, K, H: L & (K == 9), "every crop safe", False,
     "base correct and all nine correct -- any action is equivalent, nothing to learn"),
    ("2", lambda L, K, H: L & (K <= 8), "a harmful crop exists", True,
     "base correct but some crop breaks it -- the only evidence for no-crop"),
    ("3", lambda L, K, H: (~L) & (K == 0) & H, "native only", False,
     "the pixels are there but the 3x3 tiling loses them -- outside the action space"),
    ("4", lambda L, K, H: (~L) & (K == 0) & (~H), "impossible", False,
     "wrong at every resolution -- a limit of the model, not of the selector"),
    ("5", lambda L, K, H: (~L) & (K == 9), "any crop works", True,
     "only resolution was needed, not location -- easy to pick, impossible to miss"),
    ("6", lambda L, K, H: (~L) & (K >= 1) & (K <= 8), "one specific crop works", True,
     "the choice decides the outcome -- the selector's reason to exist"),
]
KEEP_CATEGORIES = {k for k, _, _, keep, _ in CATEGORIES if keep}


def assign_categories(L, K, H):
    """-> (per-category boolean masks, keep mask). Verifies the partition.

    The partition check is not decoration: these predicates are the single
    place the training population is defined, and an overlap or a gap would
    silently change what every downstream number is computed over.
    """
    masks = {k: pred(L, K, H) for k, pred, _, _, _ in CATEGORIES}
    stacked = np.stack([masks[k] for k, *_ in CATEGORIES])
    counts = stacked.sum(0)
    if not (counts == 1).all():
        bad = int((counts != 1).sum())
        raise AssertionError(f"categories are not a partition: {bad} images land in "
                             f"{sorted(set(counts.tolist()))} categories")
    keep = np.zeros(len(L), bool)
    for k in KEEP_CATEGORIES:
        keep |= masks[k]
    return masks, keep


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class CropScorer(nn.Module):
    """Per-candidate token set -> question cross-attention -> Set-Transformer
    over the ten actions -> one score head.

    `q_tokens` and `sub_resolved` are the two switches that define the ablation
    ladder; `constant_question` blanks the question entirely, which is the
    control that measures whether the question matters at all.
    """

    def __init__(self, n_layers_vit, n_sub, d_detail, d_model=64, n_blocks=2,
                 n_heads=4, dropout=0.2, q_tokens=True, sub_resolved=True,
                 constant_question=False, use_midlayers=True, drop_nocrop=False):
        super().__init__()
        self.q_tokens = q_tokens
        self.sub_resolved = sub_resolved
        self.constant_question = constant_question
        self.use_midlayers = use_midlayers
        # drop_nocrop removes the tenth action from the STRUCTURE. Removing it
        # from the loss alone leaves an unsupervised no-crop token in the
        # encoder, splitting attention with the nine real candidates; the two
        # together give a pure ranking model.
        self.drop_nocrop = drop_nocrop
        self.n_sub, self.n_layers_vit = n_sub, n_layers_vit

        d_tok = D_CELL + n_layers_vit * D_LAYER
        self.tok_ln = nn.LayerNorm(d_tok)
        self.tok_proj = nn.Linear(d_tok, d_model)
        self.sub_pos = nn.Parameter(torch.zeros(n_sub, d_model))
        nn.init.normal_(self.sub_pos, std=0.02)

        self.q_ln = nn.LayerNorm(D_Q)
        self.q_proj = nn.Linear(D_Q, d_model)
        self.xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.xattn_ln = nn.LayerNorm(d_model)

        # Question<->candidate bilinear interaction. REQUIRED, not optional.
        # Cross-attention alone cannot carry the question when sub_resolved is
        # False, because there is then exactly ONE key per candidate and a
        # softmax over one element is 1.0 regardless of the query -- which made
        # rung "a" mathematically identical to the question-free control
        # (verified: maximum output difference 0.0). This path gives the
        # question an effect at every rung, so the ladder measures sub-cell
        # selectivity on TOP of question conditioning rather than instead of it.
        self.inter_heads, self.inter_dim = 8, 8
        self.q_inter = nn.Linear(d_model, self.inter_heads * self.inter_dim, bias=False)
        self.c_inter = nn.Linear(d_model, self.inter_heads * self.inter_dim, bias=False)
        self.inter_ln = nn.LayerNorm(self.inter_heads)

        self.detail_ln = nn.LayerNorm(d_detail)
        self.detail_proj = nn.Linear(d_detail, d_model // 2)
        self.sig_ln = nn.LayerNorm(3)
        self.sig_proj = nn.Linear(3, d_model // 4)
        fuse_in = d_model + d_model // 2 + d_model // 4 + self.inter_heads
        self.fuse = nn.Linear(fuse_in, d_model)

        self.grid_pos = nn.Parameter(torch.zeros(N_CROP, d_model))
        nn.init.normal_(self.grid_pos, std=0.02)
        self.nocrop_tok = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.nocrop_tok, std=0.02)
        self.nocrop_mix = nn.Linear(d_model, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_model * 4, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.score_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Linear(d_model // 2, 1))

    def forward(self, cell, layers, detail, signals, q, qtok, qmask, q_const=None):
        """cell [B,9,S,D_CELL]; layers [B,9,L,S,D_LAYER]; detail [B,9,Dd];
        signals [B,9,3]; q [B,D_Q]; qtok [B,T,D_Q]; qmask [B,T]."""
        b = cell.shape[0]
        lay = layers.permute(0, 1, 3, 2, 4).reshape(b, N_CROP, self.n_sub, -1)
        if not self.use_midlayers:
            # Parameter count unchanged, information removed -- the honest form
            # of this ablation, since a narrower model would confound capacity
            # with the thing being ablated.
            lay = torch.zeros_like(lay)
        tok = torch.cat([cell, lay], dim=-1)
        tok = self.tok_proj(self.tok_ln(tok)) + self.sub_pos
        if not self.sub_resolved:
            tok = tok.mean(dim=2, keepdim=True)

        if self.constant_question:
            qv = (q_const.unsqueeze(0).expand(b, -1)
                  if (q_const is not None and q_const.dim() == 1) else torch.zeros_like(q))
            qseq = self.q_proj(self.q_ln(qv)).unsqueeze(1)
        elif self.q_tokens:
            qseq = self.q_proj(self.q_ln(qtok))
        else:
            qseq = self.q_proj(self.q_ln(q)).unsqueeze(1)

        s = tok.shape[2]
        # the question attends over each candidate's token set, independently
        keys = tok.reshape(b * N_CROP, s, -1)
        qrep = (qseq.unsqueeze(1).expand(b, N_CROP, qseq.shape[1], qseq.shape[2])
                    .reshape(b * N_CROP, qseq.shape[1], -1))
        att, _ = self.xattn(qrep, keys, keys, need_weights=False)
        if self.q_tokens and not self.constant_question:
            m = qmask.unsqueeze(1).expand(b, N_CROP, -1).reshape(b * N_CROP, -1, 1)
            att = (att * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            att = att.mean(1)
        att = self.xattn_ln(att).reshape(b, N_CROP, -1)

        q_sum = ((qseq * qmask.unsqueeze(-1)).sum(1) / qmask.sum(1, keepdim=True).clamp(min=1e-6)
                 if (self.q_tokens and not self.constant_question) else qseq.mean(1))
        qi = self.q_inter(q_sum).view(b, 1, self.inter_heads, self.inter_dim)
        ci = self.c_inter(tok.mean(dim=2)).view(b, N_CROP, self.inter_heads, self.inter_dim)
        inter = self.inter_ln((qi * ci).sum(-1) / (self.inter_dim ** 0.5))

        h = self.fuse(torch.cat([att,
                                 self.detail_proj(self.detail_ln(detail)),
                                 self.sig_proj(self.sig_ln(signals)),
                                 inter], dim=-1)) + self.grid_pos
        if self.drop_nocrop:
            # Callers expect [B, 10]; keep the slot and pin it far below any
            # reachable score rather than changing every downstream shape.
            out = self.score_head(self.encoder(h)).squeeze(-1)
            pad = torch.full((b, 1), -30.0, device=out.device, dtype=out.dtype)
            return torch.cat([out, pad], dim=1)
        h_nc = self.nocrop_tok.unsqueeze(0) + self.nocrop_mix(h.mean(dim=1))
        h = torch.cat([h, h_nc.unsqueeze(1)], dim=1)
        return self.score_head(self.encoder(h)).squeeze(-1)


class BaseSufficiencyHead(nn.Module):
    """Is the downscaled base view alone already correct?

    Reads a genuine whole-image + question embedding from its own forward pass,
    NOT a pooled summary of the nine candidate features. The pooled proxy was
    measured 2.4pp worse: mean-pooling nine crops describes what is in the
    image, while this head's question is about what the model can do with the
    image at low resolution, and those are different quantities.
    """

    def __init__(self, d_in, d_model=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class KHead(nn.Module):
    """How many of the ranked crops to show: K in {1, 2, 3}.

    Reads the Selection head's score distribution rather than the raw features:
    the question "is one crop enough" is answered by how PEAKED the ranking is,
    not by what is in any single cell.

    The multi-cell reward is approximated as the max of the top-K single-cell
    rewards. That approximation is conservative and was validated by actually
    generating the multi-cell combinations -- it slightly overstates accuracy,
    which is why any headline number from this head is re-measured with real
    generation rather than reported from the approximation.
    """

    def __init__(self, n_actions=N_CROP, d_model=64, k_max=3):
        super().__init__()
        self.k_max = k_max
        self.net = nn.Sequential(
            nn.LayerNorm(n_actions), nn.Linear(n_actions, d_model), nn.GELU(),
            nn.Linear(d_model, k_max))

    def forward(self, cell_scores):
        return self.net(torch.sort(cell_scores, dim=-1, descending=True).values)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def expected_reward_loss(scores, r, tau):
    """-E_{a ~ softmax(scores/tau)} [reward(a)]. Zero gradient on tied rewards."""
    return -(F.softmax(scores / tau, dim=1) * r).sum(dim=1).mean()


def cost_penalized_reward_loss(scores, r, cost, tau, lam):
    """Expected reward minus lam * expected token cost.

    The only thing that makes a larger action space meaningful. Without the
    cost term a joint grid-size + cell head always chooses the finest grid and
    the most crops, because more pixels never hurt the reward.
    """
    p = F.softmax(scores / tau, dim=1)
    return -(p * (r - lam * cost)).sum(dim=1).mean()


def gain_listwise_loss(scores, gains, mask, temperature):
    """Listwise CE on rows with measured, finite gain labels.

    Invalid rows must be removed BEFORE normalisation. Multiplying their
    eventual NaN loss by zero is not sufficient, because NaN * 0 == NaN.
    """
    if temperature <= 0:
        raise ValueError(f"gain temperature must be positive, got {temperature}")
    valid = mask > 0
    if not valid.any():
        return scores.sum() * 0.0
    selected = gains[valid]
    if not torch.isfinite(selected).all():
        raise ValueError("gain_mask marks rows with non-finite gain labels as valid")
    g = (selected - selected.mean(1, keepdim=True)) / (selected.std(1, keepdim=True) + 1e-8)
    tgt = F.softmax(g / temperature, dim=1)
    per_row = -(tgt * F.log_softmax(scores[valid, :N_CROP], dim=1)).sum(1)
    weight = mask[valid].to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum()


def gain_label_mask(gains):
    """Rows that carry a finite, non-tied gain supervision signal."""
    return np.isfinite(gains).all(axis=1) & (np.ptp(gains, axis=1) > 0)


def require_complete_gain_labels(d, context="gain auxiliary"):
    mask = d["gain_mask"] > 0
    finite = torch.isfinite(d["gains"]).all(1)
    covered, total = int((mask & finite).sum().item()), int(mask.numel())
    if covered != total:
        raise ValueError(f"{context} requires finite gains for every row; "
                         f"coverage is {covered}/{total}")


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------
DEFAULT_CFG = dict(d_model=64, n_blocks=2, dropout=0.2, lr=1e-3, weight_decay=1e-3,
                   epochs=250, batch_size=256, patience=60, eval_every=5,
                   tau_start=1.0, tau_end=0.05, lam_aux=0.3, aux_decay_frac=0.5,
                   gain_temperature=1.0)


def evaluate(model, d, q_const, task_of, tie_eps=1e-6):
    """Policy value under the model's argmax action, plus the bars it must beat.

    `informative` is the subset where the actions are NOT all tied. It is the
    honest headline: overall accuracy is dominated by images where every action
    gives the same result, so a selector that has learned nothing still scores
    well on it.
    """
    model.eval()
    with torch.no_grad():
        s = model(d["cell"], d["layers"], d["detail"], d["signals"],
                  d["q"], d["qtok"], d["qmask"], q_const)
        pick = s.argmax(1)
        got = d["r"].gather(1, pick.unsqueeze(1)).squeeze(1).cpu().numpy()
    r = d["r"].cpu().numpy()
    inf = (r.max(1) - r.min(1)) > tie_eps
    pk = pick.cpu().numpy()
    tasks = sorted({task_of(n) for n in d["names"]})
    return {
        "overall": float(got.mean()),
        "informative": float(got[inf].mean()) if inf.any() else float("nan"),
        "n_informative": int(inf.sum()),
        "abstain_rate": float((pk == NOCROP).mean()),
        "oracle": float(r.max(1).mean()),
        "by_task": {t: float(got[[i for i, n in enumerate(d["names"]) if task_of(n) == t]].mean())
                    for t in tasks},
    }


def train_one(cfg, tr, va, seed, device, stage, task_of, verbose=False):
    """One (rung, seed) unit. Returns the best-validation model.

    tau is annealed on a cosine from tau_start to tau_end, so the objective
    starts as a soft expectation over all ten actions and ends close to the
    argmax the deployed policy takes. The gain auxiliary decays to zero over
    the first `aux_decay_frac` of training for the same reason: it shapes the
    early ranking and must not compete with the real objective at the end.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.get("lam_aux", 0.0) > 0:
        require_complete_gain_labels(tr)

    q_tok, sub_res, const_q, use_mid = STAGES[stage]
    model = CropScorer(n_layers_vit=tr["layers"].shape[2], n_sub=tr["cell"].shape[2],
                    d_detail=tr["detail"].shape[-1], d_model=cfg["d_model"],
                    n_blocks=cfg["n_blocks"], dropout=cfg["dropout"], q_tokens=q_tok,
                    sub_resolved=sub_res, constant_question=const_q,
                    use_midlayers=use_mid).to(device)
    q_const = tr["q"].mean(0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    n = tr["r"].shape[0]
    best, best_state, since = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        frac = ep / max(1, cfg["epochs"] - 1)
        tau = cfg["tau_start"] + (cfg["tau_end"] - cfg["tau_start"]) * (1 - np.cos(np.pi * frac)) / 2
        lam = (cfg["lam_aux"] * max(0.0, 1.0 - frac / cfg["aux_decay_frac"])
               if cfg["aux_decay_frac"] > 0 else 0.0)
        model.train()
        perm = torch.randperm(n, device=device)
        for st in range(0, n, cfg["batch_size"]):
            b = perm[st:st + cfg["batch_size"]]
            s = model(tr["cell"][b], tr["layers"][b], tr["detail"][b], tr["signals"][b],
                      tr["q"][b], tr["qtok"][b], tr["qmask"][b], q_const)
            loss = expected_reward_loss(s, tr["r"][b], tau)
            if lam > 0:
                loss = loss + lam * gain_listwise_loss(
                    s, tr["gains"][b], tr["gain_mask"][b], cfg["gain_temperature"])
            if not torch.isfinite(loss):
                bad = [tr["names"][int(j)] for j in b[:3].detach().cpu()]
                raise FloatingPointError(
                    f"non-finite selector loss at epoch={ep} batch_start={st}; names={bad}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % cfg["eval_every"] == 0 or ep == 0:
            m = evaluate(model, va, q_const, task_of)
            if m["overall"] > best:
                best, best_state, since = m["overall"], copy.deepcopy(model.state_dict()), 0
            else:
                since += cfg["eval_every"]
                if since >= cfg["patience"]:
                    break
            if verbose:
                print(f"    ep {ep + 1:>4} tau={tau:.2f} lam={lam:.2f} "
                      f"loss={loss.item():+.4f} val={m['overall']:.4f} "
                      f"val_inf={m['informative']:.4f} abst={m['abstain_rate']:.1%}",
                      flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, q_const


def split3(names, test_frac=0.2, val_frac_of_rest=0.15, seed=0, group_map=None):
    """Task-stratified, group-respecting split. TEST is drawn first, never touched.

    `group_map` maps a row name to an IMAGE group. Splitting by row instead
    leaks: GQA testdev_balanced is 12,578 questions over 398 images, so a
    row-level split puts nearly every test image in train as well and the
    reported number measures memorisation.
    """
    rng = np.random.default_rng(seed)
    gm = group_map or {}
    by_group = {}
    for nm in names:
        by_group.setdefault(gm.get(nm, nm), []).append(nm)

    groups = sorted(by_group)
    rng.shuffle(groups)
    n_test = int(round(len(groups) * test_frac))
    test_g = set(groups[:n_test])
    rest = groups[n_test:]
    n_val = int(round(len(rest) * val_frac_of_rest))
    val_g = set(rest[:n_val])

    tr, va, te = [], [], []
    for g in groups:
        bucket = te if g in test_g else (va if g in val_g else tr)
        bucket.extend(by_group[g])
    return tr, va, te
