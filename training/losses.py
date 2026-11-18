#!/usr/bin/env python3
"""The two training objectives of the paper."""
import torch
import torch.nn.functional as F


def behavioral_kl(teacher_topk_ids, teacher_topk_lp, student_logits):
    """L_merge (Eq. 2): mean over answer positions of KL(p_T || p_S).

    The teacher distribution is its cached top-k, renormalised over the k ids.
    -> (kl, fraction of positions where the student's argmax is the teacher's).
    `teacher_topk_ids` / `teacher_topk_lp` are [L, K] sorted descending, so
    column 0 is the teacher's own greedy token and the agreement is exact.
    """
    lsm = F.log_softmax(student_logits.float(), -1)            # [L, V]
    tlm = F.log_softmax(teacher_topk_lp, -1)                   # renormalise over top-k
    p = tlm.exp()
    kl = (p * (tlm - torch.gather(lsm, 1, teacher_topk_ids))).sum(-1).mean()
    agree = (student_logits.argmax(-1) == teacher_topk_ids[:, 0]).float().mean().item()
    return kl, agree


def tie_aware_selector_loss(cell_scores, winners):
    """L_selector (Eq. 4): -log( sum_{c in W} exp(s_c) / sum_j exp(s_j) ).

    Indifferent between tied winners; only pushes mass away from the losers.
    A plain cross-entropy against one arbitrarily chosen winner would penalise
    ranking an equally correct cell first.
    """
    mask = torch.zeros_like(cell_scores, dtype=torch.bool)
    mask[list(winners)] = True
    if not bool(mask.any()) or bool(mask.all()):
        raise ValueError(f"winner set {sorted(winners)} carries no spatial preference; "
                         f"all-nine ties must be dropped upstream")
    return -(torch.logsumexp(cell_scores[mask], dim=0) - torch.logsumexp(cell_scores, dim=0))
