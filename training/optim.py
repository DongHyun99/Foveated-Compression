#!/usr/bin/env python3
"""Optimiser and schedule shared by both training stages (Table A1)."""
import torch


def adamw_with_warmup(params, lr, weight_decay, warmup_steps):
    """AdamW plus a linear warmup over `warmup_steps`, then a constant rate."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))
    return opt, sched
