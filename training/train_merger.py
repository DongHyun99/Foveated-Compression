#!/usr/bin/env python3
"""Stage 1: behavioural self-distillation of the Foveated Merger (Sec. 3.2).

Only the merger is trained; the vision encoder, Qwen's native visual merger and
the language model stay frozen. The native model's greedy answer y_{1:L} and
its next-token distributions are cached (`data.teacher_cache`), the
uniformly compressed model is teacher-forced on the same ids, and the loss is

    L_merge = 1/L sum_l KL(p_T^(l) || p_S^(l))                      (Eq. 2)

There is no ground-truth answer anywhere in this loop. The student differs
from the teacher in exactly one respect -- its visual tokens were compressed --
so the gradient measures what compression destroyed and nothing else. Training
toward gold answers instead lets the model learn dataset answer priors: on a
wider-channel variant the loss fell 1.40 -> 0.86 while accuracy fell 58% -> 45%
and teacher agreement collapsed 86% -> 48%.

THE STUDENT SEQUENCE. During training each compressed token is written into all
nine native slots of its unit, so sequence length and M-RoPE stay native while
the content is fully compressed. This replicated form was measured to agree
with the genuinely shorter mixed-resolution sequence used at evaluation
(+0.16pp [-0.78, +1.10]); nothing is ever REPORTED from it, because it does not
share the packed sequence's cost.

DISTRIBUTED. Launched under torchrun, every rank holds a full frozen backbone,
processes its own slice of the items, and merger gradients are averaged across
ranks before each optimiser step -- 8-GPU data parallelism with gradient
accumulation 4 (App. A.1). Every rank runs the same number of steps, so the
item list is truncated to a multiple of the world size.
"""
import os

import torch

import config
from evaluation.metrics import task_of
from models import foveated_merger as fm
from models.foveated_compression import compress, encode_native
from models.qwen2_5_vl import forced_answer_logits
from models.token_injection import TokenInjector
from utils.geometry import block_index, training_view
from training.losses import behavioral_kl
from training.optim import adamw_with_warmup


def replicate_compressed(native, blocks, compressed):
    """[N, D] student tokens: every unit's native slots carry its compressed token."""
    out = native.clone().to(compressed.dtype)
    out[blocks.reshape(-1)] = compressed.repeat_interleave(blocks.shape[1], dim=0)
    return out


def _dist():
    """(rank, world_size), initialising the process group under torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), world


def _average_grads(params, world):
    if world == 1:
        return
    import torch.distributed as dist

    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad /= world


def save_checkpoint(path, merger, step, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"merger": fm.state(merger), "step": int(step), "train_config": dict(cfg)}, tmp)
    os.replace(tmp, path)
    return path


def merger_step(mw, merger, injector, teacher, item, max_visual_tokens):
    """Forward + backward for one item. -> (kl, agreement); gradients accumulate."""
    name, image, question, _answer = item
    task = task_of(name)
    visual = mw.model.model.visual
    factor = merger.merge * visual.spatial_merge_size * visual.patch_size
    image = training_view(image, task, factor, max_visual_tokens)

    native, patch_grid, grid = encode_native(mw, injector, image)
    compressed = compress(mw, merger, patch_grid, grid)
    blocks = torch.as_tensor(block_index(*grid, merger.merge)[0], device=native.device)
    student = replicate_compressed(native, blocks, compressed)

    ids, t_ids, t_lp = teacher.get(name, native.device)
    logits = forced_answer_logits(mw, config.image_entry(image, "native"), question, task,
                                  ids, injector, repl=student)
    return behavioral_kl(t_ids, t_lp, logits)


def train_merger(mw, items, teacher, out_path, arch="foveated_merger",
                 cfg=config.MERGER_TRAIN, log_every=25, save_every=200):
    """Train the Foveated Merger with Table A1's merger settings."""
    rank, world = _dist()
    dev = mw.model.device
    visual = mw.model.model.visual
    merger = fm.build(arch=arch, **config.MERGER).to(dev).train()
    injector = TokenInjector(visual, capture_patch=True)
    opt, sched = adamw_with_warmup(merger.parameters(), cfg["lr"], cfg["weight_decay"],
                                   cfg["warmup_steps"])

    items = [it for it in items if it[0] in teacher]
    items = items[:len(items) // world * world][rank::world]
    if rank == 0:
        print(f"merger: {sum(p.numel() for p in merger.parameters()) / 1e6:.2f}M params, "
              f"{len(items)} items/rank x {world} ranks", flush=True)

    step, micro, kls, agrees = 0, 0, [], []
    for _epoch in range(cfg["epochs"]):
        for item in items:
            kl, agree = merger_step(mw, merger, injector, teacher, item,
                                    cfg["max_visual_tokens"])
            (kl / cfg["grad_accum"]).backward()
            kls.append(float(kl.detach()))
            agrees.append(agree)
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            _average_grads(merger.parameters(), world)
            torch.nn.utils.clip_grad_norm_(merger.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % log_every == 0:
                n = log_every * cfg["grad_accum"]
                print(f"step {step:>6}  kl {sum(kls[-n:]) / len(kls[-n:]):.4f}  "
                      f"agree {sum(agrees[-n:]) / len(agrees[-n:]):.3f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if rank == 0 and step % save_every == 0:
                save_checkpoint(out_path, merger, step, cfg)
    if rank == 0:
        save_checkpoint(out_path, merger, step, cfg)
        print(f"saved {out_path} at step {step}")
    return merger
