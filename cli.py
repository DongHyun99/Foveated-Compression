#!/usr/bin/env python3
"""Command-line entry point, one subcommand per stage of the paper, in run order.

    dump-teacher    The native model's greedy answers and top-k next-token
                    log-probs on TRAIN splits: Stage 1's supervision. Shard it
                    across GPUs with --shard-idx / --num-shards; --report
                    summarises a finished dump.

    train-merger    Stage 1 (Sec. 3.2). Behavioural self-distillation of the
                    Foveated Merger from cached native-model answers. Run under
                    torchrun for the paper's 8-GPU data parallelism.

    label           Exhaustive K=1 interventions on TRAIN splits with the frozen
                    merger (Sec. 3.3, App. B): the selector's supervision.

    train-selector  Stage 2 (Sec. 3.3). The Foveated Selector on tie-aware
                    winner sets, merger and backbone frozen.

    eval            The conditions of Table 1 on the eight benchmarks.

    crop-selector   Train/ablate the cache-based crop selector (not in the paper).

Every model-loading subcommand loads the backbone through
`config.model_kwargs()` and asserts it matches, because a wrapper loaded with
different pixel bounds produces a different number of visual tokens for the
same image, which makes every cost number in the run incomparable.
"""
import argparse
import json
import os
import sys

import torch

import config


def _device(args):
    """torchrun's LOCAL_RANK pins each rank to its own GPU."""
    local = os.environ.get("LOCAL_RANK")
    return f"cuda:{local}" if local is not None else args.device


def _backbone(args):
    from models.qwen2_5_vl import load_frozen_backbone

    return load_frozen_backbone(device=_device(args))


def _load_items(args):
    """(name, image, question, answer) tuples across the requested tasks."""
    from data.benchmarks import sample_task_pool

    return sample_task_pool([t for t in args.tasks.split(",") if t], args.cap_per_task,
                            shard_idx=args.shard_idx, num_shards=args.num_shards)


def _atomic_json(path, payload):
    from evaluation.interventions import atomic_json

    atomic_json(path, payload)


# --------------------------------------------------------------------------
def cmd_dump_teacher(args):
    from data import teacher_cache

    if args.report:
        teacher_cache.report(args.out)
        return
    config.seed_everything(args.seed)
    mw = _backbone(args)
    teacher_cache.dump(mw, _load_items(args), args.out, topk=args.topk,
                       max_visual_tokens=config.MERGER_TRAIN["max_visual_tokens"],
                       merge=config.MERGER["merge"], resume_from=args.resume_from,
                       checkpoint_every=args.checkpoint_every, limit=args.limit)


def cmd_train_merger(args):
    from data.teacher_cache import TeacherCache
    from training.train_merger import train_merger

    config.seed_everything(args.seed)
    mw = _backbone(args)
    teacher = TeacherCache(args.teacher_dump)
    print(f"teacher: {len(teacher)} cached answers", flush=True)
    train_merger(mw, _load_items(args), teacher, args.out, arch=args.arch,
                 log_every=args.log_every, save_every=args.save_every)


def cmd_label(args):
    from evaluation.interventions import label_cells
    from models import foveated_merger as fm
    from models.token_injection import TokenInjector

    config.seed_everything(args.seed)
    mw = _backbone(args)
    merger = fm.load(args.merger).to(mw.model.device).eval()
    injector = TokenInjector(mw.model.model.visual, capture_patch=True)
    records = label_cells(mw, injector, merger, _load_items(args), args.out_json,
                          checkpoint_every=args.checkpoint_every)
    print(f"{len(records)} labelled items -> {args.out_json}")


def cmd_train_selector(args):
    from data.selector_labels import load_winner_sets
    from models import foveated_merger as fm
    from training.train_selector import train_selector

    config.seed_everything(args.seed)
    labels = load_winner_sets(args.labels, single_winner=args.single_winner)
    print(f"winner sets: {len(labels)} spatially informative items", flush=True)
    mw = _backbone(args)
    merger = fm.load(args.merger)
    train_selector(mw, merger, _load_items(args), labels, args.out,
                   use_question=not args.no_question, seed=args.seed)


def cmd_eval(args):
    from data.selector_labels import best_fixed_cell, load_winner_sets
    from evaluation import evaluate
    from models.question_embedding import QuestionEmbedder

    config.seed_everything(args.seed)
    conditions = tuple(c for c in args.conditions.split(",") if c)
    unknown = sorted(set(conditions) - set(evaluate.CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; choose from {evaluate.CONDITIONS}")
    fixed_cell = args.fixed_cell
    if fixed_cell is None and args.fixed_cell_labels:
        fixed_cell = best_fixed_cell(load_winner_sets(args.fixed_cell_labels))
        print(f"fixed_k1 cell from training labels: {fixed_cell}", flush=True)

    mw = _backbone(args)
    merger, selector = evaluate.load_merger_and_selector(args.merger, args.selector,
                                                         mw.model.device)
    qemb = (QuestionEmbedder(mw.model, mw.tokenizer)
            if selector is not None and selector.use_question else None)
    records = evaluate.run(mw, _load_items(args), merger, selector, qemb, conditions,
                           out_json=args.out_json or None, fixed_cell=fixed_cell)
    summary = evaluate.aggregate(records, conditions)
    print(evaluate.report(summary))
    if args.out_summary:
        _atomic_json(args.out_summary, summary)


def cmd_crop_selector(args):
    """Train the cache-based crop selector across its ablation ladder.

    Multiple seeds per rung, always: a single-seed difference between two rungs
    is within run-to-run variance.
    """
    from evaluation.metrics import task_of
    from models import crop_selector as cs

    device = torch.device(args.device)
    data = torch.load(args.cache, map_location=device, weights_only=False)

    cfg = dict(cs.DEFAULT_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs
        cfg["patience"] = max(cfg["eval_every"], args.epochs)

    results = {}
    for stage in args.stages.split(","):
        stage = stage.strip()
        if stage not in cs.STAGES:
            raise SystemExit(f"unknown rung {stage!r}; choose from {sorted(cs.STAGES)}")
        per_seed = []
        for seed in range(args.n_seeds):
            model, best, q_const = cs.train_one(cfg, data["train"], data["val"], seed,
                                                device, stage, task_of,
                                                verbose=args.verbose)
            metrics = cs.evaluate(model, data["test"], q_const, task_of)
            per_seed.append(metrics)
            print(f"{stage} seed {seed}: val {best:.4f}  test {metrics['overall']:.4f}  "
                  f"informative {metrics['informative']:.4f}", flush=True)
        results[stage] = {
            "overall_mean": float(sum(m["overall"] for m in per_seed) / len(per_seed)),
            "informative_mean": float(sum(m["informative"] for m in per_seed) / len(per_seed)),
            "seeds": per_seed,
        }
    if args.out_json:
        _atomic_json(args.out_json, results)
    for stage, v in results.items():
        print(f"{stage:>10}  overall {v['overall_mean'] * 100:.2f}%  "
              f"informative {v['informative_mean'] * 100:.2f}%")


# --------------------------------------------------------------------------
def build_parser():
    from evaluation.evaluate import DEFAULT_CONDITIONS

    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=config.SEED)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def items(p, default_tasks, cap):
        p.add_argument("--tasks", default=",".join(default_tasks))
        p.add_argument("--cap-per-task", type=int, default=cap,
                       help="evenly spaced rows per task; omit for the full split")
        p.add_argument("--shard-idx", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1)

    p = sub.add_parser("dump-teacher", help="cache the native model's answers for Stage 1")
    items(p, config.TRAIN_TASKS, 4000)
    p.add_argument("--topk", type=int, default=config.MERGER_TRAIN["teacher_topk"])
    p.add_argument("--resume-from", default="",
                   help="an earlier shard whose items are skipped; must differ from --out")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=-1, help="stop after this many items")
    p.add_argument("--report", action="store_true",
                   help="summarise the shards matching --out instead of writing one")
    p.add_argument("--out", required=True, help="shard path (a glob with --report)")
    p.set_defaults(fn=cmd_dump_teacher)

    p = sub.add_parser("train-merger", help="Stage 1: distil the Foveated Merger")
    items(p, config.TRAIN_TASKS, 4000)
    p.add_argument("--teacher-dump", required=True, help="glob of cached teacher npz shards")
    p.add_argument("--arch", default="foveated_merger",
                   choices=["foveated_merger", "question_conditioned_merger"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train_merger)

    p = sub.add_parser("label", help="exhaustive K=1 interventions on train splits")
    items(p, config.TRAIN_TASKS, 4000)
    p.add_argument("--merger", required=True, help="Stage-1 checkpoint")
    p.add_argument("--out-json", required=True)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("train-selector", help="Stage 2: train the Foveated Selector")
    items(p, config.TRAIN_TASKS, 4000)
    p.add_argument("--merger", required=True, help="the frozen Stage-1 checkpoint")
    p.add_argument("--labels", required=True, help="glob of `label` output files")
    p.add_argument("--no-question", action="store_true",
                   help="ablation: remove question and spatial inputs (Table 2)")
    p.add_argument("--single-winner", action="store_true",
                   help="ablation: single-winner instead of tie-aware supervision (Table 2)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train_selector)

    p = sub.add_parser("eval", help="the conditions of Table 1")
    items(p, config.EVAL_TASKS, None)
    p.add_argument("--merger", required=True)
    p.add_argument("--selector", default="")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    p.add_argument("--fixed-cell", type=int, default=None)
    p.add_argument("--fixed-cell-labels", default="",
                   help="choose the fixed_k1 cell from these TRAINING label files")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-summary", default="")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("crop-selector", help="train/ablate the crop selector (not in the paper)")
    p.add_argument("--cache", required=True)
    p.add_argument("--stages", default="control,a,b,ab,ab_nomid")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-json", default="")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_crop_selector)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
