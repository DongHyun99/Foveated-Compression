#!/usr/bin/env python3
"""The frozen Qwen2.5-VL-7B-Instruct backbone: loading, generation, teacher forcing.

Every generation in the repository goes through `generate_answer`, so all
conditions -- native included -- share one protocol: the common short-answer
instruction, greedy decoding, a single beam, at most 16 new tokens (App. A.2).
"""
import torch
import torch.nn.functional as F

import config
from models import mixed_resolution as mr


class ModelWrapper:
    """The model, its processor and tokenizer, held together with the protocol.

    Holds no compression state: everything that varies between conditions is a
    property of the TOKENS handed to the model, not of the model.
    """

    def __init__(self, model, processor, system_prompt, min_pixels, max_pixels):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels


def load_qwen2_5_vl(device="cuda", pretrained=None, torch_dtype="bfloat16",
                    min_pixels=None, max_pixels=None, system_prompt=None,
                    attn_implementation="flash_attention_2"):
    """Load Qwen2.5-VL under this protocol's pixel bounds.

    The pixel bounds are passed to the PROCESSOR, not applied afterwards: the
    processor re-resizes on top of whatever it is handed, so bounds supplied
    later are silently overridden and every token count shifts.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = pretrained or config.BACKBONE["model_id"]
    mn = config.BACKBONE["min_pixels"] if min_pixels is None else int(min_pixels)
    mx = config.BACKBONE["max_pixels"] if max_pixels is None else int(max_pixels)
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    processor = AutoProcessor.from_pretrained(model_id, min_pixels=mn, max_pixels=mx)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_implementation,
        device_map=device).eval()
    return ModelWrapper(model, processor,
                        system_prompt or config.BACKBONE["system_prompt"], mn, mx)


def load_frozen_backbone(device="cuda", needs_attention=False):
    """Load the backbone under `config.model_kwargs()`, verify it, freeze it."""
    mw = load_qwen2_5_vl(device=device, **config.model_kwargs(needs_attention))
    config.assert_matches(mw, needs_attention)
    for p in mw.model.parameters():
        p.requires_grad_(False)
    return mw


def _inputs(mw, entries, question_text, task):
    """(chat text, images) with every image already sized by its own entry's bounds."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": mw.system_prompt},
        {"role": "user", "content": entries + [
            {"type": "text", "text": config.build_question(task, question_text)}]},
    ]
    text = mw.processor.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return text, image_inputs


def _encode(mw, text, image_inputs):
    """Processor call that never re-resizes.

    `process_vision_info` has already sized every image by its entry's role
    bounds. The processor's own 200704-pixel floor would silently upscale a
    downsampled view or a mixed-resolution placeholder frame on top of that,
    changing the token count, so its bounds are opened here.
    """
    return mw.processor(text=[text], images=image_inputs, videos=None, return_tensors="pt",
                        min_pixels=config.PROCESSOR_MIN_PIXELS, max_pixels=10 ** 9)


@torch.inference_mode()
def generate_answer(mw, entries, question_text, task, confidence_sink=None):
    """The model's greedy answer under the shared generation protocol.

    `confidence_sink`, when given, also records what generate() already
    computed -- first-token log-prob and entropy (the decoder-side signal of
    Sec. 4.4), mean and minimum token log-prob, token count. Decoding is
    unchanged.
    """
    text, image_inputs = _inputs(mw, entries, question_text, task)
    inputs = _encode(mw, text, image_inputs).to(mw.model.device)
    gen_kwargs = config.generation_kwargs(mw.tokenizer)

    if confidence_sink is None:
        out = mw.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    out = mw.model.generate(**inputs, output_scores=True, return_dict_in_generate=True,
                            **gen_kwargs)
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    eos = {mw.tokenizer.eos_token_id, mw.tokenizer.pad_token_id}
    logps, first_entropy = [], None
    for step, logits in enumerate(out.scores):
        tok = int(gen_ids[step])
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        if step == 0:
            first_entropy = float(-(logp.exp() * logp).sum())
        if tok in eos and step > 0:
            break
        logps.append(float(logp[tok]))
    if not logps:
        logps = [float(torch.log_softmax(out.scores[0][0].float(), -1)[int(gen_ids[0])])]
    confidence_sink.update({
        "first_logprob": logps[0], "first_entropy": first_entropy,
        "mean_logprob": float(sum(logps) / len(logps)),
        "min_logprob": float(min(logps)), "sum_logprob": float(sum(logps)),
        "n_tokens": len(logps),
    })
    return mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def teacher_topk(mw, entry, question_text, task, topk):
    """Greedy answer ids and the top-k of the model's RAW next-token log-probs.

    -> (ids [L], topk_ids [L, K], topk_lp [L, K], retained mass [L],
        visual tokens) or None for an empty answer.

    The answer is decoded exactly as `generate_answer` decodes it. The stored
    distribution comes from `output_logits` (the model's own logits), NOT
    `output_scores`: the checkpoint's generation config carries
    repetition_penalty=1.05, which `scores` include even under greedy
    decoding, while the student's teacher-forced logits never see it. The
    prompt goes through the same `_inputs`/`_encode` as `forced_answer_logits`,
    so teacher and student tokenise the same sequence.
    """
    text, image_inputs = _inputs(mw, [entry], question_text, task)
    inputs = _encode(mw, text, image_inputs).to(mw.model.device)
    out = mw.model.generate(**inputs, output_logits=True, return_dict_in_generate=True,
                            **config.generation_kwargs(mw.tokenizer))
    ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    if ids.numel() == 0:
        return None
    lp = torch.log_softmax(torch.stack([s[0] for s in out.logits]).float(), -1)
    top_lp, top_ids = torch.topk(lp, min(topk, lp.shape[-1]), dim=-1)
    thw = inputs["image_grid_thw"][0]
    n_visual = int(thw.prod()) // mw.model.model.visual.spatial_merge_size ** 2
    return (ids.int().cpu().numpy(), top_ids.int().cpu().numpy(),
            top_lp.half().cpu().numpy(), top_lp.exp().sum(-1).cpu().numpy(), n_visual)


@torch.inference_mode()
def generate_mixed(mw, frame_entry, question_text, task, tokens, lay):
    """Greedy answer from a mixed-resolution sequence instead of the image's own tokens."""
    with mr.MixedFrame(mw, tokens, lay):
        return generate_answer(mw, [frame_entry], question_text, task)


@torch.inference_mode()
def answer_logprob(mw, entries, question_text, answer_text, task):
    """Mean per-token log p(answer_text | entries, question). Teacher-forced.

    MEAN, not sum: a longer gold answer would otherwise always score lower,
    which makes values incomparable across items and across tasks.
    """
    prompt_text, image_inputs = _inputs(mw, entries, question_text, task)
    prompt_inputs = _encode(mw, prompt_text, image_inputs)
    full_inputs = _encode(mw, prompt_text + answer_text, image_inputs).to(mw.model.device)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    answer_ids = full_inputs["input_ids"][0, prompt_len:]
    if answer_ids.numel() == 0:
        # Byte-pair merging occasionally absorbs a short answer into the
        # prompt's final token; that is "no measurement", not a 0.0 log-prob.
        return float("-inf")

    out = mw.model(**full_inputs, use_cache=False)
    pred_logits = out.logits[0, prompt_len - 1: prompt_len - 1 + answer_ids.shape[0]]
    logprobs = F.log_softmax(pred_logits.float(), dim=-1)
    return logprobs.gather(1, answer_ids.unsqueeze(1)).squeeze(1).mean().item()


def forced_answer_logits(mw, entry, question, task, answer_ids, injector, repl=None):
    """Logits over the answer span, teacher-forcing exactly `answer_ids`.

    Position i of the returned slice is the distribution the model emits for
    answer token i -- the same alignment as the cached teacher. `repl` replaces
    the vision encoder's output for this one forward (gradients flow into it);
    repl=None leaves the native tokens untouched.
    """
    text, image_inputs = _inputs(mw, [entry], question, task)
    inp = _encode(mw, text, image_inputs).to(mw.model.device)
    L = int(answer_ids.shape[0])
    aid = answer_ids.to(inp["input_ids"].dtype)[None]
    ids = torch.cat([inp["input_ids"], aid], 1)
    mask = torch.cat([inp["attention_mask"], torch.ones_like(aid)], 1)

    injector.repl = repl
    try:
        out = mw.model(input_ids=ids, attention_mask=mask,
                       pixel_values=inp["pixel_values"],
                       image_grid_thw=inp["image_grid_thw"],
                       use_cache=False, logits_to_keep=L + 1)
    finally:
        injector.repl = None
    return out.logits[0, -L - 1:-1]
