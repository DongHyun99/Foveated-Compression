#!/usr/bin/env python3
"""Question tokens for the Foveated Selector (Sec. 3.3): an embedding lookup only.

The question is tokenised, truncated to 48 tokens, and read from the FROZEN
language model's input-embedding table. No forward pass: the selector needs
neither an additional language-model pass nor an additional vision pass.
"""
import torch

import config


class QuestionEmbedder:
    """Input embeddings of the RAW question text.

    Pass the dataset's question, NOT config.build_question(task, q) -- the
    appended instruction is constant, so including it only spends the 48-token
    budget on text that carries no information about where to look.
    """

    def __init__(self, model, tokenizer, max_tokens=config.SELECTOR["max_question_tokens"]):
        self.emb = model.get_input_embeddings()
        self.tok = tokenizer
        self.max_tokens = max_tokens

    @torch.no_grad()
    def tokens(self, question):
        """[T, D] float32, T <= max_tokens; None for an empty question."""
        ids = self.tok(question, add_special_tokens=False)["input_ids"][:self.max_tokens]
        if not ids:
            return None
        t = torch.as_tensor(ids, device=self.emb.weight.device)
        return self.emb(t).to(torch.float32)

    @torch.no_grad()
    def __call__(self, question):
        """[D] mean-pooled embedding, for the question-conditioned merger variant."""
        tokens = self.tokens(question)
        return tokens.mean(0) if tokens is not None else None
