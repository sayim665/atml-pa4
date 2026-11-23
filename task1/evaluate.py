"""Perplexity suites and zero-shot HellaSwag scoring (shared by every condition)."""
import math
import re

import torch
import torch.nn.functional as F

from task1 import common, config


@torch.no_grad()
def sequence_nll(model, sequences, batch_size=config.EVAL_BATCH):
    """Per-sequence summed NLL of tokens 1..len-1 given the preceding tokens (token 0 is context only).

    Each sequence is evaluated independently (no state carried across windows).
    Returns lists (nll_sum, n_scored) aligned with `sequences`.
    """
    dev = common.device()
    sums, counts = [0.0] * len(sequences), [0] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        longest = max(len(sequences[i]) for i in idx)
        ids = torch.zeros(len(idx), longest, dtype=torch.long)
        mask = torch.zeros(len(idx), longest, dtype=torch.long)
        for row, i in enumerate(idx):
            seq = sequences[i]
            ids[row, :len(seq)] = torch.tensor(seq)
            mask[row, :len(seq)] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for row, i in enumerate(idx):
            n = len(sequences[i])
            loss = F.cross_entropy(logits[row, :n - 1].float(), ids[row, 1:n], reduction="sum")
            sums[i] = float(loss)
            counts[i] = n - 1
        del logits
    return sums, counts


def perplexity_suite(model, suite):
    windows = common.load_windows(suite)
    sequences = []
    for w in windows:
        lo, hi = w["score_target_offsets"]
        assert lo == 1 and hi - lo == w["scored_tokens"]
        sequences.append(w["token_ids"][:hi])
    sums, counts = sequence_nll(model, sequences)
    total_nll, total_n = sum(sums), sum(counts)
    per_window = [dict(source_row=w["source_row"], token_offset=w["token_offset"], nll_sum=s, n=c)
                  for w, s, c in zip(windows, sums, counts)]
    return dict(suite=suite, n_windows=len(windows), n_scored_tokens=total_n, mean_nll=total_nll / total_n,
                ppl=math.exp(total_nll / total_n)), per_window


def development_nll(model):
    rows = common.load_development()
    sums, counts = sequence_nll(model, [r["token_ids"] for r in rows])
    return sum(sums) / sum(counts)


# ---------------------------------------------------------------- HellaSwag (lm-evaluation-harness convention)

HELLASWAG_TASK_VERSION = "lm-eval-harness hellaswag v1 preprocessing; zero-shot; ctx_a + ' ' + ctx_b.capitalize()"


def _preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def hellaswag_doc(example):
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    query = _preprocess(example["activity_label"] + ": " + ctx)
    choices = [_preprocess(e) for e in example["endings"]]
    return query, choices, int(example["label"])


def _encode_pair(tokenizer, context, continuation):
    # lm-eval: move trailing context whitespace onto the continuation, encode whole, split by context length
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tokenizer(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


@torch.no_grad()
def hellaswag(model, tokenizer, limit=None, batch_size=config.HELLASWAG_BATCH):
    rows = common.load_hellaswag(limit)
    dev = common.device()
    requests = []
    for r_index, row in enumerate(rows):
        query, choices, gold = hellaswag_doc(row["example"])
        for c_index, choice in enumerate(choices):
            ctx, cont = _encode_pair(tokenizer, query, " " + choice)
            requests.append((r_index, c_index, ctx, cont))
    loglik = {}
    order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][2]) + len(requests[i][3])))
    for start in range(0, len(order), batch_size):
        batch = [requests[i] for i in order[start:start + batch_size]]
        seqs = [ctx + cont for _, _, ctx, cont in batch]
        longest = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), longest, dtype=torch.long)
        mask = torch.zeros(len(seqs), longest, dtype=torch.long)
        for row, s in enumerate(seqs):
            ids[row, :len(s)] = torch.tensor(s)
            mask[row, :len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        for row, (r_index, c_index, ctx, cont) in enumerate(batch):
            lo, hi = len(ctx), len(ctx) + len(cont)
            logp = F.log_softmax(logits[row, lo - 1:hi - 1].float(), dim=-1)
            target = torch.tensor(cont, device=logp.device)
            loglik[(r_index, c_index)] = float(logp.gather(1, target[:, None]).sum())
        del logits
    per_example = []
    correct = correct_norm = 0
    for r_index, row in enumerate(rows):
        query, choices, gold = hellaswag_doc(row["example"])
        scores = [loglik[(r_index, c)] for c in range(len(choices))]
        lengths = [len(c) for c in choices]
        norm = [s / l for s, l in zip(scores, lengths)]
        pred = max(range(len(scores)), key=lambda c: scores[c])
        pred_norm = max(range(len(norm)), key=lambda c: norm[c])
        correct += pred == gold
        correct_norm += pred_norm == gold
        per_example.append(dict(order=row["order"], source_row_index=row["source_row_index"], gold=gold,
                                loglik=scores, char_len=lengths, pred=pred, pred_norm=pred_norm))
    n = len(rows)
    summary = dict(n=n, acc=correct / n, acc_norm=correct_norm / n, task_version=HELLASWAG_TASK_VERSION)
    return summary, per_example
