"""Storage accounting and measured latency / peak memory (dense FP16 execution of the effective weights)."""
import statistics

import torch

from task1 import common, config, quant


def other_params_bytes(enum_info):
    # embeddings (tied with lm_head, counted once), norms, biases: FP16
    return 2 * enum_info["other_params"]


def dense_fp16_bytes(enum_info):
    return 2 * enum_info["eligible_params"] + other_params_bytes(enum_info)


def sparse_storage(model, enum_info):
    eligible = 0
    nonzeros = 0
    for _, _, _, m in common.eligible_linears(model):
        nz = int((m.weight != 0).sum())
        nonzeros += nz
        eligible += quant.sparse_tensor_bytes(tuple(m.weight.shape), nz)["total"]
    return dict(format="mask bit per eligible weight + FP16 per nonzero (per-tensor byte alignment)",
                eligible_nonzeros=nonzeros,
                eligible_bytes=eligible, total_bytes=eligible + other_params_bytes(enum_info),
                dense_fp16_total_bytes=dense_fp16_bytes(enum_info),
                dense_with_zeros_total_bytes=dense_fp16_bytes(enum_info))


def quantized_storage(model, enum_info, bits_by_block, serialized_scale_bytes=0, bits_by_name=None):
    payload = metadata = 0
    for name, block, _, m in common.eligible_linears(model):
        bits = bits_by_name[name] if bits_by_name is not None else bits_by_block[block]
        b = quant.quantized_tensor_bytes(tuple(m.weight.shape), bits)
        payload += b["payload"]
        metadata += b["metadata"]
    total = payload + metadata + serialized_scale_bytes + other_params_bytes(enum_info)
    eligible = enum_info["eligible_params"]
    return dict(payload_bytes=payload, group_metadata_bytes=metadata,
                serialized_awq_scale_bytes=serialized_scale_bytes,
                uncompressed_fp16_bytes=other_params_bytes(enum_info), total_bytes=total,
                nominal_bits_per_eligible_weight=8 * payload / eligible,
                effective_bits_per_eligible_weight=8 * (payload + metadata + serialized_scale_bytes) / eligible,
                dense_fp16_total_bytes=dense_fp16_bytes(enum_info))


@torch.no_grad()
def measure(model, token_source):
    """Prefill latency at 512/2048 tokens, 64-token greedy decode latency after a 512-token prompt,
    peak allocated/reserved memory. Batch 1, fixed input ids, CUDA-synchronized, warmed up."""
    dev = common.device()
    ids_all = torch.tensor(token_source, dtype=torch.long)
    out = dict(unit="milliseconds", repetitions=config.TIMING_REPS, warmup=config.WARMUP_REPS, batch=1,
               execution="dense FP16 matmuls on the effective (dequantized / zero-filled) weights")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for length in config.PREFILL_LENGTHS:
        ids = ids_all[:length][None].to(dev)
        times = []
        for rep in range(config.WARMUP_REPS + config.TIMING_REPS):
            with common.Timer() as t:
                model(input_ids=ids, use_cache=False)
            if rep >= config.WARMUP_REPS:
                times.append(t.seconds * 1000)
        out[f"prefill_{length}_ms"] = _summary(times)
    prompt = ids_all[:512][None].to(dev)
    times = []
    for rep in range(config.WARMUP_REPS + config.TIMING_REPS):
        first = model(input_ids=prompt, use_cache=True)
        past = first.past_key_values
        token = first.logits[:, -1:].argmax(-1)
        with common.Timer() as t:
            for _ in range(config.DECODE_TOKENS):
                step = model(input_ids=token, past_key_values=past, use_cache=True)
                past = step.past_key_values
                token = step.logits[:, -1:].argmax(-1)
        if rep >= config.WARMUP_REPS:
            times.append(t.seconds * 1000)
    out[f"decode_{config.DECODE_TOKENS}_ms"] = _summary(times)
    if torch.cuda.is_available():
        out["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        out["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return out


def _summary(times):
    return dict(mean=statistics.mean(times), std=statistics.pstdev(times), min=min(times), all=times)
