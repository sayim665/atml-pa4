"""Calibration statistics collected once from the unchanged FP16 model on the 128 C4 calibration sequences.

Per eligible projection input: channel sum of squares (Wanda ||X_j||_2), channel mean |x| (AWQ),
and a fixed, seeded token subsample of the inputs (AWQ scale search, layer-output error).
Per block: ShortGPT block influence BI = 1 - E_t cos(h_in, h_out).
"""
import time

import torch

from task1 import common, config

# projections that read the same input tensor share one record
INPUT_GROUP = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
               "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}


def group_key(block, proj):
    return f"{block}.{INPUT_GROUP[proj]}"


@torch.no_grad()
def collect(model, rows=None, subsample=config.CALIB_SUBSAMPLE_TOKENS, batch_size=config.CALIB_BATCH):
    rows = rows if rows is not None else common.load_calibration()
    seq_len = len(rows[0]["token_ids"])
    total_tokens = len(rows) * seq_len
    gen = torch.Generator().manual_seed(config.SEED)
    keep = torch.randperm(total_tokens, generator=gen)[:min(subsample, total_tokens)].sort().values
    stats, samples, handles = {}, {}, []
    bi_sum = torch.zeros(len(model.model.layers), dtype=torch.float64)
    state = dict(offset=0)

    def make_hook(key):
        def hook(module, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float()
            rec = stats.setdefault(key, dict(sq=torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device),
                                             abs=torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device),
                                             n=0))
            if rec.get("_seen_offset") == state["offset"]:
                return  # q/k/v (and gate/up) share an input: count it once per batch
            rec["_seen_offset"] = state["offset"]
            rec["sq"] += (flat ** 2).sum(0).double()
            rec["abs"] += flat.abs().sum(0).double()
            rec["n"] += flat.shape[0]
            lo, hi = state["offset"], state["offset"] + flat.shape[0]
            sel = keep[(keep >= lo) & (keep < hi)] - lo
            if len(sel):
                samples.setdefault(key, []).append(flat[sel.to(flat.device)].half().cpu())
        return hook

    def make_block_hook(block):
        def hook(module, args, kwargs, output):
            h_in = args[0] if args else kwargs["hidden_states"]
            h_out = output[0] if isinstance(output, tuple) else output
            cos = torch.nn.functional.cosine_similarity(h_in.float(), h_out.float(), dim=-1)
            bi_sum[block] += float((1 - cos).sum())
        return hook

    for name, block, proj, module in common.eligible_linears(model):
        handles.append(module.register_forward_pre_hook(make_hook(group_key(block, proj))))
    for block, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_block_hook(block), with_kwargs=True))
    start = time.perf_counter()
    dev = common.device()
    try:
        for i in range(0, len(rows), batch_size):
            ids = torch.tensor([r["token_ids"] for r in rows[i:i + batch_size]], device=dev)
            model(input_ids=ids)
            state["offset"] += ids.numel()
    finally:
        for h in handles:
            h.remove()
    common.sync()
    out = dict(
        input_norm={k: v["sq"].sqrt().float().cpu() for k, v in stats.items()},
        input_mean_abs={k: (v["abs"] / v["n"]).float().cpu() for k, v in stats.items()},
        samples={k: torch.cat(v) for k, v in samples.items()},
        block_influence=(bi_sum / total_tokens).tolist(),
        n_tokens=total_tokens, subsample_tokens=len(keep), subsample_index=keep.tolist(),
        calibration_source_rows=[r["source_row"] for r in rows],
        seconds=time.perf_counter() - start)
    return out


def load_or_collect(model):
    path = config.OUT_DIR / "calibration_stats.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    stats = collect(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    return stats


def input_for(stats, block, proj, kind):
    return stats[kind][group_key(block, proj)]
