"""Shared helpers: data loading, model loading, eligible-weight enumeration, weight snapshots."""
import gzip
import json
import random
import time

import numpy as np
import torch

from task1 import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def locked(name):
    return config.DATA_DIR / "public_locked" / name


def load_calibration():
    rows = read_jsonl(locked("c4_calibration_locked.jsonl.gz"))
    if config.SMOKE:
        rows = rows[:4]
    return rows


def load_development():
    rows = read_jsonl(locked("c4_development_locked.jsonl.gz"))
    if config.SMOKE:
        rows = rows[:2]
    return rows


def load_windows(suite):
    name = {"wikitext2": "wikitext2_test_windows_locked.jsonl.gz",
            "c4": "c4_validation_windows_locked.jsonl.gz"}[suite]
    rows = read_jsonl(locked(name))
    if config.SMOKE:
        rows = rows[:3]
    return rows


def load_hellaswag(limit=None):
    rows = read_jsonl(locked("hellaswag_val_locked.jsonl.gz"))
    rows.sort(key=lambda r: r["order"])
    if limit is not None:
        rows = rows[:limit]
    if config.SMOKE:
        rows = rows[:6]
    return rows


def source_revisions():
    path = config.DATA_DIR / "local_source_revisions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["revisions"]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id=config.MODEL_ID, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if revision is None:
        revision = source_revisions().get("model_base")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 attn_implementation="sdpa")
    model.to(device()).eval()
    model.requires_grad_(False)
    return model, tokenizer, revision


def eligible_linears(model):
    """Ordered list of (name, block index, projection type, nn.Linear) for every eligible projection."""
    out = []
    for block_index, layer in enumerate(model.model.layers):
        for proj in config.ELIGIBLE:
            parent = layer.self_attn if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            module = getattr(parent, proj)
            out.append((f"model.layers.{block_index}.{'self_attn' if parent is layer.self_attn else 'mlp'}.{proj}",
                        block_index, proj, module))
    return out


def check_enumeration(model):
    """Assert eligible weights are enumerated exactly once and tied weights are counted once."""
    linears = eligible_linears(model)
    ids = [id(m.weight) for _, _, _, m in linears]
    assert len(ids) == len(set(ids)), "an eligible weight appears twice"
    assert len(linears) == len(model.model.layers) * len(config.ELIGIBLE)
    unique = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        unique.setdefault(param.data_ptr(), []).append(name)
    tied_ptrs = [ptr for ptr, names in unique.items() if len(names) > 1]
    tied = [unique[ptr] for ptr in tied_ptrs]
    eligible_ptrs = {m.weight.data_ptr() for _, _, _, m in linears}
    assert not (set(tied_ptrs) & eligible_ptrs), "a tied weight is eligible"
    total_unique = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    eligible = sum(m.weight.numel() for _, _, _, m in linears)
    return dict(n_eligible_matrices=len(linears), eligible_params=eligible,
                total_unique_params=total_unique, other_params=total_unique - eligible,
                tied_groups=tied, tie_word_embeddings=bool(getattr(model.config, "tie_word_embeddings", False)))


class WeightSnapshot:
    """CPU copy of every parameter that a compression condition may touch, used to restore the FP16 model."""

    def __init__(self, model):
        self.tensors = {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters()
                        if _touchable(name)}

    def restore(self, model):
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved in self.tensors.items():
                params[name].copy_(saved.to(params[name].device))


def _touchable(name):
    # eligible projection weights plus norm weights (AWQ verification folds scales into norms)
    return name.endswith(".weight") and (any(p in name for p in config.ELIGIBLE) or "layernorm" in name)


def assert_restored(model, snapshot):
    params = dict(model.named_parameters())
    for name, saved in snapshot.tensors.items():
        if not torch.equal(params[name].detach().cpu(), saved):
            raise AssertionError(f"{name} differs from the FP16 checkpoint")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.seconds = time.perf_counter() - self.start


def environment():
    import platform
    import sys
    import transformers
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                transformers=transformers.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                attention_backend="sdpa", precision="fp16" if torch.cuda.is_available() else "fp32")
    try:
        import huggingface_hub
        info["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        pass
    return info
