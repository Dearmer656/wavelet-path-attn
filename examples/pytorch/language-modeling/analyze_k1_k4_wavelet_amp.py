#!/usr/bin/env python
"""K1-vs-K4 QWAB wavelet-bias amplitude checkpoint analysis.

This script is intentionally analysis-only. It sets the existing per-instance
``_pat234_cap`` capture dict on PaTHAttention layers at runtime and optionally
wraps the bound ``_build_ctxscale_shift_logit_bias_v0`` method on loaded layer
instances for a one-batch logits-delta cross-check.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def causal_centered_rms_amp(x: torch.Tensor, q0: int, eps: float = 0.0) -> torch.Tensor:
    """Causal-centered row RMS for a chunk ``x`` with shape ``[..., Q, T_keys]``.

    For row ``i``, only keys ``k <= q0 + i`` are included, the row is centered
    over those valid keys, and one amplitude is returned per prefix element.
    """
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims [..., Q, T], got {tuple(x.shape)}")
    q_len = int(x.shape[-2])
    t_keys = int(x.shape[-1])
    if q_len <= 0 or t_keys <= 0:
        raise ValueError(f"x has invalid chunk/key shape {tuple(x.shape)}")

    device = x.device
    dtype = torch.float32 if not torch.is_floating_point(x) else x.dtype
    xf = x.to(dtype=torch.float32)
    q_abs = torch.arange(int(q0), int(q0) + q_len, device=device).view(q_len, 1)
    k_idx = torch.arange(t_keys, device=device).view(1, t_keys)
    mask = (k_idx <= q_abs).to(dtype=torch.float32)
    count = mask.sum(dim=-1).clamp_min(1.0)
    expand_shape = (1,) * (xf.dim() - 2) + mask.shape
    mask = mask.view(expand_shape)
    count = count.view((1,) * (xf.dim() - 2) + (q_len,))

    mean = (xf * mask).sum(dim=-1) / count
    centered_sq = (xf - mean.unsqueeze(-1)).square() * mask
    var = centered_sq.sum(dim=-1) / count
    if eps:
        var = var + float(eps)
    return torch.sqrt(var).to(dtype=dtype)


def causal_centered_peak_valley(x: torch.Tensor, q0: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal-centered per-row (peak, valley) for a chunk ``x`` with shape ``[..., Q, T_keys]``.

    Same causal-restriction + mean-centering convention as ``causal_centered_rms_amp``
    (future keys excluded, row mean over valid keys subtracted so a constant
    per-row shift -- which softmax ignores -- doesn't count as a peak/valley).
    Returns (peak, valley) each shaped ``[..., Q]``: peak = max centered value,
    valley = min centered value, over valid keys k <= q0+i.
    """
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims [..., Q, T], got {tuple(x.shape)}")
    q_len = int(x.shape[-2])
    t_keys = int(x.shape[-1])
    if q_len <= 0 or t_keys <= 0:
        raise ValueError(f"x has invalid chunk/key shape {tuple(x.shape)}")

    device = x.device
    dtype = torch.float32 if not torch.is_floating_point(x) else x.dtype
    xf = x.to(dtype=torch.float32)
    q_abs = torch.arange(int(q0), int(q0) + q_len, device=device).view(q_len, 1)
    k_idx = torch.arange(t_keys, device=device).view(1, t_keys)
    mask = (k_idx <= q_abs)
    expand_shape = (1,) * (xf.dim() - 2) + mask.shape
    mask_b = mask.view(expand_shape)
    count = mask_b.to(torch.float32).sum(dim=-1).clamp_min(1.0)

    mean = (xf * mask_b.to(torch.float32)).sum(dim=-1) / count
    centered = xf - mean.unsqueeze(-1)
    neg_inf = torch.finfo(torch.float32).min
    pos_inf = torch.finfo(torch.float32).max
    peak = torch.where(mask_b, centered, torch.full_like(centered, neg_inf)).amax(dim=-1)
    valley = torch.where(mask_b, centered, torch.full_like(centered, pos_inf)).amin(dim=-1)
    return peak.to(dtype=dtype), valley.to(dtype=dtype)


def component_weighted_upper_bound_amp(
    basis_by_scale: torch.Tensor,
    pi_scale: torch.Tensor,
    q0: int,
) -> torch.Tensor:
    """Sum per-scale amplitudes weighted by routing probabilities.

    ``basis_by_scale`` must have shape ``[B, Q, T, K]`` and ``pi_scale`` must
    have shape ``[B, Q, K]``. Amplitude is computed before summing components.
    """
    if basis_by_scale.dim() != 4:
        raise ValueError(f"basis_by_scale must be [B,Q,T,K], got {tuple(basis_by_scale.shape)}")
    if pi_scale.dim() != 3:
        raise ValueError(f"pi_scale must be [B,Q,K], got {tuple(pi_scale.shape)}")
    if basis_by_scale.shape[0] != pi_scale.shape[0] or basis_by_scale.shape[1] != pi_scale.shape[1]:
        raise ValueError(
            f"basis/pi batch-query mismatch: {tuple(basis_by_scale.shape)} vs {tuple(pi_scale.shape)}"
        )
    if basis_by_scale.shape[-1] != pi_scale.shape[-1]:
        raise ValueError(
            f"basis/pi scale mismatch: {tuple(basis_by_scale.shape)} vs {tuple(pi_scale.shape)}"
        )
    # [B,Q,T,K] -> [B,K,Q,T] so causal_centered_rms_amp treats K as a prefix dim.
    basis_bkqt = basis_by_scale.permute(0, 3, 1, 2).contiguous()
    amps_bkq = causal_centered_rms_amp(basis_bkqt, q0=q0).to(dtype=torch.float32)
    amps_bqk = amps_bkq.permute(0, 2, 1).contiguous()
    return (amps_bqk * pi_scale.to(dtype=torch.float32)).sum(dim=-1)


def read_kv_config(path: str | Path) -> dict:
    cfg = {}
    path = Path(path)
    if not path.exists():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Bad cfg line in {path}: {line}")
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                v = ast.literal_eval(v)
            except Exception:
                pass
            cfg[k] = v
    return cfg


def _cfg_bool(cfg: dict, key: str, default: bool) -> bool:
    if key not in cfg:
        return bool(default)
    v = cfg[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)


def _cfg_int(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _cfg_float(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def _cfg_float_or_list(cfg: dict, key: str, default: Any) -> Any:
    v = cfg.get(key, default)
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    try:
        return float(v)
    except Exception:
        return v


def _cfg_str(cfg: dict, key: str, default: str) -> str:
    return str(cfg.get(key, default))


def _normalize_rel_use_layer_list(v: Any, n_layer: Optional[int] = None):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "all", "*"):
            return None
        items = [tok for tok in re.split(r"[,\s]+", s.strip("[]()")) if tok]
    elif isinstance(v, (list, tuple, set)):
        items = list(v)
    else:
        items = [v]
    out, seen = [], set()
    for it in items:
        try:
            lid = int(it)
        except Exception:
            continue
        if n_layer is not None and int(n_layer) > 0 and lid < 0:
            lid = int(n_layer) + lid
        if n_layer is not None and (lid < 0 or lid >= int(n_layer)):
            continue
        if lid not in seen:
            seen.add(lid)
            out.append(lid)
    return sorted(out) if out else None


def apply_supply_cfg_to_config(config: Any, cfg: dict, cfg_path: Path) -> Any:
    """Mirror the run_clm.py cfg_path behavior needed for PaTH/QWAB eval loading."""
    for k, v in cfg.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    for prefix in (
        "eval_attn_heatmap",
        "wavelet_mode",
        "wavelet_ctxscale",
        "wavelet_logit_bias",
        "wavelet_router_sigmoid",
        "wavelet_ctx_feat",
        "rel_use_layer",
    ):
        for k, v in cfg.items():
            if k.startswith(prefix):
                setattr(config, k, v)
    for key in ("wavelet_mode", "rel_use_layer_list"):
        if key in cfg:
            setattr(config, key, cfg[key])

    config.attn_implementation = "path_attn"
    config.path_attn_impl = "pytorch"
    config.wavelet_mode = _cfg_str(cfg, "wavelet_mode", str(getattr(config, "wavelet_mode", "router_rel")))
    config.wavelet_ctxscale_k = _cfg_int(cfg, "wavelet_ctxscale_k", int(getattr(config, "wavelet_ctxscale_k", 8)))
    config.wavelet_ctxscale_scale_max_exp = _cfg_float_or_list(
        cfg,
        "wavelet_ctxscale_scale_max_exp",
        getattr(config, "wavelet_ctxscale_scale_max_exp", 14.0),
    )
    config.wavelet_ctxscale_scale_mask = _cfg_str(
        cfg, "wavelet_ctxscale_scale_mask", str(getattr(config, "wavelet_ctxscale_scale_mask", ""))
    )
    config.wavelet_ctxscale_ko_layers = _cfg_str(
        cfg, "wavelet_ctxscale_ko_layers", str(getattr(config, "wavelet_ctxscale_ko_layers", ""))
    )
    config.wavelet_ctx_feat_rms_eps = _cfg_float(
        cfg, "wavelet_ctx_feat_rms_eps", float(getattr(config, "wavelet_ctx_feat_rms_eps", 1e-6))
    )
    config.wavelet_ctx_feat_detach_delta = _cfg_bool(
        cfg, "wavelet_ctx_feat_detach_delta", bool(getattr(config, "wavelet_ctx_feat_detach_delta", False))
    )
    config.router_jitter_flip_ratio = _cfg_float(
        cfg,
        "router_jitter_flip_ratio",
        float(getattr(config, "router_jitter_flip_ratio", getattr(config, "router_jitter_std", 0.0))),
    )
    config.router_jitter_std = float(config.router_jitter_flip_ratio)
    config.wavelet_ctxscale_film_hidden = _cfg_int(
        cfg, "wavelet_ctxscale_film_hidden", int(getattr(config, "wavelet_ctxscale_film_hidden", 64))
    )
    config.wavelet_ctxscale_film_alpha = _cfg_float(
        cfg, "wavelet_ctxscale_film_alpha", float(getattr(config, "wavelet_ctxscale_film_alpha", 0.5))
    )
    config.wavelet_ctxscale_film_beta = _cfg_float(
        cfg, "wavelet_ctxscale_film_beta", float(getattr(config, "wavelet_ctxscale_film_beta", 0.1))
    )
    config.wavelet_ctxscale_film_clamp = _cfg_float(
        cfg, "wavelet_ctxscale_film_clamp", float(getattr(config, "wavelet_ctxscale_film_clamp", 8.0))
    )
    config.wavelet_gate_grad_clip = _cfg_float(
        cfg, "wavelet_gate_grad_clip", float(getattr(config, "wavelet_gate_grad_clip", 1.0))
    )
    config.wavelet_ctxscale_lock_window = _cfg_int(
        cfg, "wavelet_ctxscale_lock_window", int(getattr(config, "wavelet_ctxscale_lock_window", 300))
    )
    config.wavelet_ctxscale_lock_grad_eps = _cfg_float(
        cfg, "wavelet_ctxscale_lock_grad_eps", float(getattr(config, "wavelet_ctxscale_lock_grad_eps", 1e-6))
    )
    config.wavelet_ctxscale_lock_update_eps = _cfg_float(
        cfg, "wavelet_ctxscale_lock_update_eps", float(getattr(config, "wavelet_ctxscale_lock_update_eps", 1e-6))
    )
    config.wavelet_gate_autofix = _cfg_bool(
        cfg, "wavelet_gate_autofix", bool(getattr(config, "wavelet_gate_autofix", False))
    )
    config.wavelet_gate_autofix_clamp_abs = _cfg_float(
        cfg, "wavelet_gate_autofix_clamp_abs", float(getattr(config, "wavelet_gate_autofix_clamp_abs", 4.0))
    )
    config.wavelet_ctxscale_disable_layer_gate = _cfg_bool(
        cfg,
        "wavelet_ctxscale_disable_layer_gate",
        bool(getattr(config, "wavelet_ctxscale_disable_layer_gate", False)),
    )
    config.wavelet_ctxscale_rho_override = (
        float(cfg["wavelet_ctxscale_rho_override"])
        if cfg.get("wavelet_ctxscale_rho_override", None) is not None
        else getattr(config, "wavelet_ctxscale_rho_override", None)
    )
    n_layer = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    rel_layers = _normalize_rel_use_layer_list(cfg.get("rel_use_layer_list", getattr(config, "rel_use_layer_list", None)), n_layer)
    config.rel_use_layer_list = "all" if rel_layers is None else rel_layers
    config.wavelet_analysis_export = False
    config.wavelet_viz_export = False
    config.eval_attn_heatmap_enabled = False
    config.cfg_path = str(cfg_path)
    return config


def discover_checkpoints(run_dir: Path) -> Dict[int, Path]:
    out = {}
    for p in run_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)$", p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def finite_or_nan(v: Optional[float]) -> float:
    return float("nan") if v is None else float(v)


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    layers: List[Any]
    manifest: Dict[str, Any]


def find_path_layers(model: Any) -> List[Any]:
    layers = []
    for _, module in model.named_modules():
        if hasattr(module, "_build_ctxscale_shift_logit_bias_v0") and hasattr(module, "wavelet_ctxscale_k"):
            layers.append(module)
    layers.sort(key=lambda m: int(getattr(m, "layer_idx", len(layers)) or 0))
    if not layers:
        raise RuntimeError("No PaTHAttention-like layers with _build_ctxscale_shift_logit_bias_v0 found.")
    return layers


def load_model_for_analysis(checkpoint_dir: Path, run_dir: Path, device: str, dtype_name: str) -> LoadedModel:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg_path = run_dir / "supply_model.cfg"
    cfg = read_kv_config(cfg_path)
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    config = apply_supply_cfg_to_config(config, cfg, cfg_path)
    dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        config=config,
        trust_remote_code=True,
        dtype=dtype,
    )
    tokenizer_src = run_dir if (run_dir / "tokenizer.json").exists() else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), trust_remote_code=True)
    model.eval().to(device)
    layers = find_path_layers(model)
    for layer in layers:
        if bool(getattr(layer, "wavelet_ctxscale_use_head_gate", False)):
            raise AssertionError(
                f"layer {getattr(layer, 'layer_idx', '?')} has wavelet_ctxscale_use_head_gate=True; "
                "S4 captures are only valid for wavelet_ctxscale_use_head_gate=False."
            )
    first = layers[0]
    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "cfg_path": str(cfg_path),
        "wavelet_router_sigmoid_mode": str(getattr(first, "wavelet_router_sigmoid_mode", "")),
        "wavelet_mode": str(getattr(config, "wavelet_mode", "")),
        "wavelet_ctxscale_k": int(getattr(first, "wavelet_ctxscale_k")),
        "scale_list": [float(x) for x in getattr(first, "wavelet_ctxscale_scales").detach().float().cpu().tolist()],
        "wavelet_ctxscale_scale_max_exp": getattr(config, "wavelet_ctxscale_scale_max_exp", None),
        "multiscale_norm_requested": str(getattr(first, "multiscale_norm_requested", "")),
        "multiscale_sum_scale": float(getattr(first, "multiscale_sum_scale")),
        "historical_k4_factor_reproduced": abs(float(getattr(first, "multiscale_sum_scale")) - 0.5) < 1e-6,
        "num_layers": len(layers),
        "dtype": str(dtype),
        "path_attn_impl": str(getattr(config, "path_attn_impl", "")),
    }
    return LoadedModel(model=model, tokenizer=tokenizer, layers=layers, manifest=manifest)


def _tokenize_split_ids(tokenizer: Any, split: str, cache_dir: Optional[str]) -> List[int]:
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
    text_col = "text"

    def tokenize_function(examples):
        return tokenizer(examples[text_col], add_special_tokens=True)

    tokenized = raw.map(tokenize_function, batched=True, remove_columns=raw.column_names, desc=f"Tokenizing wikitext {split}")
    all_ids: List[int] = []
    for ids in tokenized["input_ids"]:
        all_ids.extend(ids)
    return all_ids


def build_eval_samples(tokenizer: Any, eval_length: int, num_samples: int, seed: int, cache_dir: Optional[str] = None) -> torch.Tensor:
    # `validation` alone is enough at short lengths (e.g. 512) but runs out of
    # non-overlapping chunks at longer lengths (2048/4096) for num_samples=128;
    # fall back to appending `train` tokens only when validation isn't enough,
    # so the L512 sample set (already validated) is completely unaffected.
    all_ids = _tokenize_split_ids(tokenizer, "validation", cache_dir)
    total = (len(all_ids) // int(eval_length)) * int(eval_length)
    n_chunks = total // int(eval_length)
    if n_chunks < int(num_samples):
        print(
            f"[build_eval_samples] validation split only yields {n_chunks} chunks at "
            f"eval_length={eval_length} (< {num_samples} requested); appending train split tokens."
        )
        all_ids = all_ids + _tokenize_split_ids(tokenizer, "train", cache_dir)
        total = (len(all_ids) // int(eval_length)) * int(eval_length)
    if total < int(eval_length):
        raise RuntimeError(f"Not enough tokens for eval_length={eval_length}")
    chunks = np.asarray(all_ids[:total], dtype=np.int64).reshape(-1, int(eval_length))
    if len(chunks) < num_samples:
        raise RuntimeError(f"Requested {num_samples} samples, but only {len(chunks)} chunks are available")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(chunks), size=int(num_samples), replace=False)
    return torch.tensor(chunks[idx], dtype=torch.long)


def _set_capture(layers: Sequence[Any], enabled: bool) -> None:
    for layer in layers:
        layer._pat234_cap = {} if enabled else None


def _clear_capture(layers: Sequence[Any]) -> None:
    for layer in layers:
        layer._pat234_cap = None


@contextmanager
def wrapped_logits_capture(layers: Sequence[Any], max_records_per_layer: int = 1):
    records: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    originals = []
    for layer in layers:
        original = layer._build_ctxscale_shift_logit_bias_v0
        sig = inspect.signature(original)
        if "E_base_raw" not in sig.parameters:
            raise RuntimeError(f"Unexpected _build_ctxscale_shift_logit_bias_v0 signature: {sig}")
        lid = int(getattr(layer, "layer_idx", len(originals)) or 0)

        def make_wrapper(orig, layer_idx):
            def wrapper(*args, **kwargs):
                if "E_base_raw" not in kwargs:
                    raise RuntimeError("_build_ctxscale_shift_logit_bias_v0 was called without E_base_raw kwarg")
                result = orig(*args, **kwargs)
                logits_out = result[0] if isinstance(result, tuple) else result
                if len(records[layer_idx]) < int(max_records_per_layer):
                    records[layer_idx].append(
                        (
                            kwargs["E_base_raw"].detach().float().cpu(),
                            logits_out.detach().float().cpu(),
                        )
                    )
                return result

            return wrapper

        originals.append((layer, original))
        layer._build_ctxscale_shift_logit_bias_v0 = make_wrapper(original, lid)
    try:
        yield records
    finally:
        for layer, original in originals:
            layer._build_ctxscale_shift_logit_bias_v0 = original


def _entries_by_q0(entries: Iterable[Tuple]) -> Dict[int, List[Tuple]]:
    out: Dict[int, List[Tuple]] = defaultdict(list)
    for entry in entries:
        out[int(entry[-2]) if len(entry) == 4 else int(entry[1])].append(entry)
    return out


def assemble_effective_from_s4(s4_entries: Sequence[Tuple[int, int, torch.Tensor]], t: int) -> torch.Tensor:
    if not s4_entries:
        raise RuntimeError("Missing S4post_postclamp capture entries")
    b = int(s4_entries[0][2].shape[0])
    full = torch.zeros((b, t, t), dtype=torch.float32)
    for _, q0, chunk in s4_entries:
        q_len = int(chunk.shape[1])
        full[:, int(q0) : int(q0) + q_len, :] = chunk.float()
    return full


def collect_layer_amplitudes(layer: Any, t: int) -> List[Dict[str, Any]]:
    cap = getattr(layer, "_pat234_cap", None)
    if cap is None:
        raise RuntimeError("Layer capture dict is not set")
    # S2_postp99 was removed from path_attn.py along with the p99 clamp step
    # (commit d339c6292a, 2026-07-31 13:33) -- basis_table now has no clamp
    # between S1_postnorm and the per-scale gain multiply, so S1_postnorm is
    # the current equivalent of "post-p99" (identity when clamp is absent).
    s2 = cap.get("S2_postp99", []) or cap.get("S1_postnorm", [])
    s3 = cap.get("S3_postgain", [])
    s4pre = cap.get("S4pre_preclamp", [])
    s4post = cap.get("S4post_postclamp", [])
    if not s2 or not s3 or not s4pre or not s4post:
        raise RuntimeError(
            f"Missing capture entries for layer {getattr(layer, 'layer_idx', '?')}: "
            f"S2={len(s2)} S3={len(s3)} S4pre={len(s4pre)} S4post={len(s4post)}"
        )
    pi_full = getattr(layer, "_last_ctxscale_router_prob", None)
    if pi_full is None:
        raise RuntimeError(f"Missing _last_ctxscale_router_prob for layer {getattr(layer, 'layer_idx', '?')}")
    pi_scale_all = pi_full.detach().float().cpu()
    if pi_scale_all.dim() == 4:
        pi_scale_all = pi_scale_all[:, :, 0, :]
    elif pi_scale_all.dim() != 3:
        raise RuntimeError(f"Unexpected router prob shape {tuple(pi_scale_all.shape)}")

    s2_by_q: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
    for _lid, scale_idx, q0, basis in s2:
        s2_by_q[int(q0)][int(scale_idx)] = basis.float()
    s3_by_q: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for _lid, _scale_idx, q0, contrib in s3:
        s3_by_q[int(q0)].append(contrib.float())
    s4pre_by_q = {int(q0): chunk.float() for _lid, q0, chunk in s4pre}
    s4post_by_q = {int(q0): chunk.float() for _lid, q0, chunk in s4post}

    rows = []
    k_expected = int(getattr(layer, "wavelet_ctxscale_k"))
    for q0 in sorted(s4post_by_q):
        basis_by_scale = s2_by_q.get(q0)
        if basis_by_scale is None or len(basis_by_scale) != k_expected:
            raise RuntimeError(f"Layer {getattr(layer, 'layer_idx', '?')} q0={q0}: expected {k_expected} S2 scales")
        scale_order = sorted(basis_by_scale)
        basis = torch.stack([basis_by_scale[i] for i in scale_order], dim=-1)
        pi = pi_scale_all[:, q0 : q0 + basis.shape[1], :]
        if list(scale_order) != list(range(k_expected)):
            pi = pi.index_select(-1, torch.tensor(scale_order, dtype=torch.long))
        mixture_pre = torch.stack(s3_by_q[q0], dim=0).sum(dim=0)
        mixture_post = mixture_pre * float(getattr(layer, "multiscale_sum_scale"))
        amp_component = component_weighted_upper_bound_amp(basis, pi, q0=q0)
        amp_pre = causal_centered_rms_amp(mixture_pre, q0=q0)
        amp_post = causal_centered_rms_amp(mixture_post, q0=q0)
        amp_s4pre = causal_centered_rms_amp(s4pre_by_q[q0], q0=q0)
        amp_eff = causal_centered_rms_amp(s4post_by_q[q0], q0=q0)
        peak_eff, valley_eff = causal_centered_peak_valley(s4post_by_q[q0], q0=q0)
        if not bool((amp_pre <= amp_component + 5e-5).all()):
            diff = (amp_pre - amp_component).max().item()
            raise AssertionError(f"Convexity check failed in layer {getattr(layer, 'layer_idx', '?')} q0={q0}: max diff {diff}")
        for b in range(int(amp_eff.shape[0])):
            rows.append(
                {
                    "layer": int(getattr(layer, "layer_idx", 0) or 0),
                    "sample_in_batch": b,
                    "q0": int(q0),
                    "component_weighted_amp": amp_component[b].numpy().astype(np.float32),
                    "mixture_pre_amp": amp_pre[b].numpy().astype(np.float32),
                    "mixture_post_amp": amp_post[b].numpy().astype(np.float32),
                    "s4pre_amp": amp_s4pre[b].numpy().astype(np.float32),
                    "effective_amp": amp_eff[b].numpy().astype(np.float32),
                    "effective_peak": peak_eff[b].numpy().astype(np.float32),
                    "effective_valley": valley_eff[b].numpy().astype(np.float32),
                }
            )
    return rows


def summarize_values(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def forward_collect(
    loaded: LoadedModel,
    samples: torch.Tensor,
    device: str,
    batch_size: int,
    cross_check: bool = False,
) -> Tuple[pd.DataFrame, Optional[float]]:
    all_rows: List[Dict[str, Any]] = []
    max_cross_diff = None
    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            input_ids = samples[start : start + int(batch_size)].to(device)
            _set_capture(loaded.layers, True)
            if cross_check and start == 0:
                with wrapped_logits_capture(loaded.layers) as logit_records:
                    loaded.model(input_ids=input_ids, labels=None, use_cache=False)
                diffs = []
                for layer in loaded.layers:
                    lid = int(getattr(layer, "layer_idx", 0) or 0)
                    if lid not in logit_records or not logit_records[lid]:
                        continue
                    e_base, e_wav = logit_records[lid][0]
                    s4 = assemble_effective_from_s4(layer._pat234_cap.get("S4post_postclamp", []), t=input_ids.shape[1])
                    delta = e_wav - e_base
                    diffs.append(float((delta - s4.unsqueeze(1)).abs().max().item()))
                max_cross_diff = max(diffs) if diffs else float("nan")
            else:
                loaded.model(input_ids=input_ids, labels=None, use_cache=False)
            for layer in loaded.layers:
                layer_rows = collect_layer_amplitudes(layer, t=input_ids.shape[1])
                for r in layer_rows:
                    r["sample"] = int(start + r.pop("sample_in_batch"))
                all_rows.extend(layer_rows)
            _clear_capture(loaded.layers)
    return pd.DataFrame(all_rows), max_cross_diff


def explode_query_arrays(raw_df: pd.DataFrame, meta_cols: Sequence[str]) -> pd.DataFrame:
    records = []
    metric_cols = [
        "component_weighted_amp",
        "mixture_pre_amp",
        "mixture_post_amp",
        "s4pre_amp",
        "effective_amp",
        "effective_peak",
        "effective_valley",
    ]
    for _, row in raw_df.iterrows():
        n = len(row["effective_amp"])
        for i in range(n):
            rec = {c: row[c] for c in meta_cols}
            rec["query"] = int(row["q0"] + i)
            for m in metric_cols:
                rec[m] = float(row[m][i])
            records.append(rec)
    return pd.DataFrame.from_records(records)


def aggregate(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = ["checkpoint_step", "model_variant", "num_samples", "eval_length", "layer", "sample", "q0"]
    flat = explode_query_arrays(raw_df, meta)
    by_layer_rows = []
    for keys, grp in flat.groupby(["checkpoint_step", "model_variant", "num_samples", "eval_length", "layer"], sort=True):
        d = dict(zip(["checkpoint_step", "model_variant", "num_samples", "eval_length", "layer"], keys))
        for m in ("component_weighted_amp", "mixture_pre_amp", "mixture_post_amp", "effective_amp", "effective_peak", "effective_valley"):
            stats = summarize_values(grp[m].to_numpy())
            d[f"{m}_mean"] = stats["mean"]
            d[f"{m}_median"] = stats["median"]
            d[f"{m}_p25"] = stats["p25"]
            d[f"{m}_p75"] = stats["p75"]
        d["cancellation_ratio_mean"] = float(d["mixture_pre_amp_mean"] / max(d["component_weighted_amp_mean"], 1e-12))
        d["peak_to_valley_range_mean"] = float(d["effective_peak_mean"] - d["effective_valley_mean"])
        by_layer_rows.append(d)
    by_layer = pd.DataFrame(by_layer_rows)

    summary_rows = []
    for keys, grp in by_layer.groupby(["checkpoint_step", "model_variant", "num_samples", "eval_length"], sort=True):
        d = dict(zip(["checkpoint_step", "model_variant", "num_samples", "eval_length"], keys))
        for m in ("component_weighted_amp", "mixture_pre_amp", "mixture_post_amp", "effective_amp", "effective_peak", "effective_valley"):
            d[f"{m}_mean"] = float(grp[f"{m}_mean"].mean())
        eff_flat = flat[
            (flat["checkpoint_step"] == d["checkpoint_step"])
            & (flat["model_variant"] == d["model_variant"])
            & (flat["eval_length"] == d["eval_length"])
        ]["effective_amp"].to_numpy()
        eff_stats = summarize_values(eff_flat)
        d["effective_amp_median"] = eff_stats["median"]
        d["effective_amp_p25"] = eff_stats["p25"]
        d["effective_amp_p75"] = eff_stats["p75"]
        d["cancellation_ratio_mean"] = float(d["mixture_pre_amp_mean"] / max(d["component_weighted_amp_mean"], 1e-12))
        d["peak_to_valley_range_mean"] = float(d["effective_peak_mean"] - d["effective_valley_mean"])
        summary_rows.append(d)
    summary = pd.DataFrame(summary_rows)
    for col in ("k4_pre_vs_k1", "k4_post_vs_k1", "k4_effective_vs_k1", "k4_peak_vs_k1", "k4_valley_vs_k1", "k4_range_vs_k1"):
        summary[col] = np.nan
    for (step, length), grp in summary.groupby(["checkpoint_step", "eval_length"]):
        k1 = grp[grp["model_variant"] == "K1"]
        k4_idx = grp[grp["model_variant"] == "K4"].index
        if k1.empty or len(k4_idx) == 0:
            continue
        k1_pre = float(k1.iloc[0]["mixture_pre_amp_mean"])
        k1_post = float(k1.iloc[0]["mixture_post_amp_mean"])
        k1_eff = float(k1.iloc[0]["effective_amp_mean"])
        k1_peak = float(k1.iloc[0]["effective_peak_mean"])
        k1_valley = float(k1.iloc[0]["effective_valley_mean"])
        k1_range = float(k1.iloc[0]["peak_to_valley_range_mean"])
        for idx in k4_idx:
            summary.loc[idx, "k4_pre_vs_k1"] = float(summary.loc[idx, "mixture_pre_amp_mean"] / max(k1_pre, 1e-12))
            summary.loc[idx, "k4_post_vs_k1"] = float(summary.loc[idx, "mixture_post_amp_mean"] / max(k1_post, 1e-12))
            summary.loc[idx, "k4_effective_vs_k1"] = float(summary.loc[idx, "effective_amp_mean"] / max(k1_eff, 1e-12))
            summary.loc[idx, "k4_peak_vs_k1"] = float(summary.loc[idx, "effective_peak_mean"] / max(k1_peak, 1e-12))
            summary.loc[idx, "k4_valley_vs_k1"] = float(summary.loc[idx, "effective_valley_mean"] / min(k1_valley, -1e-12))
            summary.loc[idx, "k4_range_vs_k1"] = float(summary.loc[idx, "peak_to_valley_range_mean"] / max(k1_range, 1e-12))
    columns = [
        "checkpoint_step",
        "model_variant",
        "num_samples",
        "eval_length",
        "component_weighted_amp_mean",
        "mixture_pre_amp_mean",
        "mixture_post_amp_mean",
        "effective_amp_mean",
        "effective_amp_median",
        "effective_amp_p25",
        "effective_amp_p75",
        "effective_peak_mean",
        "effective_valley_mean",
        "peak_to_valley_range_mean",
        "cancellation_ratio_mean",
        "k4_pre_vs_k1",
        "k4_post_vs_k1",
        "k4_effective_vs_k1",
        "k4_peak_vs_k1",
        "k4_valley_vs_k1",
        "k4_range_vs_k1",
    ]
    return summary[columns].sort_values(["eval_length", "checkpoint_step", "model_variant"]), by_layer.sort_values(
        ["eval_length", "checkpoint_step", "model_variant", "layer"]
    )


def plot_summary(summary: pd.DataFrame, output_dir: Path, eval_length: int, num_samples: int) -> None:
    import matplotlib.pyplot as plt

    sub = summary[summary["eval_length"] == int(eval_length)].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [
        ("K1", "effective_amp_mean", "K1 effective bias", "-"),
        ("K4", "mixture_pre_amp_mean", "K4 mixture_pre", "-"),
        ("K4", "mixture_post_amp_mean", "K4 mixture_post", "-"),
        ("K4", "effective_amp_mean", "K4 effective bias", "-"),
        ("K4", "component_weighted_amp_mean", "K4 component-weighted upper bound", "--"),
    ]
    y_vals = []
    for variant, col, label, style in lines:
        g = sub[sub["model_variant"] == variant].sort_values("checkpoint_step")
        if g.empty:
            continue
        ax.plot(g["checkpoint_step"], g[col], linestyle=style, marker="o", label=label)
        y_vals.extend([float(x) for x in g[col].dropna().tolist() if float(x) > 0])
        if variant == "K1" and col == "effective_amp_mean":
            ax.fill_between(g["checkpoint_step"], g["effective_amp_p25"], g["effective_amp_p75"], alpha=0.12)
        if variant == "K4" and col == "effective_amp_mean":
            ax.fill_between(g["checkpoint_step"], g["effective_amp_p25"], g["effective_amp_p75"], alpha=0.12)
    if y_vals and max(y_vals) / max(min(y_vals), 1e-12) > 100:
        ax.set_yscale("log")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Mean per-query causal-centered RMS")
    ax.set_title(f"K1 vs K4 QWAB amplitude, eval_length={eval_length}, num_samples={num_samples}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"k1_vs_k4_checkpoint_amp_L{eval_length}.png", dpi=150)
    if int(eval_length) == int(sub["eval_length"].min()):
        fig.savefig(output_dir / "k1_vs_k4_checkpoint_amp.png", dpi=150)
    plt.close(fig)

    k4 = sub[sub["model_variant"] == "K4"].sort_values("checkpoint_step")
    if not k4.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in ("k4_pre_vs_k1", "k4_post_vs_k1", "k4_effective_vs_k1"):
            ax.plot(k4["checkpoint_step"], k4[col], marker="o", label=col)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("K4 / K1 ratio")
        ax.set_title(f"K4/K1 amplitude ratios, eval_length={eval_length}, num_samples={num_samples}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"k1_vs_k4_checkpoint_amp_ratios_L{eval_length}.png", dpi=150)
        if int(eval_length) == int(sub["eval_length"].min()):
            fig.savefig(output_dir / "k1_vs_k4_checkpoint_amp_ratios.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, style in (("K1", "-"), ("K4", "--")):
        g = sub[sub["model_variant"] == variant].sort_values("checkpoint_step")
        if g.empty:
            continue
        ax.plot(g["checkpoint_step"], g["effective_peak_mean"], linestyle=style, marker="^", color="tab:red", label=f"{variant} peak")
        ax.plot(g["checkpoint_step"], g["effective_valley_mean"], linestyle=style, marker="v", color="tab:blue", label=f"{variant} valley")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Causal-centered peak / valley (post-gate effective bias)")
    ax.set_title(f"K1 vs K4 post-gate effective-bias peak/valley, eval_length={eval_length}, num_samples={num_samples}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"k1_vs_k4_peak_valley_L{eval_length}.png", dpi=150)
    if int(eval_length) == int(sub["eval_length"].min()):
        fig.savefig(output_dir / "k1_vs_k4_peak_valley.png", dpi=150)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, files_created: Sequence[str], warnings: Sequence[str]) -> None:
    lines = ["# K1 vs K4 QWAB Wavelet-Bias Amplitude Analysis", ""]
    if summary.empty:
        lines.append("No summary rows were produced.")
    else:
        for length in sorted(summary["eval_length"].unique()):
            sub = summary[summary["eval_length"] == length]
            k4 = sub[sub["model_variant"] == "K4"].sort_values("checkpoint_step")
            lines += [f"## Eval Length {int(length)}", ""]
            if k4.empty:
                lines.append("No K4 rows were available.")
                continue
            late = k4.iloc[-1]
            early = k4.iloc[0]
            comp = float(late["component_weighted_amp_mean"])
            pre = float(late["mixture_pre_amp_mean"])
            post = float(late["mixture_post_amp_mean"])
            eff = float(late["effective_amp_mean"])
            lines.append(
                f"At checkpoint {int(late['checkpoint_step'])}, K4 mixture_pre mean amplitude is {pre:.6g} "
                f"versus component-weighted upper bound {comp:.6g} "
                f"(cancellation ratio {float(late['cancellation_ratio_mean']):.6g})."
            )
            lines.append(
                f"K4/K1 ratios at that checkpoint: pre={float(late['k4_pre_vs_k1']):.6g}, "
                f"post={float(late['k4_post_vs_k1']):.6g}, effective={float(late['k4_effective_vs_k1']):.6g}."
            )
            shrink_text = "near K1" if abs(float(late["k4_post_vs_k1"]) - 1.0) <= 0.15 else "not near K1"
            lines.append(
                f"The sqrt(4) factor gives K4 mixture_post {post:.6g}, which is {shrink_text} by the 15% heuristic."
            )
            if abs(float(late["k4_effective_vs_k1"]) - float(late["k4_post_vs_k1"])) <= 0.15:
                comp_text = "does not strongly compensate"
            else:
                comp_text = "appears to compensate materially"
            lines.append(
                f"Final effective amplitude is {eff:.6g}; comparing post ratio to effective ratio suggests g0/g_layer {comp_text} for the sqrt factor."
            )
            lines.append(
                f"Early checkpoint {int(early['checkpoint_step'])}: pre/post/effective ratios are "
                f"{float(early['k4_pre_vs_k1']):.6g}/{float(early['k4_post_vs_k1']):.6g}/"
                f"{float(early['k4_effective_vs_k1']):.6g}. Late checkpoint {int(late['checkpoint_step'])}: "
                f"{float(late['k4_pre_vs_k1']):.6g}/{float(late['k4_post_vs_k1']):.6g}/"
                f"{float(late['k4_effective_vs_k1']):.6g}."
            )
            if float(late["k4_effective_vs_k1"]) < 0.85:
                judgment = "weaker-perturbation"
            elif abs(float(late["k4_effective_vs_k1"]) - 1.0) <= 0.15:
                judgment = "amplitude-matching"
            else:
                judgment = "different-multiscale-shape"
            lines.append(
                f"Given only these amplitude measurements, K4's earlier downstream benefit is most consistent with a {judgment} explanation."
            )
            lines.append("Conclusion: the amplitude data constrain the mechanism interpretation but do not by themselves prove a downstream causal pathway.")
            lines.append("")
    lines += ["## Files", ""]
    for f in files_created:
        lines.append(f"- {f}")
    lines += ["", "This pipeline does not write under `fla/`; PaTHAttention is only read and wrapped on loaded instances at runtime.", ""]
    lines += ["## Caveats", ""]
    lines.append("- The analysis assumes K1/K4 checkpoints are comparable by step and training setup.")
    lines.append("- Full numerical results depend on the GPU runtime, local checkpoint availability, and dataset cache state.")
    if warnings:
        lines.append("- Warnings observed during analysis:")
        for w in warnings:
            lines.append(f"  - {w}")
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_variant(loaded: LoadedModel, expected_k: int, variant: str) -> None:
    for layer in loaded.layers:
        router_mode = str(getattr(layer, "wavelet_router_sigmoid_mode", "")).strip().lower()
        if router_mode != "with_null":
            raise AssertionError(
                f"{variant} layer {getattr(layer, 'layer_idx', '?')} wavelet_router_sigmoid_mode={router_mode!r}, "
                "expected 'with_null'"
            )
        wavelet_mode = str(getattr(getattr(layer, "config", None), "wavelet_mode", getattr(layer, "wavelet_mode", "")))
        if wavelet_mode != "logit_bias_ctxscale_shift_v0":
            raise AssertionError(
                f"{variant} layer {getattr(layer, 'layer_idx', '?')} wavelet_mode={wavelet_mode!r}, "
                "expected 'logit_bias_ctxscale_shift_v0'"
            )
        k = int(getattr(layer, "wavelet_ctxscale_k"))
        if k != expected_k:
            raise AssertionError(f"{variant} layer {getattr(layer, 'layer_idx', '?')} wavelet_ctxscale_k={k}, expected {expected_k}")
        if bool(getattr(layer, "wavelet_ctxscale_use_head_gate", False)):
            raise AssertionError(f"{variant} layer {getattr(layer, 'layer_idx', '?')} uses head gate; unsupported for S4 capture")
    if expected_k == 1:
        for layer in loaded.layers:
            factor = float(getattr(layer, "multiscale_sum_scale"))
            if abs(factor - 1.0) > 1e-6:
                raise AssertionError(f"K1 multiscale_sum_scale is {factor}, expected 1.0; stopping.")
    if expected_k == 4:
        first = loaded.layers[0]
        print(
            f"K4 multiscale_norm_requested={getattr(first, 'multiscale_norm_requested', None)} "
            f"multiscale_sum_scale={float(getattr(first, 'multiscale_sum_scale'))}"
        )
        for layer in loaded.layers:
            factor = float(getattr(layer, "multiscale_sum_scale"))
            if abs(factor - 0.5) >= 1e-6:
                raise AssertionError(
                    f"K4 actual multiscale_sum_scale is {factor}, not 0.5. "
                    "Stopping immediately to avoid misleading results."
                )


def run_sanity(args: argparse.Namespace, common_steps: Sequence[int], output_dir: Path) -> None:
    step = int(common_steps[0])
    n = min(8, int(args.num_samples))
    print(f"Running sanity checks on checkpoint-{step} with {n} samples")
    k1 = load_model_for_analysis(Path(args.k1_run) / f"checkpoint-{step}", Path(args.k1_run), args.device, args.dtype)
    k4 = load_model_for_analysis(Path(args.k4_run) / f"checkpoint-{step}", Path(args.k4_run), args.device, args.dtype)
    validate_variant(k1, 1, "K1")
    validate_variant(k4, 4, "K4")
    samples = build_eval_samples(k1.tokenizer, int(args.eval_lengths[0]), n, int(args.seed), cache_dir=args.cache_dir)
    k1_raw_a, diff_a = forward_collect(k1, samples, args.device, args.batch_size, cross_check=True)
    k1_raw_b, diff_b = forward_collect(k1, samples, args.device, args.batch_size, cross_check=True)
    for metric in ("mixture_pre_amp", "mixture_post_amp", "effective_amp"):
        a = np.concatenate(k1_raw_a[metric].to_numpy())
        b = np.concatenate(k1_raw_b[metric].to_numpy())
        if not np.allclose(a, b, rtol=1e-4, atol=1e-5):
            raise AssertionError(f"Repeatability check failed for K1 {metric}")
    k1_pre = np.concatenate(k1_raw_a["mixture_pre_amp"].to_numpy())
    k1_post = np.concatenate(k1_raw_a["mixture_post_amp"].to_numpy())
    if not np.allclose(k1_pre, k1_post, rtol=1e-5, atol=1e-6):
        raise AssertionError("K1 mixture_pre and mixture_post amplitudes differ despite multiscale_sum_scale=1.0")
    if np.nanmax([diff_a, diff_b]) > float(args.cross_check_atol):
        raise AssertionError(f"K1 logits delta cross-check failed: diffs {diff_a}, {diff_b}")
    k4_raw, diff_k4 = forward_collect(k4, samples, args.device, args.batch_size, cross_check=True)
    if diff_k4 is not None and diff_k4 > float(args.cross_check_atol):
        raise AssertionError(f"K4 logits delta cross-check failed: max diff {diff_k4}")
    pre = np.concatenate(k4_raw["mixture_pre_amp"].to_numpy())
    post = np.concatenate(k4_raw["mixture_post_amp"].to_numpy())
    if not np.allclose(post, 0.5 * pre, rtol=2e-3, atol=1e-5):
        raise AssertionError("K4 Amp(mixture_post) is not approximately 0.5 * Amp(mixture_pre)")
    x = torch.tensor([[[1.0, 3.0, 999.0], [2.0, 4.0, 6.0]]])
    amp0 = causal_centered_rms_amp(x, q0=0)
    x_shift = x.clone()
    x_shift[:, 0, :] += 123.0
    amp1 = causal_centered_rms_amp(x_shift, q0=0)
    if not torch.allclose(amp0[:, 0], amp1[:, 0], atol=1e-6):
        raise AssertionError("Translation-invariance pure-function sanity check failed")
    print("Sanity checks passed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k1_run", required=True)
    p.add_argument("--k4_run", required=True)
    p.add_argument("--eval_length", type=int, default=512, help="Backward-compatible single eval length.")
    p.add_argument("--eval_lengths", type=int, nargs="+", default=None)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="analysis_outputs/k1_k4_wavelet_amp/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--sanity_check_only", action="store_true")
    p.add_argument("--skip_sanity_check", action="store_true")
    p.add_argument("--cross_check_atol", type=float, default=2e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.eval_lengths = args.eval_lengths or [args.eval_length]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using dtype={args.dtype}, device={args.device}, eval_lengths={args.eval_lengths}")

    k1_ckpts = discover_checkpoints(Path(args.k1_run))
    k4_ckpts = discover_checkpoints(Path(args.k4_run))
    common_steps = sorted(set(k1_ckpts).intersection(k4_ckpts))
    if not common_steps:
        raise RuntimeError("No common checkpoint-* steps found between K1 and K4 runs")
    only_k1 = sorted(set(k1_ckpts) - set(k4_ckpts))
    only_k4 = sorted(set(k4_ckpts) - set(k1_ckpts))
    if only_k1:
        print(f"Skipping K1-only checkpoints: {only_k1}")
    if only_k4:
        print(f"Skipping K4-only checkpoints: {only_k4}")

    first_k1 = load_model_for_analysis(k1_ckpts[common_steps[0]], Path(args.k1_run), args.device, args.dtype)
    first_k4 = load_model_for_analysis(k4_ckpts[common_steps[0]], Path(args.k4_run), args.device, args.dtype)
    validate_variant(first_k1, 1, "K1")
    validate_variant(first_k4, 4, "K4")
    manifest = {
        "k1_run": str(Path(args.k1_run)),
        "k4_run": str(Path(args.k4_run)),
        "common_steps": common_steps,
        "skipped_k1_only_steps": only_k1,
        "skipped_k4_only_steps": only_k4,
        "eval_lengths": args.eval_lengths,
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "dtype": args.dtype,
        "device": args.device,
        "K1": first_k1.manifest,
        "K4": first_k4.manifest,
    }
    (output_dir / "run_config_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    del first_k1, first_k4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.sanity_check_only:
        run_sanity(args, common_steps, output_dir)
        return
    if not args.skip_sanity_check:
        run_sanity(args, common_steps, output_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_parts = []
    warnings = []
    for eval_length in args.eval_lengths:
        sample_cache = None
        for step in common_steps:
            print(f"Analyzing checkpoint-{step}, eval_length={eval_length}")
            for variant, ckpts, run_dir, expected_k in (
                ("K1", k1_ckpts, Path(args.k1_run), 1),
                ("K4", k4_ckpts, Path(args.k4_run), 4),
            ):
                loaded = load_model_for_analysis(ckpts[step], run_dir, args.device, args.dtype)
                validate_variant(loaded, expected_k, variant)
                if sample_cache is None:
                    sample_cache = build_eval_samples(
                        loaded.tokenizer, int(eval_length), int(args.num_samples), int(args.seed), cache_dir=args.cache_dir
                    )
                raw_df, cross_diff = forward_collect(
                    loaded, sample_cache, args.device, int(args.batch_size), cross_check=True
                )
                if cross_diff is not None and cross_diff > float(args.cross_check_atol):
                    msg = f"{variant} checkpoint-{step} logits delta cross-check max diff {cross_diff:.6g}"
                    print(f"WARNING: {msg}")
                    warnings.append(msg)
                raw_df["checkpoint_step"] = int(step)
                raw_df["model_variant"] = variant
                raw_df["num_samples"] = int(args.num_samples)
                raw_df["eval_length"] = int(eval_length)
                raw_parts.append(raw_df)
                del loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raw_all = pd.concat(raw_parts, ignore_index=True)
    summary, by_layer = aggregate(raw_all)
    summary.to_csv(output_dir / "checkpoint_amp_summary.csv", index=False)
    by_layer.to_csv(output_dir / "checkpoint_amp_by_layer.csv", index=False)
    raw_path = output_dir / "checkpoint_amp_raw.parquet"
    try:
        raw_all.to_parquet(raw_path, index=False)
    except Exception as e:
        raw_path = output_dir / "checkpoint_amp_raw.pkl"
        raw_all.to_pickle(raw_path)
        warnings.append(f"Parquet write failed ({e}); wrote {raw_path} instead")
    for eval_length in args.eval_lengths:
        plot_summary(summary, output_dir, int(eval_length), int(args.num_samples))
    files = [
        str(Path(__file__).resolve()),
        str((Path(__file__).resolve().parent / "test_wavelet_amp_functions.py")),
        str(output_dir / "run_config_manifest.json"),
        str(output_dir / "checkpoint_amp_summary.csv"),
        str(output_dir / "checkpoint_amp_by_layer.csv"),
        str(raw_path),
        str(output_dir / "k1_vs_k4_checkpoint_amp.png"),
        str(output_dir / "k1_vs_k4_checkpoint_amp_ratios.png"),
        str(output_dir / "analysis_report.md"),
    ]
    write_report(summary, output_dir, files, warnings)
    print(f"Done. Outputs written under {output_dir}")


if __name__ == "__main__":
    main()
