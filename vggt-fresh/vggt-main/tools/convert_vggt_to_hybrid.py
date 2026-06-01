#!/usr/bin/env python3
"""CLI to convert an attention-only VGGT checkpoint into a hybrid VGGT checkpoint.

The converter does three things:
1) resolves which global layers stay as attention and which are replaced by Mamba2,
2) instantiates the hybrid model,
3) copies directly compatible weights and initializes replaced Mamba blocks from the
   source attention block using stable heuristics.

It can operate either from a training YAML file or from an explicit model target + kwargs.
The YAML path is the most practical option for real VGGT checkpoints.

Examples
--------
# Real VGGT use: explicit attention layers.
python tools/convert_vggt_to_hybrid.py \
  --src-ckpt /path/to/vggt_base.pt \
  --dst-ckpt /path/to/vggt_hybrid_init.pt \
  --src-config training/config/paper_hybrid_multidata_h200.yaml \
  --keep-global-attn-indices 4 7 11 17 20 23 \
  --emit-config /path/to/paper_hybrid_init.yaml

# HALO-like layer selection from per-layer metrics.
python tools/convert_vggt_to_hybrid.py \
  --src-ckpt /path/to/vggt_base.pt \
  --dst-ckpt /path/to/vggt_hybrid_init.pt \
  --src-config training/config/paper_hybrid_multidata_h200.yaml \
  --layer-metrics-json /path/to/layer_metrics.json \
  --keep-topk 6 \
  --emit-config /path/to/paper_hybrid_init.yaml

# Tiny smoke test with Aggregator.
python tools/convert_vggt_to_hybrid.py \
  --src-ckpt tiny_attention.pt \
  --dst-ckpt tiny_hybrid.pt \
  --model-target vggt.models.aggregator.Aggregator \
  --model-kwargs-json '{"img_size": 28, "patch_size": 14, "embed_dim": 64, "depth": 4, "num_heads": 4, "patch_embed": "conv"}' \
  --keep-global-attn-indices 1
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_to_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an attention-only VGGT checkpoint into a hybrid checkpoint.")
    parser.add_argument("--src-ckpt", required=True, help="Source checkpoint path.")
    parser.add_argument("--dst-ckpt", required=True, help="Destination checkpoint path.")
    parser.add_argument(
        "--src-config",
        default=None,
        help="Optional YAML config path. The script reads model/checkpoint fields from this file and can emit an updated hybrid config.",
    )
    parser.add_argument(
        "--emit-config",
        default=None,
        help="Optional output YAML path. When set, the source config is copied and updated with the hybrid layout and resume checkpoint.",
    )
    parser.add_argument(
        "--model-target",
        default=None,
        help="Optional import path to the model class when you do not want to use --src-config, e.g. vggt.models.vggt.VGGT or vggt.models.aggregator.Aggregator.",
    )
    parser.add_argument(
        "--model-kwargs-json",
        default=None,
        help="Inline JSON string or JSON file path for model kwargs when --model-target is used or when you want to override YAML fields.",
    )

    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--keep-global-attn-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit list of global layer indices to keep as attention.",
    )
    layout.add_argument(
        "--replace-global-with-mamba2-indices",
        nargs="*",
        type=int,
        default=None,
        help="Explicit list of global layer indices to replace with Mamba2.",
    )
    layout.add_argument(
        "--halo-ratio",
        type=int,
        default=None,
        help="Keep every n-th global layer as attention. Example: 4 keeps layers 3,7,11,...",
    )
    layout.add_argument(
        "--layer-metrics-json",
        default=None,
        help=(
            "JSON file for HALO-like layer selection. Supported formats:\n"
            "  {\"0\": {\"recall_drop\": ..., \"csr_drop\": ...}, ...}\n"
            "  {\"layers\": {\"0\": {...}, ...}}\n"
            "  [{\"layer\": 0, \"recall_drop\": ..., \"csr_drop\": ...}, ...]"
        ),
    )
    parser.add_argument(
        "--keep-topk",
        type=int,
        default=None,
        help="How many global layers to keep as attention when --layer-metrics-json is used. Defaults to depth//4.",
    )
    parser.add_argument(
        "--layer-score-eps",
        type=float,
        default=1e-6,
        help="Numerical stabilizer used in the HALO-like score recall_drop / (csr_drop + eps).",
    )

    parser.add_argument("--mamba2-d-state", type=int, default=None, help="Override mamba2_d_state in the destination model.")
    parser.add_argument("--mamba2-d-conv", type=int, default=None, help="Override mamba2_d_conv in the destination model.")
    parser.add_argument("--mamba2-expand", type=int, default=None, help="Override mamba2_expand in the destination model.")
    parser.add_argument("--mamba-mlp-ratio", type=float, default=None, help="Override mamba_mlp_ratio in the destination model.")

    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON file for conversion metadata (selected layers, scores, load-state diagnostics).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if source weights cannot be loaded strictly into the target model.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve layout and print metadata without writing checkpoint/config files.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed diagnostics.")
    return parser.parse_args()


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _dump_yaml(path: str, data: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def _maybe_load_json_arg(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return {}
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def _import_symbol(target: str):
    module_name, symbol_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _extract_model_spec(args: argparse.Namespace) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    cfg = _load_yaml(args.src_config) if args.src_config else None

    if cfg is not None and isinstance(cfg.get("model"), dict):
        model_section = copy.deepcopy(cfg["model"])
    elif cfg is not None:
        model_section = copy.deepcopy(cfg)
    else:
        model_section = {}

    target = args.model_target or model_section.pop("_target_", None)
    if not target:
        raise ValueError("Could not resolve model target. Provide --src-config with model._target_ or use --model-target.")

    # Merge JSON overrides last so the CLI always wins.
    model_section.update(_maybe_load_json_arg(args.model_kwargs_json))
    model_section.pop("_target_", None)

    for cli_name, key in (
        ("mamba2_d_state", "mamba2_d_state"),
        ("mamba2_d_conv", "mamba2_d_conv"),
        ("mamba2_expand", "mamba2_expand"),
        ("mamba_mlp_ratio", "mamba_mlp_ratio"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            model_section[key] = value

    return target, model_section, cfg


def _instantiate_model(target: str, kwargs: Mapping[str, Any]):
    cls = _import_symbol(target)
    return cls(**dict(kwargs))


def _unwrap_checkpoint_model_state(raw_ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Optional[str]]:
    if isinstance(raw_ckpt, dict):
        if isinstance(raw_ckpt.get("model"), dict):
            return raw_ckpt["model"], dict(raw_ckpt), "model"
        if isinstance(raw_ckpt.get("state_dict"), dict):
            return raw_ckpt["state_dict"], dict(raw_ckpt), "state_dict"
        if all(isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
            return raw_ckpt, {}, None
    raise ValueError("Unsupported checkpoint format. Expected a raw state_dict or a dict containing 'model' or 'state_dict'.")


def _strip_prefix_if_majority(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return dict(state_dict)
    starts = sum(1 for k in state_dict if k.startswith(prefix))
    if starts >= max(1, int(0.8 * len(state_dict))):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return dict(state_dict)


def _normalize_checkpoint_prefixes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sd = dict(state_dict)
    # DDP / wrapper prefixes.
    for prefix in ("module.", "model."):
        sd = _strip_prefix_if_majority(sd, prefix)
    return sd


def _match_model_prefix(state_dict: Mapping[str, torch.Tensor], target_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    target_keys = set(target_model.state_dict().keys())
    sd = dict(state_dict)

    def overlap(keys: Iterable[str]) -> int:
        return sum(1 for k in keys if k in target_keys)

    candidates: List[Dict[str, torch.Tensor]] = [sd]
    if any(k.startswith("aggregator.") for k in sd):
        candidates.append({k[len("aggregator."):] if k.startswith("aggregator.") else k: v for k, v in sd.items()})
    if any(k.startswith("aggregator.") for k in target_keys) and not any(k.startswith("aggregator.") for k in sd):
        candidates.append({f"aggregator.{k}": v for k, v in sd.items()})

    best = max(candidates, key=lambda cand: overlap(cand.keys()))
    return best


def _find_mixer_container(model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    if hasattr(model, "aggregator"):
        return model.aggregator, "aggregator."
    if hasattr(model, "frame_blocks") and hasattr(model, "global_blocks"):
        return model, ""
    raise ValueError("Could not find the VGGT/Aggregator mixer container (frame_blocks/global_blocks).")


def _normalize_indices(indices: Sequence[int], depth: int, name: str) -> List[int]:
    values = sorted({int(i) for i in indices})
    for idx in values:
        if idx < 0 or idx >= depth:
            raise ValueError(f"{name} contains invalid layer index {idx} for depth={depth}")
    return values


def _resolve_layout_from_scores(metrics_path: str, depth: int, topk: Optional[int], eps: float) -> Tuple[List[int], Dict[str, float]]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("layers", payload)

    scores: Dict[str, float] = {}
    if isinstance(rows, dict):
        iterator = rows.items()
    else:
        iterator = ((str(item["layer"]), item) for item in rows)

    for layer_name, row in iterator:
        layer_idx = int(layer_name)
        if layer_idx < 0 or layer_idx >= depth:
            continue
        if isinstance(row, Mapping) and "score" in row:
            score = float(row["score"])
        elif isinstance(row, Mapping):
            recall_drop = row.get("recall_drop")
            csr_drop = row.get("csr_drop")
            if recall_drop is None and {"recall_teacher", "recall_replaced"} <= row.keys():
                recall_drop = float(row["recall_teacher"]) - float(row["recall_replaced"])
            if csr_drop is None and {"csr_teacher", "csr_replaced"} <= row.keys():
                csr_drop = float(row["csr_teacher"]) - float(row["csr_replaced"])
            if recall_drop is None or csr_drop is None:
                raise ValueError(
                    "Each layer metric row must contain either 'score' or both 'recall_drop' and 'csr_drop' "
                    "(or teacher/replaced pairs from which the drops can be computed)."
                )
            score = float(recall_drop) / (float(csr_drop) + eps)
        else:
            raise ValueError("Unsupported layer metrics format.")
        scores[str(layer_idx)] = score

    if len(scores) < depth:
        missing = sorted(set(range(depth)) - {int(k) for k in scores})
        raise ValueError(f"Layer metrics do not cover every layer. Missing indices: {missing}")

    keep_topk = topk if topk is not None else max(1, depth // 4)
    keep = sorted(int(k) for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:keep_topk])
    return keep, scores


def _resolve_layout(
    depth: int,
    keep_global_attn_indices: Optional[Sequence[int]],
    replace_global_with_mamba2_indices: Optional[Sequence[int]],
    halo_ratio: Optional[int],
    layer_metrics_json: Optional[str],
    keep_topk: Optional[int],
    eps: float,
) -> Tuple[List[int], List[int], Optional[Dict[str, float]]]:
    if keep_global_attn_indices is not None:
        keep = _normalize_indices(keep_global_attn_indices, depth, "keep_global_attn_indices")
        return keep, [i for i in range(depth) if i not in keep], None
    if replace_global_with_mamba2_indices is not None:
        replace = _normalize_indices(replace_global_with_mamba2_indices, depth, "replace_global_with_mamba2_indices")
        return [i for i in range(depth) if i not in replace], replace, None
    if layer_metrics_json is not None:
        keep, scores = _resolve_layout_from_scores(layer_metrics_json, depth, keep_topk, eps)
        return keep, [i for i in range(depth) if i not in keep], scores
    if halo_ratio is not None and halo_ratio > 0:
        keep = [i for i in range(depth) if (i + 1) % halo_ratio == 0]
        return keep, [i for i in range(depth) if i not in keep], None
    # Default: keep everything as attention.
    keep = list(range(depth))
    return keep, [], None


def _block_substate(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix_len = len(prefix)
    return {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _extract_attention_qkv_and_proj(src_block_sd: Mapping[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {
        "q_w": None,
        "k_w": None,
        "v_w": None,
        "q_b": None,
        "k_b": None,
        "v_b": None,
        "proj_w": None,
        "proj_b": None,
    }

    qkv_w = src_block_sd.get("attn.qkv.weight")
    if qkv_w is not None and qkv_w.ndim == 2 and qkv_w.shape[0] % 3 == 0:
        out["q_w"], out["k_w"], out["v_w"] = qkv_w.chunk(3, dim=0)

    qkv_b = src_block_sd.get("attn.qkv.bias")
    if qkv_b is not None and qkv_b.ndim == 1 and qkv_b.shape[0] % 3 == 0:
        out["q_b"], out["k_b"], out["v_b"] = qkv_b.chunk(3, dim=0)

    out["proj_w"] = src_block_sd.get("attn.proj.weight")
    out["proj_b"] = src_block_sd.get("attn.proj.bias")
    return out


def _fit_rows(mat: torch.Tensor, rows: int) -> torch.Tensor:
    if mat.shape[0] == rows:
        return mat.clone()
    reps = math.ceil(rows / mat.shape[0])
    return mat.repeat(reps, *([1] * (mat.ndim - 1)))[:rows].clone()


def _fit_vector(vec: torch.Tensor, size: int) -> torch.Tensor:
    if vec.shape[0] == size:
        return vec.clone()
    reps = math.ceil(size / vec.shape[0])
    return vec.repeat(reps)[:size].clone()


def _identity_dwconv(weight: torch.Tensor, causal: bool) -> torch.Tensor:
    out = torch.zeros_like(weight)
    if weight.ndim == 3:
        pos = weight.shape[-1] - 1 if causal else weight.shape[-1] // 2
        out[:, 0, pos] = 1.0
    elif weight.ndim == 2:
        pos = weight.shape[-1] - 1 if causal else weight.shape[-1] // 2
        out[:, pos] = 1.0
    else:
        raise ValueError(f"Unsupported depth-wise conv weight shape: {tuple(weight.shape)}")
    return out


def _eye_like(weight: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(weight)
    if weight.ndim != 2:
        return out
    n = min(weight.shape[0], weight.shape[1])
    out[:n, :n] = torch.eye(n, dtype=weight.dtype, device=weight.device)
    return out


def _avg_merge_weight(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    out_dim, in_dim = int(shape[0]), int(shape[1])
    half = in_dim // 2
    out = torch.zeros(shape, dtype=dtype)
    n = min(out_dim, half)
    eye = torch.eye(n, dtype=dtype)
    out[:n, :n] = 0.5 * eye
    out[:n, half : half + n] = 0.5 * eye
    return out


def _tile_proj_for_outproj(proj_w: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    out_dim, in_dim = int(target_shape[0]), int(target_shape[1])
    base = proj_w
    if base.shape[0] != out_dim:
        base = _fit_rows(base, out_dim)
    reps = math.ceil(in_dim / base.shape[1])
    tiled = base.repeat(1, reps)[:, :in_dim].clone()
    return tiled / max(reps, 1)


def _seed_in_proj_weight(target: torch.Tensor, q: Optional[torch.Tensor], k: Optional[torch.Tensor], v: Optional[torch.Tensor]) -> torch.Tensor:
    if q is None or k is None or v is None:
        return target
    rows, cols = target.shape
    d = q.shape[0]
    out = torch.zeros_like(target)

    # Segment layout matches the PureTorch fallback in this repo: [x_inner, z, B, C, dt].
    # For other Mamba2 implementations, this still provides a stable bias towards the source
    # attention weights when an in_proj.weight exists.
    primary = torch.cat([v, 0.5 * (q + k)], dim=0)
    secondary = torch.cat([q, v], dim=0)

    remaining = rows
    start = 0
    first = min(primary.shape[0], remaining)
    out[start : start + first] = _fit_rows(primary, first)
    start += first
    remaining -= first

    second = min(secondary.shape[0], remaining)
    out[start : start + second] = _fit_rows(secondary, second)
    start += second
    remaining -= second

    # B/C style rows: use low-rank slices of K and Q.
    if remaining > 0:
        bc = torch.cat([k, q], dim=0)
        out[start : start + remaining] = _fit_rows(bc, remaining)
    return out


def _seed_in_proj_bias(target: torch.Tensor, q: Optional[torch.Tensor], k: Optional[torch.Tensor], v: Optional[torch.Tensor]) -> torch.Tensor:
    if q is None or k is None or v is None:
        return target
    rows = target.shape[0]
    out = torch.zeros_like(target)
    primary = torch.cat([v, 0.5 * (q + k)], dim=0)
    secondary = torch.cat([q, v], dim=0)

    start = 0
    first = min(primary.shape[0], rows - start)
    out[start : start + first] = _fit_vector(primary, first)
    start += first
    second = min(secondary.shape[0], rows - start)
    if second > 0:
        out[start : start + second] = _fit_vector(secondary, second)
        start += second
    if start < rows:
        out[start:] = 0
    return out


@torch.no_grad()
def _init_replaced_mamba_block_from_attention(
    target_block: torch.nn.Module,
    src_block_sd: Mapping[str, torch.Tensor],
    verbose: bool = False,
) -> Dict[str, List[str]]:
    dst_sd = target_block.state_dict()
    new_sd = {k: v.detach().clone() for k, v in dst_sd.items()}
    copied: List[str] = []
    seeded: List[str] = []

    # Directly compatible submodules.
    shared_prefixes = ("norm1.", "norm2.", "mlp.", "ls1.", "ls2.")
    for key, value in src_block_sd.items():
        if any(key.startswith(prefix) for prefix in shared_prefixes) and key in new_sd and new_sd[key].shape == value.shape:
            new_sd[key] = value.detach().clone().to(dtype=new_sd[key].dtype)
            copied.append(key)

    attn = _extract_attention_qkv_and_proj(src_block_sd)

    for key, value in list(new_sd.items()):
        if key.startswith("pos_inject."):
            new_sd[key] = torch.zeros_like(value)
            seeded.append(key)
        elif key == "local_conv.weight":
            new_sd[key] = _identity_dwconv(value, causal=False)
            seeded.append(key)
        elif key == "local_conv.bias":
            new_sd[key] = torch.zeros_like(value)
            seeded.append(key)
        elif key.endswith(".in_proj.weight"):
            new_sd[key] = _seed_in_proj_weight(value, attn["q_w"], attn["k_w"], attn["v_w"]).to(dtype=value.dtype)
            seeded.append(key)
        elif key.endswith(".in_proj.bias"):
            new_sd[key] = _seed_in_proj_bias(value, attn["q_b"], attn["k_b"], attn["v_b"]).to(dtype=value.dtype)
            seeded.append(key)
        elif key.endswith(".out_proj.weight") and attn["proj_w"] is not None:
            new_sd[key] = _tile_proj_for_outproj(attn["proj_w"], value.shape).to(dtype=value.dtype)
            seeded.append(key)
        elif key.endswith(".out_proj.bias") and attn["proj_b"] is not None:
            new_sd[key] = _fit_vector(attn["proj_b"], value.shape[0]).to(dtype=value.dtype)
            seeded.append(key)
        elif key.endswith(".conv1d.weight"):
            new_sd[key] = _identity_dwconv(value, causal=True)
            seeded.append(key)
        elif key.endswith(".conv1d.bias"):
            new_sd[key] = torch.zeros_like(value)
            seeded.append(key)
        elif key.endswith(".norm.weight"):
            new_sd[key] = torch.ones_like(value)
            seeded.append(key)
        elif key == "merge.weight":
            new_sd[key] = _avg_merge_weight(value.shape, value.dtype)
            seeded.append(key)
        elif key == "out_gate.0.weight":
            new_sd[key] = _eye_like(value)
            seeded.append(key)

    missing, unexpected = target_block.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[init-mamba] copied={len(copied)} seeded={len(seeded)} missing={missing} unexpected={unexpected}")
    return {"copied": copied, "seeded": seeded, "missing": list(missing), "unexpected": list(unexpected)}


def _update_config_for_hybrid(
    cfg: Dict[str, Any],
    dst_ckpt: str,
    keep_attn: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    if "model" not in out or not isinstance(out["model"], dict):
        out["model"] = {}
    model_cfg = out["model"]

    model_cfg["halo_ratio"] = 0
    model_cfg["keep_global_attn_indices"] = [int(i) for i in keep_attn]
    model_cfg["replace_global_with_mamba2_indices"] = None
    if args.mamba2_d_state is not None:
        model_cfg["mamba2_d_state"] = args.mamba2_d_state
    if args.mamba2_d_conv is not None:
        model_cfg["mamba2_d_conv"] = args.mamba2_d_conv
    if args.mamba2_expand is not None:
        model_cfg["mamba2_expand"] = args.mamba2_expand
    if args.mamba_mlp_ratio is not None:
        model_cfg["mamba_mlp_ratio"] = args.mamba_mlp_ratio

    if "checkpoint" not in out or not isinstance(out["checkpoint"], dict):
        out["checkpoint"] = {}
    out["checkpoint"]["resume_checkpoint_path"] = dst_ckpt

    if "optim" in out and isinstance(out["optim"], dict) and "frozen_module_names" in out["optim"]:
        # After conversion the optimizer state is no longer valid, so default to no freezing.
        out["optim"]["frozen_module_names"] = []

    exp_name = out.get("exp_name")
    if isinstance(exp_name, str) and exp_name:
        out["exp_name"] = f"{exp_name}_converted"
    return out


def main() -> None:
    args = _parse_args()
    _add_repo_to_path()

    target_path, model_kwargs, src_cfg = _extract_model_spec(args)

    # Build a temporary attention-only model to infer depth cleanly.
    base_kwargs = dict(model_kwargs)
    base_kwargs["halo_ratio"] = 0
    base_kwargs["keep_global_attn_indices"] = None
    base_kwargs["replace_global_with_mamba2_indices"] = None
    temp_model = _instantiate_model(target_path, base_kwargs)
    temp_container, _ = _find_mixer_container(temp_model)
    depth = int(getattr(temp_container, "depth", len(temp_container.global_blocks)))
    del temp_container
    del temp_model

    keep_attn, replace_with_mamba, layer_scores = _resolve_layout(
        depth=depth,
        keep_global_attn_indices=args.keep_global_attn_indices,
        replace_global_with_mamba2_indices=args.replace_global_with_mamba2_indices,
        halo_ratio=args.halo_ratio,
        layer_metrics_json=args.layer_metrics_json,
        keep_topk=args.keep_topk,
        eps=args.layer_score_eps,
    )

    hybrid_kwargs = dict(model_kwargs)
    hybrid_kwargs["halo_ratio"] = 0
    hybrid_kwargs["keep_global_attn_indices"] = list(keep_attn)
    hybrid_kwargs["replace_global_with_mamba2_indices"] = None
    if args.mamba2_d_state is not None:
        hybrid_kwargs["mamba2_d_state"] = args.mamba2_d_state
    if args.mamba2_d_conv is not None:
        hybrid_kwargs["mamba2_d_conv"] = args.mamba2_d_conv
    if args.mamba2_expand is not None:
        hybrid_kwargs["mamba2_expand"] = args.mamba2_expand
    if args.mamba_mlp_ratio is not None:
        hybrid_kwargs["mamba_mlp_ratio"] = args.mamba_mlp_ratio

    hybrid_model = _instantiate_model(target_path, hybrid_kwargs)
    container, container_prefix = _find_mixer_container(hybrid_model)

    raw_ckpt = torch.load(args.src_ckpt, map_location="cpu")
    src_state_raw, wrapper, wrapper_key = _unwrap_checkpoint_model_state(raw_ckpt)
    src_state = _normalize_checkpoint_prefixes(src_state_raw)
    src_state = _match_model_prefix(src_state, hybrid_model)

    missing, unexpected = hybrid_model.load_state_dict(src_state, strict=args.strict)

    block_reports: Dict[str, Dict[str, List[str]]] = {}
    for idx in replace_with_mamba:
        src_prefix = f"{container_prefix}global_blocks.{idx}."
        src_block_sd = _block_substate(src_state, src_prefix)
        if not src_block_sd:
            raise KeyError(f"Could not find source attention weights under prefix '{src_prefix}'.")
        report = _init_replaced_mamba_block_from_attention(container.global_blocks[idx], src_block_sd, verbose=args.verbose)
        block_reports[str(idx)] = report

    metadata: Dict[str, Any] = {
        "model_target": target_path,
        "keep_global_attn_indices": keep_attn,
        "replace_global_with_mamba2_indices": replace_with_mamba,
        "layer_scores": layer_scores,
        "missing_after_initial_load": list(missing),
        "unexpected_after_initial_load": list(unexpected),
        "replaced_block_reports": block_reports,
        "source_checkpoint": os.path.abspath(args.src_ckpt),
        "destination_checkpoint": os.path.abspath(args.dst_ckpt),
    }

    if args.verbose or args.dry_run:
        print(json.dumps(metadata, indent=2))

    if args.dry_run:
        return

    out_ckpt = wrapper if wrapper else {}
    if wrapper_key is None:
        out_ckpt = {"model": hybrid_model.state_dict(), "conversion_meta": metadata}
    else:
        out_ckpt = dict(wrapper)
        out_ckpt[wrapper_key] = hybrid_model.state_dict()
        # Optimizer state is not reusable after architecture conversion.
        out_ckpt.pop("optimizer", None)
        out_ckpt.pop("scaler", None)
        out_ckpt.pop("steps", None)
        out_ckpt["conversion_meta"] = metadata

    Path(args.dst_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.dst_ckpt)

    if args.emit_config is not None:
        if src_cfg is None:
            raise ValueError("--emit-config requires --src-config so the base YAML can be copied and updated.")
        updated_cfg = _update_config_for_hybrid(src_cfg, args.dst_ckpt, keep_attn, args)
        _dump_yaml(args.emit_config, updated_cfg)

    if args.metadata_out is not None:
        Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metadata_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if args.verbose:
        print(f"[done] wrote hybrid checkpoint to {args.dst_ckpt}")
        if args.emit_config is not None:
            print(f"[done] wrote hybrid config to {args.emit_config}")
        if args.metadata_out is not None:
            print(f"[done] wrote metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
