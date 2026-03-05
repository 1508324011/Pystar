from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np


def _rule_value(rule: Any, key: str, default: Any) -> Any:
    if isinstance(rule, Mapping):
        return rule.get(key, default)
    return getattr(rule, key, default)


def _rule_params(rule: Any) -> Dict[str, Any]:
    params = _rule_value(rule, "params", {})
    if params is None:
        return {}
    return dict(params)


def rule_quality_mean_lt(context: Dict[str, Any], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_scores = context["base_scores"]
    is_valid = context["is_valid"]
    threshold = float(params.get("threshold", context.get("quality_threshold", 0.5)))

    n_spots = len(is_valid)
    keep = np.zeros(n_spots, dtype=bool)
    penalty = np.zeros(n_spots, dtype=np.float32)

    if np.any(is_valid):
        mean_scores = np.mean(base_scores[is_valid], axis=1)
        local_keep = mean_scores < threshold
        keep[is_valid] = local_keep
        penalty[is_valid] = np.maximum(mean_scores - threshold, 0.0)
        n_pass = int(local_keep.sum())
    else:
        n_pass = 0

    details = {
        "threshold": threshold,
        "n_valid_input": int(np.sum(is_valid)),
        "n_pass": n_pass,
    }
    return keep, penalty, details


def rule_channel_margin_mean_gt(context: Dict[str, Any], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    norm_matrix = context["norm_matrix"]
    is_valid = context["is_valid"]
    threshold = float(params.get("threshold", 0.0))

    n_spots = len(is_valid)
    keep = np.zeros(n_spots, dtype=bool)
    penalty = np.zeros(n_spots, dtype=np.float32)

    if np.any(is_valid):
        sorted_vals = np.sort(norm_matrix, axis=2)
        max_vals = sorted_vals[:, :, -1]
        second_vals = sorted_vals[:, :, -2]
        margins = max_vals - second_vals
        mean_margin = np.mean(margins, axis=1)

        local_keep = mean_margin >= threshold
        keep[is_valid] = local_keep[is_valid]
        penalty[is_valid] = np.maximum(threshold - mean_margin[is_valid], 0.0)
        n_pass = int(local_keep[is_valid].sum())
        avg_margin_valid = float(mean_margin[is_valid].mean())
    else:
        n_pass = 0
        avg_margin_valid = 0.0

    details = {
        "threshold": threshold,
        "n_valid_input": int(np.sum(is_valid)),
        "n_pass": n_pass,
        "avg_margin_valid": avg_margin_valid,
    }
    return keep, penalty, details


def rule_end_base_pattern(context: Dict[str, Any], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    df = context["df"]
    validator = context["validator"]

    keep = df["barcode"].astype(str).apply(validator).to_numpy(dtype=bool)
    penalty = (~keep).astype(np.float32)

    details = {
        "n_total": int(len(df)),
        "n_pass": int(keep.sum()),
    }
    return keep, penalty, details


def rule_keep_all(context: Dict[str, Any], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_items = int(context["n_items"])
    keep = np.ones(n_items, dtype=bool)
    penalty = np.zeros(n_items, dtype=np.float32)
    details = {"n_pass": n_items}
    return keep, penalty, details


SPOT_RULES = {
    "quality_mean_lt": rule_quality_mean_lt,
    "channel_margin_mean_gt": rule_channel_margin_mean_gt,
    "keep_all": rule_keep_all,
}

BARCODE_RULES = {
    "end_base_pattern": rule_end_base_pattern,
    "keep_all": rule_keep_all,
}


def default_rules(quality_threshold: float) -> List[Dict[str, Any]]:
    return [
        {
            "name": "quality_mean_lt",
            "stage": "spot",
            "enabled": True,
            "hard": True,
            "weight": 1.0,
            "params": {"threshold": float(quality_threshold)},
        },
        {
            "name": "end_base_pattern",
            "stage": "barcode",
            "enabled": True,
            "hard": True,
            "weight": 1.0,
            "params": {},
        },
    ]


def apply_rule_pipeline(
    *,
    stage: str,
    n_items: int,
    rules: List[Any],
    context: Dict[str, Any],
    max_soft_penalty: float | None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if stage == "spot":
        registry = SPOT_RULES
    elif stage == "barcode":
        registry = BARCODE_RULES
    else:
        raise ValueError(f"Unknown stage: {stage}")

    hard_keep = np.ones(n_items, dtype=bool)
    soft_penalty = np.zeros(n_items, dtype=np.float32)
    reports: List[Dict[str, Any]] = []

    base_context = dict(context)
    base_context["n_items"] = n_items

    for rule in rules:
        name = str(_rule_value(rule, "name", "")).strip()
        if not name:
            continue
        enabled = bool(_rule_value(rule, "enabled", True))
        rule_stage = str(_rule_value(rule, "stage", stage))
        if not enabled or rule_stage != stage:
            continue

        hard = bool(_rule_value(rule, "hard", True))
        weight = float(_rule_value(rule, "weight", 1.0))
        params = _rule_params(rule)

        fn = registry.get(name)
        if fn is None:
            raise ValueError(f"Unknown decoding rule '{name}' at stage '{stage}'")

        before = int(hard_keep.sum())
        local_keep, local_penalty, details = fn(base_context, params)

        if local_keep.shape[0] != n_items:
            raise ValueError(
                f"Rule '{name}' returned invalid keep-mask length {local_keep.shape[0]} != {n_items}"
            )
        if local_penalty.shape[0] != n_items:
            raise ValueError(
                f"Rule '{name}' returned invalid penalty length {local_penalty.shape[0]} != {n_items}"
            )

        if hard:
            hard_keep &= local_keep
        soft_penalty += (weight * local_penalty).astype(np.float32)
        after = int(hard_keep.sum())

        reports.append(
            {
                "stage": stage,
                "name": name,
                "hard": hard,
                "weight": weight,
                "before": before,
                "after": after,
                "failed_by_rule": int((~local_keep).sum()),
                "params": params,
                "details": details,
            }
        )

    final_keep = hard_keep.copy()
    if max_soft_penalty is not None:
        penalty_gate = soft_penalty <= float(max_soft_penalty)
        before = int(final_keep.sum())
        final_keep &= penalty_gate
        after = int(final_keep.sum())
        reports.append(
            {
                "stage": stage,
                "name": "soft_penalty_gate",
                "hard": True,
                "weight": 1.0,
                "before": before,
                "after": after,
                "failed_by_rule": int((~penalty_gate).sum()),
                "params": {"max_soft_penalty": float(max_soft_penalty)},
                "details": {},
            }
        )

    return final_keep, soft_penalty, reports
