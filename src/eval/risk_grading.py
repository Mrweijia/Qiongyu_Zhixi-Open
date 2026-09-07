"""Configurable, versioned advisory grading for model outputs.

Risk labels must come from a named, versioned rule file rather than colours
picked in the UI, so every displayed grade can be traced back to the exact
thresholds, model version and input-data completeness that produced it.

The bands are internal advisory thresholds, not official AQI categories.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

import yaml


DEFAULT_RULE_PATH = Path(__file__).resolve().parents[2] / "configs" / "risk_grading.yaml"

_REQUIRED_KEYS = ("id", "version", "unit", "disclaimer", "levels", "thresholds", "summary")
_AGGREGATIONS = ("mean", "max")


class RuleError(ValueError):
    """Raised when the grading rule file is malformed or self-inconsistent."""


@dataclass(frozen=True)
class Level:
    key: str
    label: str
    advice: str


@dataclass(frozen=True)
class GradingRule:
    id: str
    version: str
    unit: str
    disclaimer: str
    levels: tuple[Level, ...]
    thresholds: Mapping[str, tuple[float, ...]]
    focus_pollutant: str
    aggregation: str
    source: str

    @property
    def rule_ref(self) -> str:
        return f"{self.id}@{self.version}"


def load_rule(path: Path | str = DEFAULT_RULE_PATH) -> GradingRule:
    """Read and validate a grading rule file; raise RuleError when unusable."""
    rule_path = Path(path)
    raw = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuleError("分级规则文件的顶层必须是一个映射")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise RuleError(f"分级规则缺少字段：{', '.join(missing)}")

    levels_raw = raw["levels"]
    if not isinstance(levels_raw, list) or len(levels_raw) < 2:
        raise RuleError("levels 至少需要两档才能构成分级")
    levels: list[Level] = []
    for item in levels_raw:
        if not isinstance(item, dict) or not {"key", "label", "advice"} <= set(item):
            raise RuleError("每一档 level 都必须包含 key、label 和 advice")
        levels.append(Level(str(item["key"]), str(item["label"]), str(item["advice"])))
    keys = [level.key for level in levels]
    if len(set(keys)) != len(keys):
        raise RuleError("levels 的 key 不能重复")
    bounds_count = len(levels) - 1

    thresholds_raw = raw["thresholds"]
    if not isinstance(thresholds_raw, Mapping) or not thresholds_raw:
        raise RuleError("thresholds 必须是非空的污染物阈值映射")
    thresholds: dict[str, tuple[float, ...]] = {}
    for pollutant, values in thresholds_raw.items():
        if not isinstance(values, list) or len(values) != bounds_count:
            raise RuleError(f"{pollutant} 的阈值数量必须是 {bounds_count} 个上界")
        numbers: list[float] = []
        for value in values:
            try:
                numbers.append(float(value))
            except (TypeError, ValueError) as exc:
                raise RuleError(f"{pollutant} 的阈值必须是数值") from exc
        if any(not math.isfinite(number) for number in numbers):
            raise RuleError(f"{pollutant} 的阈值必须是有限数值")
        if any(low >= high for low, high in zip(numbers, numbers[1:])):
            raise RuleError(f"{pollutant} 的阈值必须严格递增")
        thresholds[str(pollutant)] = tuple(numbers)

    summary = raw["summary"]
    if not isinstance(summary, dict):
        raise RuleError("summary 必须是映射")
    focus = str(summary.get("focus_pollutant") or "")
    if focus not in thresholds:
        raise RuleError(f"summary.focus_pollutant={focus!r} 未在 thresholds 中定义")
    aggregation = str(summary.get("aggregation") or "mean")
    if aggregation not in _AGGREGATIONS:
        raise RuleError(f"summary.aggregation 只支持 {' 或 '.join(_AGGREGATIONS)}")

    return GradingRule(
        id=str(raw["id"]),
        version=str(raw["version"]),
        unit=str(raw["unit"]),
        disclaimer=str(raw["disclaimer"]),
        levels=tuple(levels),
        thresholds=thresholds,
        focus_pollutant=focus,
        aggregation=aggregation,
        source=rule_path.name,
    )


_cache: dict[str, tuple[float, GradingRule]] = {}
_cache_lock = threading.Lock()


def get_rule(path: Path | str = DEFAULT_RULE_PATH) -> GradingRule:
    """Cached rule lookup that still notices on-disk edits without a restart."""
    rule_path = Path(path)
    try:
        key = str(rule_path.resolve())
        mtime = rule_path.stat().st_mtime
    except OSError:
        return load_rule(rule_path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        rule = load_rule(rule_path)
        _cache[key] = (mtime, rule)
        return rule


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def grade(rule: GradingRule, pollutant: str, value: Any) -> dict[str, Any] | None:
    """Grade one concentration. Returns None when the pollutant or value is unusable."""
    bounds = rule.thresholds.get(str(pollutant))
    if bounds is None:
        return None
    number = _finite(value)
    if number is None:
        return None
    index = len(bounds)
    for position, upper in enumerate(bounds):
        if number <= upper:
            index = position
            break
    level = rule.levels[index]
    return {
        "pollutant": str(pollutant),
        "value": round(number, 3),
        "unit": rule.unit,
        "key": level.key,
        "label": level.label,
        "advice": level.advice,
        "rule": rule.rule_ref,
    }


def summarize(rule: GradingRule, values: Iterable[Any]) -> dict[str, Any]:
    """Overall page-level grade from the focus pollutant aggregated over stations."""
    numbers = [number for number in (_finite(value) for value in values) if number is not None]
    base = {
        "focus_pollutant": rule.focus_pollutant,
        "aggregation": rule.aggregation,
        "unit": rule.unit,
        "rule": rule.rule_ref,
        "stations_used": len(numbers),
    }
    if not numbers:
        return {**base, "status": "insufficient_data", "label": "数据不足", "key": "unknown",
                "advice": "焦点污染物没有可用预测值，无法给出风险提示"}
    aggregate = max(numbers) if rule.aggregation == "max" else sum(numbers) / len(numbers)
    graded = grade(rule, rule.focus_pollutant, aggregate)
    if graded is None:  # guarded by load_rule, kept for defence in depth
        return {**base, "status": "ungraded", "label": "未分级", "key": "unknown",
                "advice": "焦点污染物没有配置阈值"}
    return {**base, "status": "ok", "aggregate_value": graded["value"],
            "key": graded["key"], "label": graded["label"], "advice": graded["advice"]}


def describe(rule: GradingRule) -> dict[str, Any]:
    """Machine-readable rule snapshot attached to every prediction response."""
    return {
        "id": rule.id,
        "version": rule.version,
        "rule": rule.rule_ref,
        "unit": rule.unit,
        "disclaimer": rule.disclaimer,
        "source": rule.source,
        "levels": [{"key": lv.key, "label": lv.label, "advice": lv.advice} for lv in rule.levels],
        "thresholds": {name: list(bounds) for name, bounds in sorted(rule.thresholds.items())},
        "focus_pollutant": rule.focus_pollutant,
        "aggregation": rule.aggregation,
    }