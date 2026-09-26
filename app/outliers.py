"""展柜采集质量审计：孤立异常读数定位（纯计算，只读）。

基于**已合并**的温湿度读数，在每个**连续段**内用邻窗中位数 + MAD
（中位数绝对偏差）识别孤立异常点：

- 相邻读数间隔**大于** ``max_gap_seconds`` 时切断序列（恰好等于仍视为
  连续），异常判定**绝不跨越断档**取邻点；
- 对每个读数，仅取**本连续段内**左侧最多 ``window_size`` 个、右侧最多
  ``window_size`` 个邻点（**排除自身**）；任一侧邻点不足 2 个时该点
  **不参与判定**（段首段尾与过短的段天然被跳过）；
- 温度、湿度**各自独立**计算邻点中位数与
  ``MAD = median(|邻点 − 中位数|)``，尺度取
  ``max(1.4826 × MAD, 该维 min_delta)``——``1.4826`` 为正态一致性
  常数（MAD × 1.4826 ≈ 标准差），``min_delta`` 为邻点完全相同时的
  尺度下限，避免零尺度把一切非零偏差都判为异常；
- 原值相对中位数的**绝对偏差严格大于** ``k × 尺度`` 时标记该维度
  （恰好等于不标记）；任一维度被标记即为疑点。

返回的疑点按时间升序排列。全部计算确定性：相同输入恒得相同结果；
整个过程只读，不修改读数、分段或版本号。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from statistics import median
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 正态一致性常数：正态分布下 MAD × 1.4826 ≈ 标准差
MAD_CONSISTENCY_CONSTANT = 1.4826
# 任一侧邻点不足该数量时该点不参与判定
MIN_NEIGHBORS_PER_SIDE = 2


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class DimensionFinding:
    """单个被标记维度的判定依据（邻点中位数与原值绝对偏差）。"""

    dimension: str
    median: float
    deviation: float  # |原值 − 邻点中位数|


@dataclass(frozen=True)
class ReadingOutlier:
    """一个孤立异常读数（疑点）及其被标记维度的判定依据。"""

    ts: dt.datetime
    temperature: float  # 原始温度读数（°C）
    humidity: float  # 原始湿度读数（%RH）
    findings: tuple[DimensionFinding, ...]  # temperature 在前


@dataclass(frozen=True)
class OutlierScan:
    """整柜扫描结果：读数/连续段/参与判定数量与按时间升序的疑点。"""

    sample_count: int
    segment_count: int
    evaluated_count: int  # 两侧邻点均达到最低数量、实际参与判定的读数数量
    outliers: tuple[ReadingOutlier, ...]


def _continuous_segments(
    ordered: list[_SampleLike], max_gap_seconds: float
) -> list[list[_SampleLike]]:
    """按相邻读数间隔切连续段：间隔 > max_gap_seconds 切断（等于仍连续）。"""
    segments: list[list[_SampleLike]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if (cur.ts - prev.ts).total_seconds() > max_gap_seconds:
            segments.append([])
        segments[-1].append(cur)
    return segments


def _dimension_finding(
    value: float,
    neighbor_values: list[float],
    *,
    dimension: str,
    k: float,
    min_delta: float,
) -> DimensionFinding | None:
    """单维判定：|原值 − 邻点中位数| 严格大于 k × 尺度时返回判定依据。"""
    med = float(median(neighbor_values))
    mad = float(median(abs(v - med) for v in neighbor_values))
    scale = max(MAD_CONSISTENCY_CONSTANT * mad, min_delta)
    deviation = abs(value - med)
    if deviation > k * scale:
        return DimensionFinding(dimension=dimension, median=med, deviation=deviation)
    return None


def detect_reading_outliers(
    samples: list[_SampleLike],
    *,
    window_size: int,
    k: float,
    temp_min_delta: float,
    humidity_min_delta: float,
    max_gap_seconds: float,
) -> OutlierScan:
    """在连续段内以邻窗中位数 + MAD 定位孤立异常读数。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    参数合法性（window_size ∈ [2, 50]，k / min_delta / max_gap_seconds
    均为正）由接口层校验。
    """
    ordered = sorted(samples, key=lambda s: s.ts)
    if not ordered:
        return OutlierScan(
            sample_count=0, segment_count=0, evaluated_count=0, outliers=()
        )

    segments = _continuous_segments(ordered, max_gap_seconds)

    outliers: list[ReadingOutlier] = []
    evaluated = 0
    for segment in segments:
        n = len(segment)
        for i in range(n):
            # 仅取本连续段内左右各最多 window_size 个邻点，排除自身
            left = segment[max(0, i - window_size) : i]
            right = segment[i + 1 : i + 1 + window_size]
            if (
                len(left) < MIN_NEIGHBORS_PER_SIDE
                or len(right) < MIN_NEIGHBORS_PER_SIDE
            ):
                continue  # 任一侧不足 2 个邻点：该点不判
            evaluated += 1
            neighbors = left + right

            findings: list[DimensionFinding] = []
            for dimension, attr, min_delta in (
                (TEMPERATURE, "temperature", temp_min_delta),
                (HUMIDITY, "humidity", humidity_min_delta),
            ):
                finding = _dimension_finding(
                    getattr(segment[i], attr),
                    [getattr(s, attr) for s in neighbors],
                    dimension=dimension,
                    k=k,
                    min_delta=min_delta,
                )
                if finding is not None:
                    findings.append(finding)

            if findings:
                outliers.append(
                    ReadingOutlier(
                        ts=segment[i].ts,
                        temperature=segment[i].temperature,
                        humidity=segment[i].humidity,
                        findings=tuple(findings),
                    )
                )

    return OutlierScan(
        sample_count=len(ordered),
        segment_count=len(segments),
        evaluated_count=evaluated,
        outliers=tuple(outliers),
    )
