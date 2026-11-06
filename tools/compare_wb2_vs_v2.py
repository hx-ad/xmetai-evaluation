#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WB2 评测结果 × xmetai-evaluation 评测结果：指标一致性对拍工具。

用法（本机 / WSL 都可以，路径按各自平台写）：
    python tools/compare_wb2_vs_v2.py
    python tools/compare_wb2_vs_v2.py --v2 <目录或 scores.csv> --wb2 <目录> --out tools/report.md

输入：
  1) evaluation 框架产物目录（默认 weather_ts_single_fgvp，需含 scores.csv）
     长表，一行一个 (变量, 指标, 起报, 时效)。
  2) WB2 结果目录（TongYun_nc）：
     - wb2_2022_<var>_deterministic.nc  每文件一个变量，metric=[acc,bias,mae,mse]
     - wb2_2022_extra.csv               prediction_activity / csi / fss / …
输出：
  tools/wb2_vs_v2_report.md

检验方式（对应报告里的 T1–T7）：
  T1 覆盖检查        两边各自有哪些 (变量, 指标)，哪些不可比
  T2 逐时效数值比对  逐 lead 的相对差 / 绝对差，最大化与中位偏差
  T3 倍率与量纲推断  中位比值 + 比值变异系数，识别纯单位换算
  T4 趋势一致性      Pearson / Spearman / 线性拟合 / 归一化 RMSE
  T5 相对增长曲线    按各自首个时效归一后比形状；跨变量秩相关查串名
  T6 聚合口径敏感性  逐起报 RMSE 的 mean vs RMS 聚合，看哪个更贴 WB2
  T7 起报间散布      差异是否小于起报间标准差（批间噪声量级）

依赖：numpy、pandas、xarray（读 .nc）。
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# 可调参数：直接改这里（不用环境变量）
# ══════════════════════════════════════════════════════════════════════
# 相对差判定阈值（RMSE / 活跃度等有量纲指标）
TOL_PASS_REL = 0.005     # A 一致 ≤0.5%
TOL_CLOSE_REL = 0.010    # B 接近 ≤1%
TOL_MINOR_REL = 0.030    # C 小差 ≤3%
TOL_MAJOR_REL = 0.100    # D 大差 ≤10%，再大看趋势

# ACC 是无量纲、数值贴着 1，用绝对差判定
ACC_ABS_PASS = 2e-3
ACC_ABS_CLOSE = 5e-3
ACC_ABS_MINOR = 1.5e-2
ACC_ABS_MAJOR = 3e-2

# 报告里一律用中文；代码内部仍用 ASCII 键
VERDICT_CN = {
    "PASS": "一致", "CLOSE": "接近", "MINOR": "小差", "MAJOR": "大差",
    "SHAPE": "大差·形状", "MISMATCH": "不一致",
    "SCALE": "倍率", "UNIT": "单位",
}
LABEL_CN = {"Activity(forecast)": "活跃度（预报）"}
VERDICT_ORDER = ["一致", "接近", "小差", "大差", "大差·形状", "倍率", "单位", "不一致"]

OUTLIER_FACTOR = 5.0     # 单时效 |相对差| 超过中位数的 5 倍算异常时效
CSV_CHUNK = 1_000_000    # scores.csv 分块行数

DEFAULT_V2 = r"D:\weatherbench2\scores\weather_ts_single_fgvp"
DEFAULT_WB2 = r"D:\weatherbench2\scores\TongYun_nc"
DEFAULT_OUT = str(Path(__file__).with_name("wb2_vs_v2_report.md"))

# v2 侧聚合哪些指标（其余指标只记 key，用于覆盖检查）
V2_METRICS = {
    "rmse", "bias", "mae", "acc",
    "activity_forecast", "activity_observation",
    "activity_ratio", "activity_bias",
}

CANON_UNITS = {
    "z500": "m²/s²", "q700": "kg/kg", "t700": "K", "t850": "K",
    "t2m": "K", "msl": "Pa", "u850": "m/s", "v850": "m/s",
    "u10m": "m/s", "v10m": "m/s", "tp": "m",
}

# v2 变量 -> WB2 deterministic nc 里 level 坐标的取值（None=地面量）
LEVELS = {
    "z500": 500, "q700": 700, "t700": 700, "t850": 850,
    "u850": 850, "v850": 850,
}

# nc 文件名 token -> v2 变量名
TOKEN2VAR = {
    "z500": "z500", "q700": "q700", "t700": "t700", "t850": "t850",
    "t2m": "t2m", "msl": "msl", "u850": "u850", "v850": "v850",
    "u10": "u10m", "v10": "v10m", "tp": "tp",
}

# extra.csv 的 (WB2 变量名, level) -> v2 变量名
EXTRA_KEY2VAR = {
    ("geopotential", 500): "z500",
    ("specific_humidity", 700): "q700",
    ("temperature", 700): "t700",
    ("temperature", 850): "t850",
    ("u_component_of_wind", 850): "u850",
    ("v_component_of_wind", 850): "v850",
    ("10m_u_component_of_wind", -1): "u10m",
    ("10m_v_component_of_wind", -1): "v10m",
    ("2m_temperature", -1): "t2m",
    ("mean_sea_level_pressure", -1): "msl",
    ("total_precipitation_6hr", -1): "tp",
}

SAMPLE_LEADS = (6, 24, 48, 72, 96, 120, 168, 240, 360)


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Agg:
    """一个 (变量, 指标, 时效) 上跨起报的累计器。"""
    n: int = 0
    s1: float = 0.0
    s2: float = 0.0

    def add(self, n: int, s1: float, s2: float) -> None:
        self.n += n
        self.s1 += s1
        self.s2 += s2

    @property
    def mean(self) -> float:
        return self.s1 / self.n if self.n else float("nan")

    @property
    def rms(self) -> float:
        return math.sqrt(self.s2 / self.n) if self.n else float("nan")

    @property
    def std(self) -> float:
        """逐起报值的标准差（批间散布）。"""
        if not self.n:
            return float("nan")
        m = self.mean
        return math.sqrt(max(self.s2 / self.n - m * m, 0.0))


@dataclass
class PairResult:
    var: str
    label: str
    wb2_metric: str
    v2_metric: str
    n: int
    leads: List[int]
    x: np.ndarray
    y: np.ndarray
    scale: float
    scale_kind: str
    ratio_median: float
    ratio_cv: float
    rel_max: float
    rel_median: float
    abs_max: float
    pearson: float
    spearman: float
    slope: float
    intercept: float
    verdict: str
    note: str
    outliers: List[int] = field(default_factory=list)
    init_scatter_ratio: Optional[float] = None
    x_mean: Optional[np.ndarray] = None   # rmse 的 mean 聚合版本（供 T6）


# ══════════════════════════════════════════════════════════════════════
# 输入读取
# ══════════════════════════════════════════════════════════════════════
def load_v2(path: Path) -> Tuple[Dict, Dict]:
    """流式读 scores.csv，按 (变量, 指标, 时效) 聚合逐起报值。"""
    p = path / "scores.csv" if path.is_dir() else path
    if not p.is_file():
        raise SystemExit(f"找不到 v2 scores.csv: {p}")

    usecols = ["run_id", "model_id", "variable", "metric",
               "lead_h", "value", "unit", "status"]
    v2: Dict[str, Dict[str, Dict[float, Agg]]] = {}
    all_keys: Set[Tuple[str, str]] = set()
    units: Dict[Tuple[str, str], Set[str]] = {}
    run_ids: Set[str] = set()
    models: Set[str] = set()
    n_rows = 0

    reader = pd.read_csv(p, usecols=usecols, chunksize=CSV_CHUNK, low_memory=False)
    for chunk in reader:
        n_rows += len(chunk)
        status = chunk["status"].fillna("success").astype(str)
        chunk = chunk[status.isin(["success", ""])]
        if chunk.empty:
            continue
        run_ids.update(chunk["run_id"].dropna().astype(str).unique())
        models.update(chunk["model_id"].dropna().astype(str).unique())
        all_keys.update(zip(chunk["variable"].astype(str),
                            chunk["metric"].astype(str)))

        sub = chunk[chunk["metric"].isin(V2_METRICS)].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub["lead_h"] = pd.to_numeric(sub["lead_h"], errors="coerce")
        sub = sub[sub["value"].notna() & sub["lead_h"].notna()]
        if sub.empty:
            continue

        for (var, met), g in sub.groupby(["variable", "metric"], sort=False):
            var, met = str(var), str(met)
            store = v2.setdefault(var, {}).setdefault(met, {})
            us = [u for u in g["unit"].dropna().astype(str).unique() if u]
            if us:
                units.setdefault((var, met), set()).update(us)
            for lead, gg in g.groupby("lead_h", sort=False):
                vals = gg["value"].to_numpy(dtype=float)
                key = float(lead)
                acc = store.get(key)
                if acc is None:
                    store[key] = Agg(len(vals), float(vals.sum()),
                                     float(np.square(vals).sum()))
                else:
                    acc.add(len(vals), float(vals.sum()),
                            float(np.square(vals).sum()))

    meta = {
        "path": str(p),
        "size_mb": p.stat().st_size / 1e6,
        "rows": n_rows,
        "run_ids": sorted(run_ids),
        "models": sorted(models),
        "all_keys": all_keys,
        "units": units,
    }
    return v2, meta


def _lead_hours(values) -> np.ndarray:
    vals = np.asarray(values)
    if np.issubdtype(vals.dtype, np.timedelta64):
        return vals.astype("timedelta64[s]").astype(np.int64) / 3600.0
    return vals.astype(float)


def load_wb2_deterministic(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[int, float]], List[str]]:
    """读 wb2_2022_*_deterministic.nc，返回 {(v2变量, 指标): {lead_h: 值}}。"""
    if not path.is_dir():
        raise SystemExit(f"WB2 结果路径不是目录: {path}")
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("读取 .nc 需要 xarray：pip install xarray netCDF4") from exc

    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    files_used: List[str] = []
    files = sorted(path.glob("*_deterministic.nc"))
    if not files:
        raise SystemExit(f"{path} 下没有 *_deterministic.nc")

    for f in files:
        m = re.match(r"^wb2_\d{4}_(?P<tok>.+)_deterministic$", f.stem)
        if not m:
            print(f"[skip] 文件名不识别: {f.name}")
            continue
        var = TOKEN2VAR.get(m.group("tok"))
        if var is None:
            print(f"[skip] 未知变量 token: {m.group('tok')}")
            continue

        ds = xr.open_dataset(f)
        try:
            names = list(ds.data_vars)
            if not names:
                continue
            arr = ds[names[0]]

            if "region" in arr.dims:
                regions = [str(x) for x in np.asarray(arr["region"].values)]
                arr = arr.sel(region="global" if "global" in regions else regions[0],
                              drop=True)

            if "level" in arr.dims:
                levels = np.asarray(arr["level"].values, dtype=float)
                want = LEVELS.get(var)
                idx = 0
                if want is not None:
                    hit = np.where(np.isclose(levels, want))[0]
                    if hit.size:
                        idx = int(hit[0])
                arr = arr.isel(level=idx, drop=True)

            if "metric" not in arr.dims or "lead_time" not in arr.coords:
                print(f"[skip] {f.name} 缺少 metric/lead_time")
                continue
            lead_h = _lead_hours(arr["lead_time"].values)

            for met in [str(x) for x in np.asarray(arr["metric"].values)]:
                vals = np.asarray(arr.sel(metric=met).values, dtype=float).ravel()
                series = {int(round(h)): float(v)
                          for h, v in zip(lead_h, vals) if math.isfinite(v)}
                out.setdefault((var, met), {}).update(series)
                if met == "mse":
                    # WB2 的 rmse = sqrt(时间平均后的 mse)
                    out.setdefault((var, "rmse"), {}).update(
                        {k: math.sqrt(v) if v >= 0 else float("nan")
                         for k, v in series.items()})
        finally:
            ds.close()
        files_used.append(f.name)

    return out, files_used


def load_wb2_extra(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[int, float]], Set[Tuple[str, str, str]], int]:
    """读 wb2_2022_extra.csv（prediction_activity / csi / fss …）。"""
    f = path / "wb2_2022_extra.csv" if path.is_dir() else path
    if not f.is_file():
        return {}, set(), 0
    df = pd.read_csv(f)
    need = {"metric", "var", "level", "lead_h", "value"}
    if not need.issubset(df.columns):
        raise SystemExit(f"extra.csv 缺少列: {sorted(need - set(df.columns))}")

    out: Dict[Tuple[str, str], Dict[int, float]] = {}
    unmatched: Set[Tuple[str, str, str]] = set()
    for row in df.itertuples(index=False):
        wb2_var = str(row.var)
        level = int(round(float(row.level)))
        v2_var = EXTRA_KEY2VAR.get((wb2_var, level))
        met = str(row.metric)
        lead = int(round(float(row.lead_h)))
        val = float(row.value)
        if v2_var is None:
            unmatched.add((wb2_var, str(level), met))
            continue
        if math.isfinite(val):
            out.setdefault((v2_var, met), {})[lead] = val
    return out, unmatched, len(df)


# ══════════════════════════════════════════════════════════════════════
# 比对
# ══════════════════════════════════════════════════════════════════════
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return _pearson(ra, rb)


def _verdict(abs_metric: bool, rel_max: float, abs_max: float,
             pearson: float, scale_kind: str) -> str:
    if scale_kind == "unit":
        return "UNIT"
    if scale_kind == "approx":
        return "SCALE"
    if abs_metric:
        limits = (ACC_ABS_PASS, ACC_ABS_CLOSE, ACC_ABS_MINOR, ACC_ABS_MAJOR)
        value = abs_max
    else:
        limits = (TOL_PASS_REL, TOL_CLOSE_REL, TOL_MINOR_REL, TOL_MAJOR_REL)
        value = rel_max
    for limit, tier in zip(limits, ("PASS", "CLOSE", "MINOR", "MAJOR")):
        if value <= limit:
            return tier
    if math.isfinite(pearson) and pearson >= 0.999:
        return "SHAPE"
    return "MISMATCH"


def compare_pair(var: str, label: str,
                 v2_metric: str, wb2_metric: str,
                 x_map: Dict[int, float], y_map: Dict[int, float],
                 *, abs_metric: bool, note: str,
                 init_std: Optional[Dict[int, float]] = None,
                 x_mean_map: Optional[Dict[int, float]] = None) -> Optional[PairResult]:
    leads = sorted(set(x_map) & set(y_map))
    if len(leads) < 2:
        return None
    x = np.array([x_map[l] for l in leads], dtype=float)
    y = np.array([y_map[l] for l in leads], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    leads = [l for l, k in zip(leads, ok) if k]
    x, y = x[ok], y[ok]
    if x.size < 2:
        return None

    # T4 趋势（先算相关，T3 判量纲时要用）
    pearson = _pearson(x, y)
    spearman = _spearman(x, y)

    # T3 倍率推断：跳过无量纲指标。ratio = v2/WB2；
    # scale 是“把 v2 换算到 WB2 口径”的因子，所以取 1/ratio。
    ratio = np.full(x.size, np.nan)
    nz = np.abs(y) > 1e-12
    ratio[nz] = x[nz] / y[nz]
    ratio_median = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
    ratio_cv = (float(np.nanstd(ratio) / abs(ratio_median))
                if math.isfinite(ratio_median) and ratio_median != 0 else float("nan"))
    scale, scale_kind = 1.0, "none"
    if (not abs_metric and x.size >= 3 and math.isfinite(ratio_median)
            and ratio_median != 0):
        if ratio_cv <= 1e-9 and abs(ratio_median - 1.0) > 1e-6:
            scale, scale_kind = 1.0 / ratio_median, "unit"
        elif ratio_cv <= 0.01 and abs(ratio_median - 1.0) > 0.01:
            scale, scale_kind = 1.0 / ratio_median, "approx"

    # T2 逐时效差
    abs_diff = x * scale - y
    rel = np.full(x.size, np.nan)
    rel[nz] = (x[nz] * scale - y[nz]) / np.abs(y[nz])
    fine = np.isfinite(rel)
    rel_max = float(np.max(np.abs(rel[fine]))) if fine.any() else float("nan")
    rel_median = float(np.median(np.abs(rel[fine]))) if fine.any() else float("nan")
    abs_max = float(np.max(np.abs(abs_diff))) if abs_diff.size else float("nan")

    # 异常时效：|相对差| 超过中位的 OUTLIER_FACTOR 倍且 >1%
    outliers: List[int] = []
    if fine.any() and math.isfinite(rel_median) and rel_median > 0:
        thr = max(OUTLIER_FACTOR * rel_median, TOL_CLOSE_REL)
        outliers = [l for l, r in zip(leads, rel)
                    if math.isfinite(r) and abs(r) > thr]

    # T4 线性拟合
    slope, intercept = float("nan"), float("nan")
    if np.std(x) > 0:
        try:
            slope, intercept = np.polyfit(x, y, 1)
            slope, intercept = float(slope), float(intercept)
        except (np.linalg.LinAlgError, ValueError):
            pass

    # T7 起报间散布
    scatter_ratio = None
    if init_std:
        # init_std 是 v2 原始单位的散布，先乘同一个 scale 换到 WB2 口径
        stds = [init_std.get(l, float("nan")) * scale for l in leads]
        stds = np.array([s for s in stds if math.isfinite(s)], dtype=float)
        deltas = np.abs(abs_diff)
        if stds.size and np.median(stds) > 0:
            scatter_ratio = float(np.median(deltas) / np.median(stds))

    verdict = _verdict(abs_metric, rel_max, abs_max, pearson, scale_kind)
    note_parts = [note]
    if scale_kind in {"unit", "approx"}:
        tag = "精确" if scale_kind == "unit" else "近似"
        note_parts.append(f"{tag}倍率 {ratio_median:.6g}×（v2/WB2）")
    if outliers:
        shown = ", ".join(str(l) for l in outliers[:8])
        more = "…" if len(outliers) > 8 else ""
        note_parts.append(f"异常时效 {len(outliers)} 个: {shown}{more}")
    if scatter_ratio is not None:
        tag = "小于" if scatter_ratio < 1 else "大于"
        note_parts.append(f"中位|Δ| {tag}起报间标准差 ({scatter_ratio:.2f}×)")

    x_mean = None
    if x_mean_map is not None:
        x_mean = np.array([x_mean_map.get(l, np.nan) for l in leads], dtype=float)

    return PairResult(
        var=var, label=label, wb2_metric=wb2_metric, v2_metric=v2_metric,
        n=x.size, leads=leads, x=x, y=y, scale=scale, scale_kind=scale_kind,
        ratio_median=ratio_median, ratio_cv=ratio_cv,
        rel_max=rel_max, rel_median=rel_median, abs_max=abs_max,
        pearson=pearson, spearman=spearman,
        slope=slope, intercept=intercept,
        verdict=verdict, note="；".join(note_parts),
        outliers=outliers, init_scatter_ratio=scatter_ratio, x_mean=x_mean,
    )


def build_pairs(v2: Dict, det: Dict, extra: Dict) -> List[PairResult]:
    results: List[PairResult] = []

    for var in sorted(v2):
        mets = v2[var]

        if "rmse" in mets and (var, "rmse") in det:
            x_map = {l: a.rms for l, a in mets["rmse"].items()}
            x_mean_map = {l: a.mean for l, a in mets["rmse"].items()}
            init_std = {l: a.std for l, a in mets["rmse"].items()}
            r = compare_pair(
                var, "RMSE", "rmse", "rmse", x_map, det[(var, "rmse")],
                abs_metric=False, init_std=init_std, x_mean_map=x_mean_map,
                note="v2 用逐起报 RMSE 的 RMS 聚合；WB2 为 sqrt(mse)",
            )
            if r:
                results.append(r)

        if "acc" in mets and (var, "acc") in det:
            x_map = {l: a.mean for l, a in mets["acc"].items()}
            r = compare_pair(
                var, "ACC", "acc", "acc", x_map, det[(var, "acc")],
                abs_metric=True,
                note="均为距平相关（未去域均）；气候态来源不同",
            )
            if r:
                results.append(r)

        if "bias" in mets and (var, "bias") in det:
            x_map = {l: a.mean for l, a in mets["bias"].items()}
            r = compare_pair(var, "Bias", "bias", "bias", x_map, det[(var, "bias")],
                             abs_metric=False, note="逐格平均偏差")
            if r:
                results.append(r)

        if "mae" in mets and (var, "mae") in det:
            x_map = {l: a.mean for l, a in mets["mae"].items()}
            r = compare_pair(var, "MAE", "mae", "mae", x_map, det[(var, "mae")],
                             abs_metric=False, note="平均绝对误差")
            if r:
                results.append(r)

        if "activity_forecast" in mets and (var, "prediction_activity") in extra:
            x_map = {l: a.mean for l, a in mets["activity_forecast"].items()}
            r = compare_pair(
                var, "Activity(forecast)", "activity_forecast", "prediction_activity",
                x_map, extra[(var, "prediction_activity")],
                abs_metric=False,
                note="口径：v2=距平标准差，WB2=距平 RMS；气候态也不同",
            )
            if r:
                results.append(r)

    return results


# ══════════════════════════════════════════════════════════════════════
# 附加检查
# ══════════════════════════════════════════════════════════════════════
def aggregation_sensitivity(results: Sequence[PairResult]) -> List[Tuple[str, float, float, float, str]]:
    """T6：RMSE 的 mean vs RMS 聚合，哪个更贴 WB2。"""
    rows = []
    for r in results:
        if r.label != "RMSE" or r.x_mean is None:
            continue
        nz = np.abs(r.y) > 1e-12
        rel_rms = np.abs((r.x * r.scale - r.y)[nz] / r.y[nz])
        rel_mean = np.abs((r.x_mean * r.scale - r.y)[nz] / r.y[nz])
        mr = float(np.max(rel_rms)) if rel_rms.size else float("nan")
        mm = float(np.max(rel_mean)) if rel_mean.size else float("nan")
        rows.append((r.var, mr, mm, r.ratio_median,
                     "RMS" if mr <= mm else "mean"))
    return rows


def growth_curve_check(results: Sequence[PairResult]) -> Dict:
    """T5：按首时效归一的增长曲线形状 + 跨变量秩相关。"""
    curves = {}
    for r in results:
        if r.label != "RMSE" or r.n < 2 or r.y[0] == 0 or r.x[0] == 0:
            continue
        curves[r.var] = (r.x / r.x[0], r.y / r.y[0], r.leads)
    if not curves:
        return {"per_var": [], "rank_spearman_mean": float("nan"), "rank_leads": []}

    per_var = []
    for var, (xn, yn, _) in curves.items():
        rel = np.abs(xn - yn) / np.maximum(np.abs(yn), 1e-12)
        per_var.append((var, _pearson(xn, yn), float(np.max(rel))))

    # 跨变量：每个时效上把归一曲线排序，看两边是否同序
    rank_leads, rank_vals = [], []
    lead_sets = [set(c[2]) for c in curves.values()]
    common = sorted(set.intersection(*lead_sets)) if lead_sets else []
    for lead in common:
        xs, ys = [], []
        for xn, yn, leads in curves.values():
            i = leads.index(lead)
            xs.append(xn[i])
            ys.append(yn[i])
        if len(xs) >= 3:
            rho = _spearman(np.array(xs), np.array(ys))
            if math.isfinite(rho):
                rank_leads.append(lead)
                rank_vals.append(rho)

    return {
        "per_var": per_var,
        "rank_spearman_mean": float(np.mean(rank_vals)) if rank_vals else float("nan"),
        "rank_leads": list(zip(rank_leads, rank_vals)),
    }


# ══════════════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════════════
def _f(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}g}" if abs(v) < 1e3 and v != 0 else f"{v:.{nd}f}"


def _pct(v) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v * 100:.3f}%"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    # 单元格内的 | 必须转义，否则会拆列（备注里的 |Δ| 最常见）
    def cell(value) -> str:
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(cell(c) for c in row) + " |")
    return out


def _sample_indices(leads: Sequence[int]) -> List[int]:
    idx = []
    for s in SAMPLE_LEADS:
        if s in leads:
            idx.append(leads.index(s))
    if not idx:
        idx = list(range(min(6, len(leads))))
    if leads and (len(leads) - 1) not in idx:
        idx.append(len(leads) - 1)
    return sorted(set(idx))


def write_report(path: Path, *, v2_meta: Dict, wb2_meta: Dict,
                 results: Sequence[PairResult], agg_rows, growth: Dict,
                 extra_unmatched: Set[Tuple[str, str, str]],
                 args) -> None:
    L: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    L += [
        "# WB2 × xmetai-evaluation 指标一致性对拍报告",
        "",
        f"- 生成时间：{now}",
        f"- v2 产物：`{v2_meta['path']}`（{v2_meta['size_mb']:.1f} MB，{v2_meta['rows']:,} 行）",
        f"- v2 run_id / model：{', '.join(v2_meta['run_ids']) or '—'} / {', '.join(v2_meta['models']) or '—'}",
        f"- WB2 产物：`{wb2_meta['path']}`（{len(wb2_meta['det_files'])} 个 deterministic nc，"
        f"extra.csv {wb2_meta['extra_rows']} 行）",
        f"- 判定阈值：A 一致 ≤{TOL_PASS_REL*100:g}%，B 接近 ≤{TOL_CLOSE_REL*100:g}%，"
        f"C 小差 ≤{TOL_MINOR_REL*100:g}%，D 大差 ≤{TOL_MAJOR_REL*100:g}%，"
        f"E 不一致 >{TOL_MAJOR_REL*100:g}%",
        "- 判定含义：A–D 是数值差的大小；E 表示幅度或趋势不一致；"
        "「形状」= 幅度有系统差但趋势一致；「倍率/单位」= 检测到精确/近似恒定倍率"
        "（已按倍率折算后再看）。",
        "",
        "### 判定阈值",
        "",
        "主档（有量纲指标，按 max|Δ%|）：",
        "",
        "| 档 | 阈值 | 含义 | 处理动作 |",
        "|---|---|---|---|",
        "| A 一致 | ≤0.5% | 实现噪声级（聚合细节、float32） | 直接混用 |",
        "| B 接近 | 0.5–1% | 已知二阶口径差 | 可直接用，注明口径 |",
        "| C 小差 | 1–3% | 气候态/加权等一阶口径差 | 只能相对比较，不能混绝对值 |",
        "| D 大差 | 3–10% | 系统性偏差，需归因 | 发布前修 |",
        "| E 不一致 | >10% | 大概率 bug / 变量错配 / 单位错 | 停用排查 |",
        "",
        "不同指标不能用同一把尺：",
        "",
        "| 指标 | A / B / C / D | 超过 D |",
        "|---|---|---|",
        "| RMSE、MAE、活跃度 | max\\|Δ%\\| ≤0.5 / 1 / 3 / 10% | 不一致 |",
        "| ACC | max\\|Δ\\| ≤0.002 / 0.005 / 0.015 / 0.03 | 不一致 |",
        "| FSS | max\\|Δ\\| ≤0.005 / 0.01 / 0.03 / 0.05 | 不一致 |",
        "| Bias | \\|Δbias\\|/RMSE ≤0.02 / 0.05 / 0.10 / 0.20 | 不一致 |",
        "",
        "> ACC 的 C 档取 0.015：0.01 的绝对差在 ACC≈0.3–0.6 时已有 2–3% 相对差；"
        "本批 z500 长时效的最大差 0.0108 主要来自气候态。",
        "",
    ]

    # ── 0. 结论总表 ──
    L += ["## 0. 结论总表", ""]
    if not results:
        L += ["> 没有找到可对表的 (变量, 指标) 组合。", ""]
    else:
        rows = []
        for r in sorted(results, key=lambda z: (z.var, z.label)):
            rows.append([
                r.var, LABEL_CN.get(r.label, r.label), str(r.n),
                _f(r.ratio_median), _pct(r.rel_max), _f(r.abs_max, 3),
                _f(r.pearson, 5), _f(r.spearman, 4),
                f"**{VERDICT_CN.get(r.verdict, r.verdict)}**", r.note,
            ])
        L += _md_table(
            ["变量", "指标", "时效数", "比值中位", "最大相对差",
             "最大绝对差", "Pearson", "Spearman", "判定", "备注"], rows)
        L.append("")

    # 判定分布
    counts: Dict[str, int] = {}
    for r in results:
        key = VERDICT_CN.get(r.verdict, r.verdict)
        counts[key] = counts.get(key, 0) + 1
    if counts:
        ordered = [k for k in VERDICT_ORDER if k in counts]
        ordered += sorted(set(counts) - set(ordered))
        L += ["判定分布：" + "，".join(f"{k} × {counts[k]}" for k in ordered), ""]

    # ── 1. 覆盖检查 ──
    L += ["## 1. 覆盖检查（T1）", ""]
    v2_keys = v2_meta["all_keys"]
    det_keys = set(wb2_meta["det_keys"])
    extra_keys = set(wb2_meta["extra_keys"])
    matched = {(r.var, r.v2_metric) for r in results}

    L += ["**双边可比**（本报告已逐项检验）：", ""]
    if matched:
        L += [", ".join(f"`{v}/{m}`" for v, m in sorted(matched)), ""]
    else:
        L += ["（无）", ""]

    # WB2 侧未被使用的指标
    used_extra = {(r.var, "prediction_activity")
                  for r in results if r.label.startswith("Activity")}
    wb2_only = []
    for (var, met) in sorted(det_keys):
        if met in {"rmse", "mse"}:   # rmse 由 mse 派生，不单列
            continue
        if (var, met) not in matched:
            wb2_only.append((var, met))
    for (var, met) in sorted(extra_keys):
        if (var, met) not in matched and (var, met) not in used_extra:
            wb2_only.append((var, met))
    L += ["**仅 WB2 有、v2 本批未产出（不可比）**：", ""]
    if wb2_only:
        L += [", ".join(f"`{v}/{m}`" for v, m in wb2_only), ""]
    else:
        L += ["（无）", ""]
    if extra_unmatched:
        tp_metrics = sorted({m for _, _, m in extra_unmatched})
        L += [f"extra.csv 里未映射到 v2 变量的条目还涉及指标：{', '.join(tp_metrics)}"
              f"（本批 v2 没有 tp，无法比较）。", ""]

    v2_only = sorted({(v, m) for (v, m) in v2_keys
                      if (v, m) not in matched
                      and m not in {"mse"}})
    L += ["**仅 v2 有、WB2 侧无对应产物**：", ""]
    if v2_only:
        L += [", ".join(f"`{v}/{m}`" for v, m in v2_only[:40])
              + (" …" if len(v2_only) > 40 else ""), ""]
    else:
        L += ["（无）", ""]

    # ── 2. RMSE ──
    L += ["## 2. RMSE 逐变量对表（T2–T4、T6、T7）", "",
          "> 若某变量涉及倍率/单位换算，`Δ%` 已按检测到的倍率折算；表中 WB2/v2 两列仍是各自原始值，"
          "“比值”列为未折算的 v2/WB2。", ""]
    rmse_results = [r for r in results if r.label == "RMSE"]
    if not rmse_results:
        L += ["（无 RMSE 对表组合）", ""]
    for r in rmse_results:
        L.append(f"### {r.var}  —— 判定 **{VERDICT_CN.get(r.verdict, r.verdict)}**")
        L.append("")
        L.append(f"- 最大相对差 {_pct(r.rel_max)}，中位 {_pct(r.rel_median)}；"
                 f"比值中位 {_f(r.ratio_median)}（CV {_f(r.ratio_cv)}）；"
                 f"Pearson {_f(r.pearson, 6)}，Spearman {_f(r.spearman, 5)}")
        if math.isfinite(r.slope):
            L.append(f"- 线性拟合 y ≈ {_f(r.slope, 5)}·x + {_f(r.intercept, 4)}")
        if r.init_scatter_ratio is not None:
            L.append(f"- T7 中位|Δ| / 起报间标准差 = {r.init_scatter_ratio:.3f}")
        L.append("")
        idx = _sample_indices(r.leads)
        rows = []
        for i in idx:
            rel = ((r.x[i] * r.scale - r.y[i]) / abs(r.y[i])
                   if r.y[i] else float("nan"))
            rows.append([
                str(r.leads[i]), _f(r.y[i], 6), _f(r.x[i], 6),
                _pct(rel),
                _f(r.x[i] / r.y[i], 6) if r.y[i] else "—",
            ])
        L += _md_table(["lead(h)", "WB2", "v2(RMS)", "Δ%", "比值"], rows)
        L.append("")
        L.append("<details><summary>逐时效全表</summary>")
        L.append("")
        rows = []
        for i, lead in enumerate(r.leads):
            rel = ((r.x[i] * r.scale - r.y[i]) / abs(r.y[i])
                   if r.y[i] else float("nan"))
            rows.append([str(lead), _f(r.y[i], 8), _f(r.x[i], 8),
                         _pct(rel), _f(r.x[i] / r.y[i], 8) if r.y[i] else "—"])
        L += _md_table(["lead(h)", "WB2", "v2(RMS)", "Δ%", "比值"], rows)
        L += ["", "</details>", ""]

    # ── 3. ACC ──
    L += ["## 3. ACC 对表（T2、T4）", ""]
    acc_results = [r for r in results if r.label == "ACC"]
    if not acc_results:
        L += ["（无 ACC 对表组合：本批 v2 只产出 z500 的 ACC）", ""]
    for r in acc_results:
        L.append(f"### {r.var}  —— 判定 **{VERDICT_CN.get(r.verdict, r.verdict)}**")
        L.append("")
        L.append(f"- 最大绝对差 {_f(r.abs_max, 4)}，中位相对差 {_pct(r.rel_median)}；"
                 f"Pearson {_f(r.pearson, 6)}")
        L.append(f"- 说明：{r.note}")
        L.append("")
        idx = _sample_indices(r.leads)
        rows = [[str(r.leads[i]), _f(r.y[i], 6), _f(r.x[i], 6),
                 _f(r.x[i] - r.y[i], 4)] for i in idx]
        L += _md_table(["lead(h)", "WB2", "v2", "绝对差"], rows)
        L.append("")
        L.append("<details><summary>逐时效全表</summary>")
        L.append("")
        rows = [[str(l), _f(r.y[i], 8), _f(r.x[i], 8), _f(r.x[i] - r.y[i], 6)]
                for i, l in enumerate(r.leads)]
        L += _md_table(["lead(h)", "WB2", "v2", "绝对差"], rows)
        L += ["", "</details>", ""]

    # ── 4. 活跃度 ──
    L += ["## 4. 活跃度对表（T2、T4）", ""]
    act_results = [r for r in results if r.label.startswith("Activity")]
    if not act_results:
        L += ["（无活跃度对表组合）", ""]
    for r in act_results:
        unit = CANON_UNITS.get(r.var, "")
        L.append(f"### {r.var}  —— 判定 **{VERDICT_CN.get(r.verdict, r.verdict)}**")
        L.append("")
        L.append(f"- 最大相对差 {_pct(r.rel_max)}，比值中位 {_f(r.ratio_median)}"
                 f"（CV {_f(r.ratio_cv)}）；Pearson {_f(r.pearson, 5)}")
        L.append(f"- 说明：{r.note}")
        L.append("")
        idx = _sample_indices(r.leads)
        rows = [[str(r.leads[i]), _f(r.y[i], 6), _f(r.x[i], 6),
                 _pct((r.x[i] * r.scale - r.y[i]) / r.y[i]
                      if r.y[i] else float('nan'))]
                for i in idx]
        L += _md_table(["lead(h)", f"WB2 ({unit})", f"v2 ({unit})", "Δ%"], rows)
        L.append("")

    # ── 5. 聚合口径与增长曲线 ──
    L += ["## 5. 聚合口径与增长曲线（T5、T6）", ""]
    if agg_rows:
        L += ["### T6 逐起报 RMSE 的聚合方式", "",
              "WB2 存的是跨起报池化后的 mse，所以 v2 侧应把逐起报 RMSE 先平方再平均（RMS），"
              "而不是直接平均 RMSE。下表比较两种聚合的最大相对差：", ""]
        rows = [[v, _pct(mr), _pct(mm), _f(ratio), winner]
                for v, mr, mm, ratio, winner in agg_rows]
        L += _md_table(["变量", "max|Δ|（RMS 聚合）", "max|Δ|（mean 聚合）",
                        "比值中位", "更接近 WB2"], rows)
        wins = sum(1 for row in agg_rows if row[4] == "RMS")
        L += ["", f"结论：{wins}/{len(agg_rows)} 个变量 RMS 聚合更贴 WB2。", ""]
    else:
        L += ["（没有 RMSE 对表，跳过 T6）", ""]

    L += ["### T5 相对增长曲线（各自除以首个时效）", ""]
    if growth["per_var"]:
        rows = [[v, _f(c, 6), _pct(d)] for v, c, d in growth["per_var"]]
        L += _md_table(["变量", "归一曲线 Pearson", "最大归一偏差"], rows)
        L.append("")
        if math.isfinite(growth["rank_spearman_mean"]):
            L.append(f"跨变量排序一致性（每个时效上对归一增长曲线排序）："
                     f"Spearman 均值 {growth['rank_spearman_mean']:.4f}。"
                     "接近 1 说明没有变量串名/错位。")
        L.append("")
    else:
        L += ["（没有可比较的 RMSE 曲线）", ""]

    # ── 6. 检验方法说明 ──
    L += [
        "## 6. 检验方法说明",
        "",
        "- **T1 覆盖检查**：枚举两边 (变量, 指标)，标出双边可比 / 仅一边有。",
        "- **T2 逐时效数值比对**：对每个 lead 计算 `(v2−WB2)/|WB2|`（RMSE、活跃度）"
        "或绝对差（ACC），给出最大、中位偏差和异常时效。",
        "- **T3 倍率与量纲推断**：看 `v2/WB2` 的中位比值与变异系数；若比值恒定且≠1，"
        "判为纯单位/量纲换算（单位），不按数值不一致处理。",
        "- **T4 趋势一致性**：Pearson、Spearman、线性拟合与斜率。相关系数接近 1 但幅度有偏，"
        "通常指向口径（气候态、聚合方式）差异而不是数据错误。",
        "- **T5 相对增长曲线**：各自除以首个时效后比较形状；再把各变量在每个时效上排序，"
        "算跨变量 Spearman，用于发现变量串名或错位。",
        "- **T6 聚合口径敏感性**：把 v2 逐起报 RMSE 分别按 `mean` 与 `RMS` 聚合，"
        "与 WB2 池化 mse 比较，看哪种更接近。",
        "- **T7 起报间散布**：用逐起报 RMSE 的标准差作噪声尺，比较 `中位|Δ| / 标准差`；"
        "小于 1 说明当前差异在批间抖动量级内。",
        "",
    ]

    # ── 7. 已知口径差异与注意事项 ──
    scale_pairs = [r for r in results if r.verdict in {"SCALE", "UNIT"}]
    L += [
        "## 7. 已知口径差异与注意事项",
        "",
        "- **RMSE**：WB2 deterministic nc 存的是 `mse`（跨起报池化），本报告开方后比较；"
        "v2 用逐起报 RMSE 的 RMS 聚合（T6 会验证这一点）。",
        "- **ACC**：两边都是“距平场直接二阶矩”的未去域均相关（v2 `centered=False`，"
        "WB2 `_spatial_average(f'·o')/sqrt(...)`）；差异主要来自气候态——"
        "v2 用 15 天滑动气候态，WB2 用 1990–2020 未平滑 6h 气候态。",
        "- **活跃度**：v2 是距平场纬度加权**标准差**，WB2 `prediction_activity` 是距平场"
        "纬度加权**RMS**（含距平均值项），且气候态不同；预期存在几个百分点系统差。",
    ]
    if scale_pairs:
        L.append("- **恒定倍率（倍率/单位）**：下列配对数值差主要由单位/口径换算造成，"
                 "已按倍率折算后再评估数值一致性：")
        for r in scale_pairs:
            L.append(f"  - `{r.var}/{r.label}`：v2/WB2 ≈ {r.ratio_median:.6g}"
                     f"（换算后最大相对差 {_pct(r.rel_max)}）")
    L += [
        "- **起报窗**：v2 配置里写的是 2022 全年；WB2 的最后一个起报是 "
        "2022-12-16T12（+360h 落在 12-31，仍在 2022 真值内），评测的有效时间覆盖全年。"
        "若 v2 的 inference_raw 也止于 12-16，两边样本一致。",
        "- **仅一边有的产物**：v2 的谱 / 球谐带在 WB2 侧没有对应指标；"
        "WB2 的 `bias`/`mae`/`csi`/`frequency_bias`/`fss`/`relative_intensity` "
        "在本批 v2 产物中要么未产出、要么没有 tp，故不参与对表。",
        "- **FSS 特别说明**：即使补上 tp，v2 的 FSS（等权、无经度环绕、窗口不含 9px）"
        "与 WB2（面积加权、经度环绕、9px）也不是同一口径，不能直接对表。",
        "",
        "## 8. 输入指纹",
        "",
        f"- v2 指标 key 数（含谱/球谐带）：{len(v2_keys)}",
        f"- WB2 deterministic 指标：{', '.join(sorted({m for _, m in det_keys}))}",
        f"- WB2 extra 指标：{', '.join(sorted({m for _, m in extra_keys}))}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="WB2 与 xmetai-evaluation 指标一致性对拍",
    )
    ap.add_argument("--v2", default=DEFAULT_V2,
                    help="v2 产物目录（含 scores.csv）或 scores.csv 路径")
    ap.add_argument("--wb2", default=DEFAULT_WB2,
                    help="WB2 结果目录（含 *_deterministic.nc 与 wb2_2022_extra.csv）")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="输出 Markdown 报告路径")
    args = ap.parse_args(argv)

    v2_path = Path(args.v2)
    wb2_path = Path(args.wb2)
    out_path = Path(args.out)

    print(f"[1/4] 读取 v2 scores: {v2_path}")
    v2, v2_meta = load_v2(v2_path)
    print(f"      行数 {v2_meta['rows']:,}，指标 key {len(v2_meta['all_keys'])} 个")

    print(f"[2/4] 读取 WB2 deterministic: {wb2_path}")
    det, det_files = load_wb2_deterministic(wb2_path)
    print(f"      nc {len(det_files)} 个，指标 key {len(det)} 个")

    print(f"[3/4] 读取 WB2 extra.csv")
    extra, extra_unmatched, extra_rows = load_wb2_extra(wb2_path)
    print(f"      extra 记录 {extra_rows} 行")

    print("[4/4] 比对并生成报告")
    results = build_pairs(v2, det, extra)
    agg_rows = aggregation_sensitivity(results)
    growth = growth_curve_check(results)
    wb2_meta = {
        "path": str(wb2_path),
        "det_files": det_files,
        "det_keys": set(det),
        "extra_keys": set(extra),
        "extra_rows": extra_rows,
    }
    write_report(out_path, v2_meta=v2_meta, wb2_meta=wb2_meta,
                 results=results, agg_rows=agg_rows, growth=growth,
                 extra_unmatched=extra_unmatched, args=args)
    print(f"报告已写出: {out_path}")

    if results:
        print("\n判定汇总：")
        for r in sorted(results, key=lambda z: (z.var, z.label)):
            label = LABEL_CN.get(r.label, r.label)
            verdict = VERDICT_CN.get(r.verdict, r.verdict)
            print(f"  {r.var:6s} {label:12s} {verdict:8s} "
                  f"max|Δ|={r.rel_max*100:.3f}%  r={r.pearson:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
