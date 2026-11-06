# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × ERA5）：RMSE / 谱 / ACC / 活跃度。
z500 报 m²/s²（不除 g）、q 报 g/kg、tp 报 mm；ws850 由 sqrt(u²+v²) 现合成，ws10m 取 ERA5 10m 风速。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 要素表 ──
# ws850 现合成、ws10m 实况直取。
VARS = [
    "z500", "q700", "t700", "t2m", "t850", "msl",
    "u850", "v850", "u10m", "v10m", "ws10m", "ws850",
]

# ── 逐变量指标表 ──
# 对齐 wb2_2022（acc / bias / mae / mse + prediction_activity）里本流程模板
# 支持的指标：rmse / bias / acc / activity（rmse = sqrt(mse)，
# activity ≈ prediction_activity；mae 模板不支持，未接）。
# WB2 没有谱，所以 zonal_spectrum / spherical_bands 只有 z500 保留。
VAR_METRICS = {
    "z500": ["rmse", "bias", "zonal_spectrum", "spherical_bands", "acc", "activity"],
    "q700": ["rmse", "bias", "acc", "activity"],
    "t700": ["rmse", "bias", "acc", "activity"],
    "t2m": ["rmse", "bias", "acc", "activity"],
    "t850": ["rmse", "bias", "acc", "activity"],
    "msl": ["rmse", "bias", "acc", "activity"],
    "u850": ["rmse", "bias", "acc", "activity"],
    "v850": ["rmse", "bias", "acc", "activity"],
    "u10m": ["rmse", "bias", "acc", "activity"],
    "v10m": ["rmse", "bias", "acc", "activity"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# centered=False（uncentered / FDP 口径，对齐 xu）
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# max_wavenumber=720（全球 0.25° Nyquist）
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

cfg = EvalConfig(
    name="weather_rmse_single_fgvp",
    description="确定性连续量检验（FuXi × ERA5）：12 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",

    # ── 数据源 ──
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FGVP_OUTPUT", "/workspace/data/shenzw/wb2/inference_raw"
        ),
        "variables": VARS,
        "step_hours": 6.0,
    },
    observation_reader={
        # c78 v3 是单 store、归一化存储（mean/std），用 cra_zarr 读取器（它带
        # denormalize=True）；era5_zarr 读取器要求 pl/sfc 分 store 且不做反归一化，
        # 直接读会是归一化值。
        "type": "cra_zarr",
        "path": "/workspace/data/zarr_refresh_v3/era5.1979_2024.c78.p25.h6.v1",
        "variables": VARS,
    },
    # smooth_days=15：±7.5 天中心滑动（跨年首尾相接）
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        "smooth_days": 15,  # ±7.5 天中心滑动、跨年首尾相接，抹平单日气候态噪声
    },

    # ── 时段 ──
    # 20220101 .. 20221231（inference_raw 的起报范围；c78 盘覆盖到 2024）
    start_date=os.environ.get("START_DATE", "20220101"),
    end_date=os.environ.get("END_DATE", "20221231"),
    limit=None,  # 限起报数：冒烟改 2；None = 不限。改回 None 重跑时 resume
                 # 会复用冒烟算过的块，不白算

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/data/shenzw/wb2/eval_result/weather_rmse_single_fgvp",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",  # DEBUG / INFO / WARNING / ERROR

    # ── 输出视图 ──
    # writers：csv_long 主表，spectrum 逐波数谱曲线。
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ──
    options={
        # sample_by="init_lead"：逐 (起报, 时效) 一个样本
        "sample_by": "init_lead",
        # 三分带：热带 ±20°，南/北中高纬到 ±90°（闭区间，拼满全球）
        "regions": {
            # "tropics": {"lat_min": -20, "lat_max": 20},
            # "nh_extratropics": {"lat_min": 20, "lat_max": 90},
            # "sh_extratropics": {"lat_min": -90, "lat_max": -20},
        },
    },

    # ── 执行 ──
    execution={
        "mode": "processes",
        "n_workers": 16,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "window:16",
            "reference": "resident",     # 日序气候态整 run 驻留（实测 67G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
