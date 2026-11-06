# -*- coding: utf-8 -*-
"""降水空间检验（FSS）：TongYun tp × ERA5 c78 v3 格点。

对每个 (起报, 时效) 的 6h 降水，按阈值二值化后在 n×n 网格窗口内比较
预报/实况的覆盖率；窗口越大越宽容位置偏差。阈值 1/5/10 mm/6h 与
WeatherBench2 补充指标（csi / frequency_bias / fss）的档位对齐。

单位：预报 inference_raw 的 TP 是 mm/6h；实况 c78 v3 store 的 tp 经读取器
expm1 后也是 mm/6h（2022-01-01 06Z 实测：预报全球均值 0.533、实况 0.556），
两侧不需要额外换算。layouts.py 里 tp 的 unit="inch" 是给老 CRA store 留的
历史注释，对这个 v3 ERA5 store 不成立（只影响报告里的单位标签）。

口径：本实现逐格点等权、邻域越界按 0 计；WB2 的 fss 是面积加权 + 经度
环绕，数值不能直接对表。

时段 2022 全年（inference_raw 的起报范围），时效 6–360h。
"""
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="weather_precip_fss_fgvp",
    description="降水空间检验（TongYun tp × ERA5）：FSS，阈值 1/5/10 mm/6h",
    pipeline="fdp_precip_fss",

    # ── 数据源 ──
    # 预报：inference_raw/{init}/001.nc..060.nc（编号即 6h 步），TP 已是 mm/6h。
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": "/workspace/data/shenzw/wb2/inference_raw",
        "variables": ["tp"],
        "step_hours": 6.0,
    },
    # 实况：ERA5 c78 v3 单 store，0.25°/6h；tp 在 store 里是 log1p(mm)，
    # cra_zarr 读取器做 expm1 后即 mm。
    observation_reader={
        "type": "cra_zarr",
        "path": "/workspace/data/zarr_refresh_v3/era5.1979_2024.c78.p25.h6.v1",
        "variables": ["tp"],
    },
    reference_reader=None,  # FSS 不是距平统计，不需要气候态

    # ── 时段 ──
    start_date="20220101",
    end_date="20221231",
    limit=None,  # 限起报数：冒烟可改 2

    output_dir="/workspace/data/shenzw/wb2/eval_result/weather_precip_fss_fgvp",
    metric_options={
        "fss": {
            # 6h 降水量阈值（mm）。1/5/10 对齐 WB2 的 csi_gt{1,5,10}mm。
            "thresholds": [1.0, 5.0, 10.0],
            # 邻域窗口（奇数个网格点）：0.25° 下 63 点 ≈ 15.75°。
            "windows": [1, 3, 5, 15, 31, 63],
            # 指标只路由到 tp；没这一行 FSS 会按整批变量各跑一遍。
            "variables": ["tp"],
        },
    },
    writers=["csv_long"],
    log_level="INFO",

    # ── 协议参数 ──
    options={
        "sample_by": "init_lead",  # 一个样本 = 一个 (起报, 时效)
    },

    # ── 执行 ──
    # FSS 有邻域卷积，归在重指标，用进程并行。
    execution={
        "mode": "processes",
        "n_workers": 16,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {"observation": "window:16"},
        "resume": True,
    },
)
