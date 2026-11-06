#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 WB2 的 6h 平均气候态 zarr 转成 v2 daily_climatology reader 用的单文件 nc。

源: /workspace/data/shenzw/wb2/climatology/era5_1990_2020_6h_mean.zarr
    dims (hour, dayofyear[, level], latitude, longitude)，已含 WB2 的 61 天
    线性平滑（weatherbench2.utils.compute_hourly_stat_fast 同口径）。
目标: 与 /workspace/data/worm/era5_clim_phys_14.nc 同格式：
    time=1464（int64，0,6,...,8778，units=hours since 1990-01-01）、lat=721
    （90→-90）、lon=1440（0→359.75）；每个变量独立的 (time, lat, lon)，
    float32、NETCDF4、zlib level 4 + shuffle、chunks (183,91,180)。

只写当前评测用到的 11 个变量（zarr 里也只有这些）：
    z500 t700 t850 q700 u850 v850 u10m v10m t2m msl tp
单位换算：
    q700  kg/kg -> g/kg   ×1000
    tp    m     -> mm     ×1000
其余变量原样拷贝。

注意：源 zarr 已经是最终气候态，v2 侧配置必须用 smooth_days=0；
reader 的 smooth_days=15 是沿 6h 时间轴（day*4+hour 交错）做 61 步滑动，
会把日循环抹掉，不能再用。

用法：
    python tools/build_wb2_clim_nc.py
    python tools/build_wb2_clim_nc.py --input_path ... --output_path ...
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import xarray as xr
from netCDF4 import Dataset

DEFAULT_IN = "/workspace/data/shenzw/wb2/climatology/era5_1990_2020_6h_mean.zarr"
DEFAULT_OUT = "/workspace/data/shenzw/wb2/eval_clim/era5_clim_phys_wb2.nc"

# 输出变量, 源变量, 气压层(None=地面), 单位换算系数
VAR_MAP = [
    ("z500", "geopotential", 500, 1.0),
    ("t700", "temperature", 700, 1.0),
    ("t850", "temperature", 850, 1.0),
    ("q700", "specific_humidity", 700, 1000.0),
    ("u850", "u_component_of_wind", 850, 1.0),
    ("v850", "v_component_of_wind", 850, 1.0),
    ("u10m", "10m_u_component_of_wind", None, 1.0),
    ("v10m", "10m_v_component_of_wind", None, 1.0),
    ("t2m", "2m_temperature", None, 1.0),
    ("msl", "mean_sea_level_pressure", None, 1.0),
    ("tp", "total_precipitation_6hr", None, 1000.0),
]

N_DAYS = 366
N_HOURS = 4
N_TIME = N_DAYS * N_HOURS  # 1464
CHUNKS = (183, 91, 180)


def _log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", default=DEFAULT_IN)
    ap.add_argument("--output_path", default=DEFAULT_OUT)
    args = ap.parse_args()
    in_path, out_path = args.input_path, args.output_path

    _log("open zarr %s" % in_path)
    dz = xr.open_zarr(in_path, consolidated=True)

    lat = np.asarray(dz["latitude"].values, dtype="f4")
    lon = np.asarray(dz["longitude"].values, dtype="f4")
    hours = np.asarray(dz["hour"].values, dtype="i8")
    doy = np.asarray(dz["dayofyear"].values, dtype="i8")
    if hours.size != N_HOURS or doy.size != N_DAYS:
        raise SystemExit("源 zarr 的时间轴不对: hours=%s days=%s" % (hours.size, doy.size))
    if not np.array_equal(hours, np.arange(0, 24, 6, dtype="i8")):
        raise SystemExit("源 zarr 的 hour 不是 0/6/12/18: %s" % hours)
    if not np.array_equal(doy, np.arange(1, N_DAYS + 1, dtype="i8")):
        raise SystemExit("源 zarr 的 dayofyear 不是 1..366")

    # 变量与层是否存在
    missing = []
    for name, src, lev, _ in VAR_MAP:
        if src not in dz.data_vars:
            missing.append((name, src, "var"))
        elif lev is not None and lev not in np.asarray(dz["level"].values):
            missing.append((name, src, "level=%s" % lev))
    if missing:
        raise SystemExit("源 zarr 缺变量/层: %s" % missing)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _log("write nc %s" % out_path)
    root = Dataset(out_path, "w", format="NETCDF4")
    try:
        root.createDimension("time", N_TIME)
        root.createDimension("lat", lat.size)
        root.createDimension("lon", lon.size)

        tvar = root.createVariable("time", "i8", ("time",))
        tvar.units = "hours since 1990-01-01 00:00:00"
        tvar.calendar = "proleptic_gregorian"
        tvar[:] = np.arange(N_TIME, dtype="i8") * 6

        latvar = root.createVariable("lat", "f4", ("lat",))
        latvar[:] = lat
        lonvar = root.createVariable("lon", "f4", ("lon",))
        lonvar[:] = lon

        root.source_range = "1990-01-01T00:00:00 to 2020-12-31T18:00:00"
        root.freq = "6h"
        root.denormalized = "True"
        root.source_zarr = in_path
        root.description = (
            "WB2 era5_1990_2020_6h_mean 气候态转 v2 daily_climatology 单文件格式；"
            "q700 ×1000 (g/kg)，tp ×1000 (mm/6h)；源已含 61 天 dayofyear 平滑，"
            "v2 侧须用 smooth_days=0"
        )

        for name, src, lev, scale in VAR_MAP:
            da = dz[src]
            if lev is not None:
                da = da.sel(level=lev, drop=True)

            var = root.createVariable(
                name, "f4", ("time", "lat", "lon"),
                zlib=True, complevel=4, shuffle=True,
                chunksizes=CHUNKS, fill_value=np.float32(np.nan),
            )
            var.long_name = name

            started = time.time()
            # (dayofyear, hour, lat, lon) -> time = (doy-1)*4 + hour_idx
            arr = da.transpose("dayofyear", "hour", "latitude", "longitude").values
            arr = np.ascontiguousarray(arr, dtype="f4").reshape(N_TIME, lat.size, lon.size)
            if scale != 1.0:
                arr = arr * np.float32(scale)
            var[:] = arr
            _log("%-5s <- %-26s level=%-4s ×%-6g  done %.1fs"
                 % (name, src, lev, scale, time.time() - started))
            del arr, da
    finally:
        root.close()

    # ---------------------------------------------------------------- 校验
    _log("verify %s" % out_path)
    out = Dataset(out_path, "r")
    try:
        expect_shape = (N_TIME, lat.size, lon.size)
        sample = [(1, 0), (1, 18), (59, 6), (100, 12), (200, 0), (366, 18)]
        for name, src, lev, scale in VAR_MAP:
            got_shape = out.variables[name].shape
            if got_shape != expect_shape:
                raise SystemExit("%s 形状不对: %s != %s" % (name, got_shape, expect_shape))
            da = dz[src]
            if lev is not None:
                da = da.sel(level=lev, drop=True)
            worst = 0.0
            for d, h in sample:
                t = (d - 1) * 4 + int(h // 6)
                expected = np.asarray(
                    da.isel(hour=int(h // 6), dayofyear=d - 1).values, dtype="f4"
                )
                if scale != 1.0:
                    expected = expected * np.float32(scale)
                actual = np.asarray(out.variables[name][t, :, :], dtype="f4")
                worst = max(worst, float(np.nanmax(np.abs(actual - expected))))
            status = "OK" if worst == 0.0 else "DIFF %.3g" % worst
            _log("verify %-5s max|diff| over %d samples: %s" % (name, len(sample), status))
            if worst != 0.0:
                raise SystemExit("%s 校验失败，max|diff|=%g" % (name, worst))
    finally:
        out.close()

    _log("done: %s (%.2f GB)" % (out_path, os.path.getsize(out_path) / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
