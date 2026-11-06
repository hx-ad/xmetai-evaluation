# -*- coding: utf-8 -*-
"""ERA5 再分析 zarr 实况读取器。

与 ``io/gridded.py`` 的通用格点读取器的区别在数据形态，不在流程：

    通用格点源    一堆小文件，一个文件一个时次/时效，靠目录模板 + 文件名正则发现
    ERA5 zarr     **一个 store 装下全部时次**，通道名旁挂在 channel_names.json 里，
                  没有层次轴（层次编码在通道名 z_500 / u_200 中）

所以发现阶段没什么可"发现"的——store 是整块的，``Era5ZarrCatalog`` 只校验路径存在；
真正的活在读：通道名解析、轴定位、按有效时刻精确取数。

变量/单位/分组仍走 ``io/layouts.py`` 的 ``ERA5_ZARR_LAYOUT`` 声明，
配置侧写标准名（``z500``），store 里的通道名（``z_500``）由布局负责翻译。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.core.variables import TemporalKind
from xmetai_evaluation.io.base import DataCatalog, Reader
from xmetai_evaluation.io.layouts import (
    CRA_ZARR_LAYOUT,
    ERA5_ZARR_LAYOUT,
    GriddedLayout,
)

log = logging.getLogger(__name__)

#: 已经告警过的「store 里没有的变量」组合。这是 store 内容的静态事实，但 read()
#: 每块都调一次——不去重就是每块刷一条一模一样的 WARNING。同一条消息只打一次
#: （进程各存一份，fork 下每段一次）。
_WARNED_MISSING: set = set()

#: 通道名清单的旁挂文件（与 store 同级的目录里）
CHANNEL_NAMES_FILE = "channel_names.json"

#: GRIB 缺测哨兵：超过这个量级的值当缺测
MISSING_SENTINEL = 1e30

#: 时间轴单位 -> 秒
_UNIT_SECONDS = {
    "hour": 3600.0,
    "hours": 3600.0,
    "hr": 3600.0,
    "h": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "d": 86400.0,
    "minute": 60.0,
    "minutes": 60.0,
    "min": 60.0,
    "second": 1.0,
    "seconds": 1.0,
    "s": 1.0,
}

_EPOCH = datetime(1970, 1, 1)


def _seconds_since_epoch(moment: datetime) -> float:
    """把（按 UTC 理解的）时刻换算成距 1970-01-01 的秒数。"""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return (moment - _EPOCH).total_seconds()


def _parse_time_units(units: Any, path: str) -> Tuple[float, datetime]:
    """解析 zarr 时间轴的 ``units`` 属性（形如 ``hours since 2025-01-01 00:00:00``）。

    Returns:
        (每小时对应的秒数换算系数, 起算时刻)。
    """
    text = str(units or "").strip().lower()
    if " since " not in text:
        raise DecodeError(
            f"{path} 的 time 轴缺少可解析的 units（需要 '<单位> since <时刻>'，"
            f"实际是 {units!r}）"
        )
    unit, _, base = text.partition(" since ")
    factor = _UNIT_SECONDS.get(unit.strip())
    if factor is None:
        raise DecodeError(f"{path} 的 time 轴单位不认识: {unit!r}")
    stamp = base.strip().replace("z", "+00:00").replace(" ", "T")
    try:
        origin = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise DecodeError(f"{path} 的 time 轴起算时刻无法解析: {base!r}") from exc
    if origin.tzinfo is not None:
        origin = origin.astimezone(timezone.utc).replace(tzinfo=None)
    return factor, origin


def _channel_axis(shape: Sequence[int], count: int, path: str) -> int:
    """定位通道轴：按长度匹配；多个轴同长时优先 axis 1（常见 (time, channel, lat, lon)）。"""
    hits = [index for index, size in enumerate(shape) if size == count]
    if not hits:
        raise DecodeError(
            f"{path} 的 data 形状 {tuple(shape)} 里找不到长度为 {count} 的通道轴"
        )
    return 1 if 1 in hits else hits[-1]


def _axis_for_size(shape: Sequence[int], size: int, taken: set) -> Optional[int]:
    """在未被占用的轴里找长度等于 size 的那个；不唯一就返回 None。"""
    hits = [
        index
        for index, value in enumerate(shape)
        if value == size and index not in taken
    ]
    return hits[0] if len(hits) == 1 else None


class _Store:
    """一个 ERA5 zarr store 的惰性视图（通道名、时间轴、数据体）。"""

    def __init__(self, key: str, path: str):
        self.key = key
        self.path = str(path)
        self._group: Any = None
        self._names: Optional[List[str]] = None
        self._alias: Dict[str, str] = {}
        self._stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._stats_loaded = False
        self._time: Optional[Tuple[np.ndarray, float, datetime]] = None

    # -- 打开 ---------------------------------------------------------------

    def group(self) -> Any:
        if self._group is not None:
            return self._group
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise DecodeError(
                "读取 ERA5 zarr 需要 zarr 包：pip install zarr"
            ) from exc
        if not Path(self.path).exists():
            raise DecodeError(f"ERA5 zarr store 不存在: {self.path}")
        try:
            self._group = zarr.open(self.path, mode="r")
        except Exception as exc:
            raise DecodeError(f"打不开 ERA5 zarr store {self.path}: {exc}") from exc
        return self._group

    def member(self, name: str) -> Any:
        group = self.group()
        try:
            return group[name]
        except Exception as exc:
            raise DecodeError(f"{self.path} 里没有成员 {name!r}: {exc}") from exc

    # -- 通道名 -------------------------------------------------------------

    def channel_names(self) -> List[str]:
        """通道名清单：优先旁挂的 channel_names.json，退回 store 内的 channel 数组。

        名字统一 ``strip().lower()``——参考实现也是这个口径，配置侧才能写小写标准名。
        """
        if self._names is not None:
            return self._names
        sidecar = Path(self.path) / CHANNEL_NAMES_FILE
        if sidecar.is_file():
            try:
                raw = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception as exc:
                raise DecodeError(f"读不了通道名清单 {sidecar}: {exc}") from exc
            names = list(raw)
        else:
            names = self._channel_from_store()
        self._names = [str(name).strip().lower() for name in names]
        self._alias = {
            name.replace("_", ""): name
            for name in self._names
            if "_" in name
        }
        return self._names

    def _channel_from_store(self) -> List[str]:
        """没有旁挂清单时，用 store 里的 channel 数组顶替。"""
        try:
            values = np.asarray(self.member("channel")[:])
        except DecodeError as exc:
            raise DecodeError(
                f"{self.path} 既没有 {CHANNEL_NAMES_FILE}，也没有 channel 成员，"
                f"无法确定通道名"
            ) from exc
        return [self._decode_name(item) for item in values]

    @staticmethod
    def _decode_name(item: Any) -> str:
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)

    def resolve_channel(self, wanted: str) -> Optional[str]:
        """标准/通道名 -> store 里实际的通道名。

        先精确匹配；不中再按"去掉下划线"匹配（配置写 ``z500``、store 里叫 ``z_500``）。
        """
        names = self.channel_names()
        target = str(wanted).strip().lower()
        if target in names:
            return target
        return self._alias.get(target.replace("_", ""))

    # -- 归一化统计量 -------------------------------------------------------

    def stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """store 里随附的 ``mean`` / ``std``（逐通道），没有就 ``None``。

        CRA store 存的是归一化值，反归一化要用这两个数组；ERA5 store 没有成员，
        所以这里返回 ``None`` 而不是报错——要不要用由读取器的 ``denormalize`` 决定。
        """
        if not self._stats_loaded:
            self._stats_loaded = True
            arrays = []
            for name in ("mean", "std"):
                try:
                    arrays.append(
                        np.asarray(self.member(name)[:], dtype="f8").reshape(-1)
                    )
                except DecodeError:
                    arrays = []
                    break
            self._stats = (arrays[0], arrays[1]) if arrays else None
        return self._stats

    # -- 时间轴 -------------------------------------------------------------

    def time_values(self) -> Tuple[np.ndarray, float, datetime]:
        """返回 (时间轴原始值, 单位换算系数, 起算时刻)。

        两种编码都认：

        * CF 风格数值 + ``units='<单位> since <时刻>'``（ERA5 store）；
        * zarr v3 原生的 ``numpy.datetime64``（CRA store，属性里**没有** units），
          此时直接换算成距 1970-01-01 的秒数。

        结果缓存：CRA 的时间轴有近 7 万个时次，每个 chunk 都重读一遍不值得。
        """
        if self._time is None:
            axis = self.member("time")
            values = np.asarray(axis[:])
            attrs = dict(getattr(axis, "attrs", {}) or {})
            if attrs.get("units"):
                factor, origin = _parse_time_units(attrs["units"], self.path)
                raw = values.astype("f8")
            elif np.issubdtype(values.dtype, np.datetime64):
                raw = values.astype("datetime64[ns]").astype("int64") / 1e9
                factor, origin = 1.0, _EPOCH
            else:
                raise DecodeError(
                    f"{self.path} 的 time 轴既没有 units，也不是 datetime64"
                    f"（dtype={values.dtype}，属性={attrs}）"
                )
            self._time = (raw, factor, origin)
        return self._time

    # -- 数据 ---------------------------------------------------------------

    def multi_block(
        self,
        channels: Sequence[str],
        lo: int,
        hi: int,
        lat_size: int,
        lon_size: int,
        time_size: int,
    ) -> Dict[str, np.ndarray]:
        """一次读 ``[lo, hi]`` 时间块里的多个通道，返回 ``{channel: (n_time, lat, lon)}``。

        c78 这类 store 的 chunk 是 ``(1, 通道全量, lat, lon)``——每一片都装着全部
        通道。逐通道调用 ``block()`` 会把同一片解压 N 遍（16 天窗块、12 个变量
        实测 17.9s，一次取只要 4.7s），所以整轮评测走这里。
        """
        if not channels:
            return {}
        raw = self.member("data")
        shape = tuple(raw.shape)
        names = self.channel_names()
        axis = _channel_axis(shape, len(names), self.path)

        taken = {axis}
        axes: Dict[str, Optional[int]] = {"channel": axis}
        for name, size in (("time", time_size), ("lat", lat_size), ("lon", lon_size)):
            found = _axis_for_size(shape, size, taken)
            if found is not None:
                taken.add(found)
            axes[name] = found

        # 长度撞车导致定位不出来时，按剩余轴的顺序补（参考实现也是这个兜底）
        remaining = [index for index in range(len(shape)) if index not in taken]
        for name in ("time", "lat", "lon"):
            if axes[name] is None and remaining:
                axes[name] = remaining.pop(0)
        if any(axes[name] is None for name in ("time", "lat", "lon")):
            raise DecodeError(
                f"{self.path} 的 data 形状 {shape} 无法定位 time/lat/lon 轴"
            )

        slicer: List[Any] = [slice(None)] * len(shape)
        slicer[axes["time"]] = slice(lo, hi + 1)
        # 通道轴一次取多个：zarr 每片只解压一次，这是提速的来源
        slicer[axes["channel"]] = [names.index(channel) for channel in channels]
        order = [axes["time"], axes["channel"], axes["lat"], axes["lon"]]
        block = np.asarray(raw[tuple(slicer)]).transpose(order)
        # 统一转 f4 再返回：CRA store 存的是 float16，直接拿它跟 MISSING_SENTINEL
        # （1e30）比会触发 float16 溢出告警，而且后面的 NaN 掩码本来就按 f4 算。
        return {
            channel: block[:, position, :, :].astype("f4", copy=False)
            for position, channel in enumerate(channels)
        }

    def block(
        self,
        channel: str,
        lo: int,
        hi: int,
        lat_size: int,
        lon_size: int,
        time_size: int,
    ) -> np.ndarray:
        """读单个通道（``multi_block`` 的便捷包装）。"""
        return self.multi_block(
            [channel], lo, hi, lat_size, lon_size, time_size
        )[channel]


class Era5ZarrCatalog(DataCatalog):
    """ERA5 zarr 目录。

    store 本身装全部时次，没有"一个文件一个时刻"可发现，所以这里只校验
    配置声明的每个 store 都存在——路径错了要立刻报，不能等读到一半才炸。
    """

    def __init__(self, stores: Dict[str, str]):
        self.stores = {str(key): str(value) for key, value in (stores or {}).items()}

    def discover(self, request: DataRequest) -> DataIndex:
        if not self.stores:
            raise DiscoveryError(
                "era5_zarr 需要 stores 配置，例如 "
                "{'pl': '.../era5_pl_....zarr', 'sfc': '.../era5_sfc_....zarr'}"
            )
        missing = [key for key, path in self.stores.items() if not Path(path).exists()]
        if missing:
            raise DiscoveryError(
                "era5_zarr 的 store 不存在: "
                + ", ".join(f"{key}={self.stores[key]}" for key in missing)
            )
        return DataIndex(source_id=request.source_id, available=[dict(self.stores)])


class Era5ZarrReader(Reader):
    """把 ERA5 zarr store 读成 ``(valid_time, lat, lon)`` 的标准 Dataset。

    一个 run 可以用两个 store（气压层 + 地面），变量按 ``layout.groups`` 路由。
    """

    #: 写进 provenance 的读取器标识；子类改成自己的名字
    reader_id = "era5_zarr"
    #: 报错/告警里的数据源称呼
    label = "ERA5 zarr"

    def __init__(
        self,
        source_id: str,
        stores: Dict[str, str],
        layout: GriddedLayout = ERA5_ZARR_LAYOUT,
        version: str = "1.0.0",
        denormalize: bool = False,
        expm1_variables: Sequence[str] = (),
    ):
        """
        Args:
            denormalize: store 是否存归一化值。为真时用 store 自带的 ``mean``/``std``
                还原成物理量（``x*std+mean``）；缺这两个数组直接报错，不猜。
            expm1_variables: 需要在读取时做 ``expm1`` 的标准变量名（CRA 的 ``tp``
                在 store 里是 ``log1p(mm)``）。在反归一化之后、``scale`` 之前做。
        """
        super().__init__(source_id=source_id, version=version)
        self.stores = {str(key): str(value) for key, value in (stores or {}).items()}
        self.layout = layout
        self.denormalize = bool(denormalize)
        self.expm1_variables = frozenset(str(name) for name in expm1_variables)
        self._opened: Dict[str, _Store] = {}

    # -- 内部 ---------------------------------------------------------------

    def _store(self, key: str) -> _Store:
        if key not in self._opened:
            if key not in self.stores:
                raise DecodeError(
                    f"{self.reader_id} 没有声明分组 {key!r}；"
                    f"已声明: {sorted(self.stores)}"
                )
            self._opened[key] = _Store(key, self.stores[key])
        return self._opened[key]

    def _route(self, name: str) -> List[str]:
        """一个变量该去哪些 store 找：布局声明优先，找不到再挨个试。"""
        declared = self.layout.groups.get(name)
        if declared:
            return [declared]
        return list(self.stores)

    def _locate(self, name: str) -> Optional[Tuple[_Store, str]]:
        """标准名 -> (store, 通道名)。"""
        spec = self.layout.spec_for(name)
        wanted = spec.source if spec is not None and spec.source else name
        for key in self._route(name):
            channel = self._store(key).resolve_channel(wanted)
            if channel is not None:
                return self._store(key), channel
        return None

    def _time_indices(
        self, store: _Store, times: List[datetime]
    ) -> Tuple[List[int], List[datetime]]:
        """把请求的有效时刻精确匹配到 store 的时间轴上。

        不命中就报错并列出前几个缺失时刻——静默错配比报错危险得多。
        """
        raw, factor, origin = store.time_values()
        available = raw * factor + _seconds_since_epoch(origin)
        wanted = [_seconds_since_epoch(moment) for moment in times]
        order = np.argsort(available)
        available_sorted = available[order]

        indices: List[int] = []
        missing: List[datetime] = []
        for moment, want in zip(times, wanted):
            position = int(np.searchsorted(available_sorted, want))
            position = min(max(position, 0), len(available_sorted) - 1)
            if np.isclose(available_sorted[position], want, atol=1e-6):
                indices.append(int(order[position]))
            else:
                missing.append(moment)
        if missing:
            first = ", ".join(str(item) for item in missing[:5])
            raise DecodeError(
                f"{store.path} 的时间轴里没有这些有效时刻（前 5 个: {first}）；"
                f"store 覆盖 {_format_span(available)}"
            )
        return indices, missing

    # -- Reader 接口 --------------------------------------------------------

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not request.init_times:
            raise DecodeError(
                f"{self.reader_id} 需要显式给出要读的有效时刻（request.init_times），"
                "否则会整块载入整个 store"
            )
        times = sorted(request.init_times)
        resolved: Dict[str, Tuple[_Store, str]] = {}
        unresolved: List[str] = []
        for name in request.variables:
            found = self._locate(str(name))
            if found is None:
                unresolved.append(str(name))
            else:
                resolved[str(name)] = found
        if unresolved:
            # 与参考实现一致：缺的要素跳过，不因此让整轮评测失败
            message = ", ".join(unresolved)
            if message not in _WARNED_MISSING:
                _WARNED_MISSING.add(message)
                log.warning("%s 里没有这些变量，已跳过: %s", self.label, message)
        if not resolved:
            raise DecodeError(
                f"{self.label} 里没有任何请求的变量可用: {list(request.variables)}"
            )

        fields: Dict[str, xr.DataArray] = {}
        units: Dict[str, str] = {}
        semantics: Dict[str, TemporalKind] = {}
        source_variables: Dict[str, str] = {}
        files: List[str] = []

        # 同一个 store 只开一次，且只读覆盖到的时间块
        by_store: Dict[str, List[Tuple[str, _Store, str]]] = {}
        for name, (store, channel) in resolved.items():
            by_store.setdefault(store.key, []).append((name, store, channel))

        for key, items in by_store.items():
            store = self._store(key)
            files.append(store.path)
            sidecar = Path(store.path) / CHANNEL_NAMES_FILE
            if sidecar.is_file():
                files.append(str(sidecar))

            indices, _ = self._time_indices(store, times)
            lo, hi = min(indices), max(indices)
            lat = np.asarray(store.member("lat")[:], dtype="f8")
            lon = np.asarray(store.member("lon")[:], dtype="f8")
            time_size = len(store.member("time")[:])
            picked = np.asarray(indices, dtype=int) - lo

            # 归一化 store：整块只取一次 mean/std，逐通道挑下标
            stats = store.stats() if self.denormalize else None
            if self.denormalize and stats is None:
                raise DecodeError(
                    f"{store.path} 里没有 mean/std，无法反归一化"
                    f"（{self.reader_id} 的 denormalize=True）"
                )
            names_in_store = store.channel_names() if stats is not None else []

            # 同一 store 的通道一次读进来：c78 的 chunk 装着全部通道，逐变量读
            # 会把同一片解压 N 遍（16 天窗块实测 17.9s vs 一次 4.7s）。
            wanted = list(dict.fromkeys(channel for _, _, channel in items))
            blocks = store.multi_block(
                wanted, lo, hi, len(lat), len(lon), time_size
            )
            # 用完一个通道就释放它的中间态，峰值内存只比逐变量读多一块的切片
            refs = {channel: 0 for channel in wanted}
            for _, _, channel in items:
                refs[channel] += 1

            for name, _store_obj, channel in items:
                block = blocks[channel]
                refs[channel] -= 1
                if not refs[channel]:
                    blocks.pop(channel, None)
                values = np.where(np.abs(block) > MISSING_SENTINEL, np.nan, block)
                values = values[picked, :, :].astype("f4")

                if stats is not None:
                    mean, std = stats
                    position = names_in_store.index(channel)
                    if position >= len(mean) or position >= len(std):
                        raise DecodeError(
                            f"{store.path} 的 mean/std 只有 {len(mean)}/{len(std)} 个值，"
                            f"装不下 {len(names_in_store)} 个通道"
                        )
                    values = values * np.float32(std[position]) + np.float32(mean[position])

                if name in self.expm1_variables or channel in self.expm1_variables:
                    values = np.expm1(values)

                spec = self.layout.spec_for(name)
                unit = spec.unit if spec is not None else "unknown"
                if spec is not None and (spec.scale != 1.0 or spec.offset != 0.0):
                    values = values * spec.scale + spec.offset

                fields[name] = xr.DataArray(
                    values,
                    dims=("valid_time", "lat", "lon"),
                    coords={"valid_time": times, "lat": lat, "lon": lon},
                    name=name,
                    attrs={"units": unit},
                )
                units[name] = unit
                source_variables[name] = channel
                if spec is not None:
                    semantics[name] = spec.temporal_kind

        payload = xr.Dataset(fields)
        return DataBundle(
            payload=payload,
            kind=self.layout.kind,
            source_id=self.source_id,
            standard_vars=source_variables,
            semantic=SemanticMetadata(
                units=units,
                temporal_kind=semantics or TemporalKind.INSTANTANEOUS,
                grid_type="regular_latlon",
            ),
            provenance=Provenance(
                input_files=files,
                reader_id=self.reader_id,
                reader_version=self.version,
                source_variables=source_variables,
            ),
        )


class CraZarrReader(Era5ZarrReader):
    """CRA-40 的 c71 zarr store。

    与 ERA5 store 的差别全落在基类的两个开关上，没有新流程：

    * ``denormalize=True``——store 存的是归一化值，用 store 自带的 ``mean``/``std``
      还原成物理量（``x*std+mean``）。这与推理侧 ``TensorDenormalize`` 同一个口径。
    * ``expm1_variables=("tp",)``——store 里 tp 是 ``log1p(mm)``（mean=0/std=1，
      反归一化在它身上是恒等），读出来要做 ``expm1`` 才是 mm。

    布局是 ``CRA_ZARR_LAYOUT``（71 通道，单 store，无 groups）。
    """

    reader_id = "cra_zarr"
    label = "CRA zarr"

    def __init__(
        self,
        source_id: str = "cra_zarr",
        stores: Optional[Dict[str, str]] = None,
        layout: GriddedLayout = CRA_ZARR_LAYOUT,
        version: str = "1.0.0",
    ):
        super().__init__(
            source_id=source_id,
            stores=stores or {},
            layout=layout,
            version=version,
            denormalize=True,
            expm1_variables=("tp",),
        )


def _format_span(seconds: np.ndarray) -> str:
    """把"距 1970 的秒数"数组格式化成可读区间，用于报错信息。"""
    if seconds.size == 0:
        return "空时间轴"
    start = _EPOCH + timedelta(seconds=float(np.min(seconds)))
    end = _EPOCH + timedelta(seconds=float(np.max(seconds)))
    return f"{start} 到 {end}"
