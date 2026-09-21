## 用途

本项目是「全国气候网格建模与归因平台」的代码仓库，用于逐步实现该方向的建模与数据处理能力。

当前处于基线状态：只有项目骨架，尚未实现任何业务算法。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

基线尚无测试用例，收集到 0 个用例属预期结果。

## 命令行入口

安装后提供 `climate-grid-modeling` 命令：

```bash
climate-grid-modeling version    # 打印版本号
climate-grid-modeling --help     # 打印用法
```

## 现有公开接口

- 命令行程序 `climate-grid-modeling`
- Python 包 `climate_grid`，其 `__version__` 为当前版本号
- `climate_grid.interpolation.idw_grid(stations, lats, lons, *, power=2.0, radius_km=None, min_points=1)`：基于 Haversine 球面距离（地球半径 6371.0088km）的反距离加权网格插值，返回 schema 为 `climate-grid/idw-v1` 的结果字典（含 `lats`、`lons`、`values`、`counts`，其中 `values`/`counts` 按 `[lats][lons]` 嵌套；输出的 `lats`/`lons` 各项经 `round(x, 12)` 处理并将负零归一为 `0.0`，插值计算仍使用原始坐标且不修改输入）
- `climate_grid.fusion.fuse_observations(observations)`：融合多来源站点观测。丢弃值为 `None` 或质量为 `bad` 的记录；同一 `id` 的剩余记录按质量（`good` 优先于 `suspect`）、`priority` 升序、`source` 字典序取首；返回按 `id` 字典序排列的列表，各项键序为 `id, lat, lon, value, quality, source`。
- `climate_grid.dataset.build(observations, elements, lats, lons, *, power=2.0, radius_km=None, min_points=1)`：多要素数据集构建。观测项在七键之外增加非空 `element` str 键，唯一性为 `(source, id, element)`，同 `id`（含跨要素）坐标须一致；按 `elements` 顺序逐要素调用 `fuse_observations` 与 `idw_grid`（校验、计算、舍入、空网格语义完全沿用），返回键序 `schema, data`，`schema` 为 `climate-grid/dataset-v1`，`data` 按 `elements` 顺序置键、各值为该要素完整的 IDW 结果字典；不修改输入。
- `climate_grid.temporal.reconstruct(frames, *, max_gap=2)`：逐日网格数据集的时间重建。`frames` 为非空 list，各项为恰含 `time, dataset` 两键的 dict；`time` 为唯一的严格 `YYYY-MM-DD` 公历日期 str，`dataset` 为 `build` 的完整返回。帧可乱序；同一帧内各要素及跨帧的同名要素都必须具有一致的要素顺序与 `lats/lons`。`max_gap` 为非 bool 非负 int。输出时间轴自首帧日期至末帧日期逐日升序，缺帧视为全 `None`；逐要素逐格点保留已知值（`observed`，uncertainty 为 `0.0`），连续缺测仅在两端均有已知值且两端日期相距不超过 `max_gap` 天时按日期距离线性插值（`interpolated`，uncertainty 为两端值绝对差的一半，按相同规则舍入），否则为 `missing`（uncertainty 为 `None`）。返回键序 `schema, times, lats, lons, data`，`schema` 为 `climate-grid/temporal-v1`；`data` 按首帧要素顺序，各项键序为 `values, status, uncertainty`，三者均按 `[time][lat][lon]` 嵌套；所有输出数值（含已知值与共享的 `lats/lons`）经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。两层 `schema` 字段非 str 抛 `TypeError`，取值不符抛 `ValueError`；单帧内各要素 `lats/lons` 不一致抛 `ValueError`。
- `climate_grid.extremes.detect(temporal, element, threshold, *, min_duration=2)`：在时间重建结果上检测阈值极端事件。`temporal` 必须为 `reconstruct` 的完整返回（schema 为 `climate-grid/temporal-v1`）；`element` 为其 `data` 内存在的非空 str；`threshold` 为有限非 bool 数；`min_duration` 为非 bool 正 int（默认 `2`）。满足 `value > threshold`（严格大于）且 status 非 `missing` 的 `(t, i, j)` 为体素；体素在同日四邻格点（`i±1,j`、`i,j±1`）或同格点的相邻日（`t±1`）相连，事件为不同日期数不少于 `min_duration` 的三维连通分量。返回键序 `schema, events`，`schema` 为 `climate-grid/extremes-v1`；事件按其最小 `(t, i, j)` 体素坐标字典序升序排列，各事件键序为 `start, end, days, cells, peak, mean, uncertainty`：`start`/`end` 为首末日期 str，`days` 为不同日期数，`cells` 为去重后按字典序排列的 `[i, j]` 列表，`peak`/`mean` 分别为体素 `value - threshold` 的最大值与均值，`uncertainty` 为体素 uncertainty 的均值。所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。类型违例抛 `TypeError`，其他违例（schema、键集、坐标轴、日期、状态与值/不确定性的一致性、要素不存在、阈值非有限、`min_duration` 非正等）抛 `ValueError`。

## 限制

- 命令行除版本查询外没有其他功能。
- 数据来源与更多算法尚未定义；目前提供 `idw_grid` 一个插值算法、`fuse_observations` 一个数据融合函数、`build` 一个多要素数据集构建入口、`reconstruct` 一个逐日时间重建入口及 `detect` 一个极端事件检测入口。
