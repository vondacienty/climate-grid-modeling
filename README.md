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
- `climate_grid.temporal.reconstruct(frames, *, max_gap=2)`：逐日网格数据集的时间重建。`frames` 为非空 list，各项为恰含 `time, dataset` 两键的 dict；`time` 为唯一的严格 `YYYY-MM-DD` 公历日期 str，`dataset` 为 `build` 的完整返回。帧可乱序；所有帧的要素及顺序、各要素 `lats/lons` 在单帧内与跨帧间必须一致。`max_gap` 为非 bool 非负 int。输出时间轴自首帧日期至末帧日期逐日升序，缺帧视为全 `None`；逐要素逐格点保留已知值（`observed`，uncertainty 为 `0.0`），连续缺测仅在两端均有已知值且两端日期相距不超过 `max_gap` 天时按日期距离线性插值（`interpolated`，uncertainty 为两端值绝对差的一半，按相同规则舍入），否则为 `missing`（uncertainty 为 `None`）。返回键序 `schema, times, lats, lons, data`，`schema` 为 `climate-grid/temporal-v1`；`data` 按首帧要素顺序，各项键序为 `values, status, uncertainty`，三者均按 `[time][lat][lon]` 嵌套；所有输出数值（含已知值与共享的 `lats/lons`）经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.extremes.detect(temporal, element, threshold, *, min_duration=2)`：重建序列的阈值极端事件检测。`temporal` 须为 `reconstruct` 的完整返回；`element` 为 `data` 内非空 str 要素名；`threshold` 为有限非 bool 数；`min_duration` 为非 bool 正 int（默认 2）。满足 `value > threshold` 且 `status` 非 `missing` 的 `(t, i, j)` 为体素；同日四邻或同格相邻日连通，事件为日期数不少于 `min_duration` 的三维连通分量，按最小 `(t, i, j)` 字典序升序。返回键序 `schema, events`，`schema` 为 `climate-grid/extremes-v1`；事件键序 `start, end, days, cells, peak, mean, uncertainty`：首末日期、日期数、去重的 `[i, j]` 升序列表、`value - threshold` 的最大值/均值、对应 uncertainty 的均值；所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.trends.summarize(periods)`：多年极端事件指标的逐年汇总与趋势估计。`periods` 为非空 list，各项为恰含 `year, data` 两键的 dict；`year` 为严格递增的非 bool int，`data` 为非空 dict，以非空 str 要素名映射 `detect` 的完整返回，各年要素及顺序须一致。逐年逐要素生成 `count, days, cells, intensity, uncertainty`：`count` 为事件数，后四项分别为各事件 `days`、`cells` 长度、`mean`、`uncertainty` 的算术平均，无事件时为 `0` 及四个 `None`。各指标以 `year` 为自变量按输入年序求普通最小二乘斜率；`count` 使用全部年份，其余指标忽略 `None` 年份，有效点不足 2 个时斜率为 `None`。返回键序 `schema, years, data`，`schema` 为 `climate-grid/trends-v1`，`years` 按输入年序；`data` 按首年要素顺序，各项键序 `count, days, cells, intensity, uncertainty, slopes`，前五项为逐年 list，`slopes` 为按前五键序置键的斜率 dict；均值与斜率均经 `round(x, 12)` 处理并将负零归一为 `0.0`，`count` 各项为 int；不修改输入。

## 限制

- 命令行除版本查询外没有其他功能。
- 数据来源与更多算法尚未定义；目前提供 `idw_grid` 一个插值算法、`fuse_observations` 一个数据融合函数、`build` 一个多要素数据集构建入口、`reconstruct` 一个逐日时间重建入口、`detect` 一个极端事件检测入口及 `summarize` 一个多年趋势汇总入口。
