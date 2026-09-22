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
- `climate_grid.trends.summarize(periods)`：逐年极端事件汇总与趋势斜率。`periods` 为非空 list，各项为恰含 `year, data` 两键的 dict；`year` 为非 bool int 且严格递增；`data` 为非空 dict，以非空 str 要素名映射 `detect` 的完整返回，各年的要素键及顺序须一致。逐年逐要素生成 `count, days, cells, intensity, uncertainty`：`count` 为事件数（int），后四项分别为各事件 `days`、`len(cells)`、`mean`、`uncertainty` 的算术平均；无事件的年份为 `0` 与四个 `None`。各指标以 year 为 x（按 years 顺序）求普通最小二乘斜率：`count` 使用全部年份，其余忽略 `None`，有效点不足 2 时斜率为 `None`。返回键序 `schema, years, data`，`schema` 为 `climate-grid/trends-v1`；`years` 保持输入；`data` 按首年要素顺序，每要素键序 `count, days, cells, intensity, uncertainty, slopes`，前五项均为逐年 list，`slopes` 为按前五键顺序置键的斜率 dict；所有均值与斜率经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.attribution.attribute(trends, element, driver)`：要素强度趋势对外部驱动因子的归因。`trends` 须为 `summarize` 的完整返回；`element` 为 `data` 内非空 str 要素名；`driver` 为与 `years` 等长的 list，各项为有限非 bool 数或 `None`。以该要素逐年 `intensity` 为 y、`driver` 为 x，按年份顺序仅保留 x、y 均非 `None` 的年份；保留年份少于 2 或 `Σ(x-x̄)²=0` 抛 `ValueError`。`x̄`、`ȳ` 为保留值的算术均值，斜率 `b=Σ(x-x̄)(y-ȳ)/Σ(x-x̄)²`。返回键序 `schema, years, data`，`schema` 为 `climate-grid/attribution-v1`，`years` 为保留年份，`data` 为等长 list，各项键序 `contribution, uncertainty`，分别为 `b*(x-x̄)` 及该年对应的原 uncertainty；所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate(temporal, element, regions, *, min_count=1)`：重建序列的区域逐日聚合。`temporal` 须为 `reconstruct` 的完整返回；`element` 为 `data` 内非空 str 要素名；`regions` 为非空 list，各项为恰含 `name, cells` 两键的 dict，`name` 为唯一非空 str，`cells` 为非空 list，各项为升序且不重复的 `[i, j]`，`i`/`j` 为非 bool 非负 int 且在网格内；`min_count` 为非 bool 正 int。逐区域逐日取 `status` 非 `missing` 的 cells，`count` 为其数量，少于 `min_count` 时 `mean, min, max, uncertainty` 皆为 `None`，否则依次为这些格点值的算术均值、最小值、最大值及 `sqrt(Σu²)/count`（u 为各格点 uncertainty）。返回键序 `schema, element, times, regions, data`，`schema` 为 `climate-grid/regional-v1`，`element` 回显，`times` 原样返回，`regions` 为输入顺序的 name 列表，`data` 按 name 顺序排列，每值键序 `count, mean, min, max, uncertainty` 且均为等长逐日序列；所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_multi(temporal, elements, regions, *, min_count=1)`：重建序列的多要素区域逐日聚合，校验、`regions`、`min_count` 及逐日统计规则均沿用 `aggregate`。`elements` 为非空 list，各项为互异的非空 str 且均存在于 `temporal.data`；项类型错抛 `TypeError`，为空、重复或未知要素抛 `ValueError`。返回键序 `schema, elements, times, regions, data`，`schema` 为 `climate-grid/regional-multi-v1`，`elements`、`times` 原样返回，`regions` 为输入顺序的 name 列表；`data` 按 regions 顺序，各项键序 `name, elements`，内层按 `elements` 顺序，各项键序 `element, count, mean, min, max, uncertainty` 且六个序列等长；`count` 为 int，所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_window(temporal, element, regions, windows, *, min_count=1)`：重建序列的按窗口区域聚合，`temporal`、`element`、`regions`、`min_count` 的校验、异常及统计公式均沿用 `aggregate`，但统计跨闭区间内所有日期。`windows` 为非空 list，各项为恰含按序 `name, start, end` 三键的 dict；`name` 为唯一非空 str，`start`/`end` 为有效的 `YYYY-MM-DD` 公历日期、均位于 `temporal.times` 内且 `start <= end`。容器及 `name`/日期类型错抛 `TypeError`，其余窗口约束抛 `ValueError`。按窗口序、区域序，将区间内每日 `status` 非 `missing` 的格点各计一个样本，`count` 为样本数，少于 `min_count` 时 `mean, min, max, uncertainty` 皆为 `None`，否则为样本值的算术均值、最小值、最大值及 `sqrt(Σu²)/count`（u 为各样本 uncertainty）。返回键序 `schema, element, windows, regions, data`，`schema` 为 `climate-grid/window-v1`，`windows` 同序回显三键字典，`regions` 为输入顺序的 name 列表；`data` 按窗口序，各项键序 `name, regions`，区域项键序 `name, count, mean, min, max, uncertainty`；`count` 为 int，所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_exceedance(temporal, element, regions, windows, threshold, *, min_count=1)`：重建序列的按窗口区域阈值超越统计，`temporal`、`element`、`regions`、`windows`、`min_count` 的校验、异常及采样规则均沿用 `aggregate_window`。`threshold` 为有限非 bool 数；类型错抛 `TypeError`，非有限抛 `ValueError`。按窗口序、区域序，将区间内每日 `status` 非 `missing` 的格点各收集一个 `(v, u)` 样本（v 为值、u 为 uncertainty），`count` 为样本数，`exceed` 为满足 `v > threshold` 的样本数，`rate = exceed/count`，`mean_excess = Σmax(v-threshold, 0)/count`，`uncertainty = sqrt(Σu²)/count`；`count < min_count` 时除 `count` 外 `exceed`、`rate`、`mean_excess`、`uncertainty` 皆为 `None`。返回键序 `schema, element, threshold, windows, regions, data`，`schema` 为 `climate-grid/exceedance-window-v1`，`threshold`、`windows` 原样回显，`regions` 为输入顺序的 name 列表；`data` 按窗口序，各项键序 `name, regions`，区域项键序 `name, count, exceed, rate, mean_excess, uncertainty`；`count`/`exceed` 为 int，所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_quantile(temporal, element, regions, windows, quantiles, *, min_count=1)`：重建序列的按窗口区域分位数聚合，`temporal`、`element`、`regions`、`windows`、`min_count` 的校验、异常及采样规则均沿用 `aggregate_window`。`quantiles` 为非空 list，各项为有限非 bool 数、满足 `0 ≤ q ≤ 1` 且严格递增；类型错抛 `TypeError`，为空、越界或非递增抛 `ValueError`。按窗口序、区域序，将区间内每日 `status` 非 `missing` 的格点各计一个样本，`count` 为非 missing 样本数；`count < min_count` 时 `quantiles` 为与参数等长的 `None` 列表、`uncertainty` 为 `None`，否则样本升序排列，取 `h=(n-1)q`、`a=floor(h)`、`b=ceil(h)`，分位数为 `v[a]+(h-a)(v[b]-v[a])`，`uncertainty` 为 `sqrt(Σu²)/count`（u 为各样本 uncertainty）。返回键序 `schema, element, quantiles, windows, regions, data`，`schema` 为 `climate-grid/quantile-window-v1`，`quantiles`、`windows` 回显，`regions` 为输入顺序的 name 列表；`data` 按窗口序，各项键序 `name, regions`，区域项键序 `name, count, quantiles, uncertainty`；`count` 为 int，所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_coverage(temporal, element, regions, windows, *, min_count=1)`：重建序列的按窗口区域数据覆盖率统计，`temporal`、`element`、`regions`、`windows`、`min_count` 的校验、异常均沿用 `aggregate_window`（类型错抛 `TypeError`，其余契约违例抛 `ValueError`）。按窗口序、区域序，`total` 为区间天数与区域 cells 数之积，`available` 计 `status` 非 `missing` 的格点数，并分计 `observed`、`interpolated`，四个计数均为 int；`available < min_count` 时 `rate`、`observed_rate`、`interpolated_rate`、`uncertainty` 皆为 `None`，否则依次为 `available/total`、`observed/total`、`interpolated/total`、`sqrt(Σu²)/available`（u 为各非 missing 格点 uncertainty）。返回键序 `schema, element, windows, regions, data`，`schema` 为 `climate-grid/coverage-window-v1`，`windows` 同序回显三键字典，`regions` 为输入顺序的 name 列表；`data` 按窗口序，各项键序 `name, regions`，区域项键序 `name, total, available, observed, interpolated, rate, observed_rate, interpolated_rate, uncertainty`；所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。
- `climate_grid.regional.aggregate_histogram(temporal, element, regions, windows, edges, *, min_count=1)`：重建序列的按窗口区域值直方图统计，`temporal`、`element`、`regions`、`windows`、`min_count` 的校验、异常及采样规则均沿用 `aggregate_window`。`edges` 为非空 list，各项为有限非 bool 数且严格递增；类型错抛 `TypeError`，为空、非有限或非递增抛 `ValueError`。按窗口序、区域序，将区间内每日 `status` 非 `missing` 的格点各收集一个 `(v, u)` 样本（v 为值、u 为 uncertainty）；`B=len(edges)+1`，`v<edges[0]` 入首桶，`v≥edges[-1]` 入末桶，其余按 `edges[k-1]≤v<edges[k]` 入桶，恰等边界值归入右侧桶。`count` 为样本数 `n`，`bin_counts` 为长度 B 的 int 列表；`n < min_count` 时 `rates` 为等长 `None` 列表、`uncertainty` 为 `None`，否则 `rates[i] = bin_counts[i]/n`、`uncertainty = sqrt(Σu²)/n`（u 为各样本 uncertainty）。返回键序 `schema, element, edges, windows, regions, data`，`schema` 为 `climate-grid/histogram-window-v1`，`edges`、`windows` 回显，`regions` 为输入顺序的 name 列表；`data` 按窗口序，各项键序 `name, regions`，区域项键序 `name, count, bin_counts, rates, uncertainty`；`count` 与 `bin_counts` 各项为 int，所有输出浮点经 `round(x, 12)` 处理并将负零归一为 `0.0`；不修改输入。

## 限制

- 命令行除版本查询外没有其他功能。
- 数据来源与更多算法尚未定义；目前提供 `idw_grid` 一个插值算法、`fuse_observations` 一个数据融合函数、`build` 一个多要素数据集构建入口、`reconstruct` 一个逐日时间重建入口、`detect` 一个极端事件检测入口、`summarize` 一个逐年趋势汇总入口、`attribute` 一个趋势归因入口及 `aggregate`、`aggregate_multi`、`aggregate_window`、`aggregate_exceedance`、`aggregate_quantile`、`aggregate_coverage`、`aggregate_histogram` 七个区域聚合入口。
