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

## 限制

- 命令行除版本查询外没有其他功能。
- 数据来源与更多算法尚未定义；目前提供 `idw_grid` 一个插值算法及 `fuse_observations` 一个数据融合函数。
