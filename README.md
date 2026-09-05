# 穹宇智析：长沙多站空气污染预测

使用长沙 10 个监测站过去 12 小时的污染与气象观测，直接预测未来 T+1、T+2、T+3 的 PM2.5、PM10、NO2、O3。仓库包含可追溯数据处理、无泄露时间评估、baseline/创新模型、Flask API 和静态前端。

## 当前状态

- 当前流水线：`2026-09-03-scientific-audit-v1`。
- 数据覆盖：2022-01-01 至 2026-09-01，共 40,920 小时。
- 最近稳定基线 `0476ed0` 曾实测 `135 passed, 4 warnings`；整理前 WIP 快照 `a22b8bb` 已收口，当前 `miao-z1` 全量实测 `170 passed, 4 warnings`（warnings 均为 torch `enable_nested_tensor`，非新增），`outputs/runtime/` 测试前后零新增无主文件，见 `docs/tasks/current.md`。
- 低成本候选已完成 4 折历史滚动回测，评分段止于 2024-12-31。
- 2025-08-22 之后的旧测试段曾被用于模型选择，现只作 development test。
- 预测结果附带版本化分级规则 `configs/risk_grading.yaml`（`qiongyu-advisory-bands@2026-09-03-v1`），属内部辅助阈值，不是官方 AQI。
- 观测推送接口 `POST /api/v1/observations` 需 Bearer 令牌，未配置 `QIONGYU_API_TOKEN` 时自动关闭。
- 部署模型 `multistep_v2` 已用审计后数据重训（模型哈希 `818017da87b7`，训练窗 2022-01-01→2025-02-16，验证窗至 2025-08-22）。
- 除当前 `multistep_v2` 外，其余旧 seed、消融、Wind、图模型和融合 checkpoint 尚未完成同等级验收，不得自动进入当前论文、部署或模型中心“可用”清单。
- 真正 final holdout 尚未收集，现有分数不得称为最终无偏测试成绩。

## 快速开始

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.13 -m pip install -r requirements\modeling.txt
py -3.13 -m pip install -r requirements\app.txt

py -3.13 scripts\audit_scientific_pipeline.py
py -3.13 -m pytest -q
```

正式训练入口固定为：

```text
data/cleaned/training/pollution_2022_2026.csv
data/cleaned/training/weather_2022_2026.csv
```

不要使用 `data/raw/weather.csv` 训练当前模型；该旧文件为 UTC，会造成与北京时间污染数据错位。

## 当前主要结果

10站 Engineered LightGBM 四折宏污染物 R²：

| T+1 | T+2 | T+3 |
|---:|---:|---:|
| 0.886145 | 0.804051 | 0.740090 |

它在全部 12 个“污染物×步长”组合中均为 4/4 折 MAE、R² 优于 Plain。固定9站消融保持同一结论。PM2.5 Residual Huber 在三个步长均为 4/4 折 MAE 优于 Plain 和 Persistence。

完整证据见 `docs/reports/model-experiments.md` 和 `outputs/experiments/rolling_backtest_summary.json`。

## 文档

开发任务从 [AGENTS.md](AGENTS.md) 和 [当前任务](docs/tasks/current.md) 开始：

- [文档索引](docs/README.md)
- [项目概览](docs/project/overview.md)
- [架构与模块](docs/project/architecture.md)
- [项目待办](docs/tasks/backlog.md)
- [2026-09-04 验收报告](docs/reports/acceptance-2026-09-04.md)
- [模型实验记录](docs/reports/model-experiments.md)
- [开发与测试](docs/project/development.md)
- [部署指南](docs/project/deployment.md)

## 目录

```text
app/                 Flask 后端和静态前端
archive/legacy/      已归档的旧实现与独立研究快照，不作为正式入口
configs/             正式、消融和创新模型配置
data/
  source/            原始来源与清单
  cleaned/training/  正式训练输入
  samples/           Web 示例输入
docs/                当前协作文档
models/checkpoints/  模型包和历史 checkpoint
outputs/
  audits/             科学审计 JSON
  experiments/       指标、配置快照和日志
scripts/             数据构建、审计和汇总脚本
src/
  data/               时间轴、缺失、缩放和图构建
  models/             T-GCN/MoE 与创新模型
  training/           统一训练入口
  eval/               baseline、滚动回测和融合
  inference/          模型包加载与批量推理
tests/                数据/模型/滚动边界测试
```

`papers/` 保存本地论文并被 Git 忽略；运行时上传、预测明细和 smoke checkpoint 也不应提交。

## 常用命令

```powershell
# Baseline，默认只评估 validation
py -3.13 src\eval\baselines.py --config configs\multistep_2022_2026.yaml

# 10站与9站四折历史回测
py -3.13 src\eval\rolling_feature_lgbm.py
py -3.13 src\eval\rolling_feature_lgbm.py `
  --drop-station 1344A `
  --output-dir outputs/experiments/feature_lgbm_9station_rolling4_historical

# PM2.5 残差四折回测
py -3.13 src\eval\rolling_pm25_residual_lgbm.py

# 校验三份回测并生成统一汇总
py -3.13 scripts\summarize_rolling_results.py

# 创新模型；默认 eval-split=validation
py -3.13 src\training\train_innovation.py `
  --config configs\innovations\wind_gated_tcn.yaml

# 启动本地 Web
py -3.13 -m app.backend.main
```

## 协作纪律

- 当前最终整合分支为 `miao-z1`；`wind*`、`kun*` 的贡献和吸收决策以 `docs/reports/branch-model-comparison.md` 为准，不直接整分支合并或用跨协议分数替换部署模型。
- 当前工作只以 `docs/tasks/current.md` 和 `docs/tasks/backlog.md` 为准；后续功能使用独立分支，未经明确要求不合并或推送 `master`。
- 不在 development test 上调参，不把 smoke 分数写进论文。
- 新实验必须记录配置、seed、commit、pipeline revision、数据哈希、时间范围、指标和模型哈希。
- 提交前执行 `pytest`、`py_compile`、`git diff --check`。
- 大模型权重优先使用 Git LFS/Release，不要让普通 Git 历史持续膨胀。
