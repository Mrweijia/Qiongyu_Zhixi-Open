# 穹宇智析（公开发布包）

这是从私有集成仓库按 `configs/public_release_assets.json` 白名单生成的独立公开发布包。它提供可公开的应用、SDK、推理与评估代码、合成演示污染数据、NOAA 气象样例、已验收的 `multistep_v2` 模型包，以及面向用户的**展示材料（项目信息表）**、**全部真实模型包**与**完整训练模型代码**。

这个仓库刻意不包含真实污染观测、训练语料、CNEMC/官方材料、运行时状态、smoke 测试权重，以及视频 / PPT / 文档的生成代码。它不是私有仓的 Git 历史导出。

## 许可证与数据来源

- 项目自有代码、配置、文档和模型包采用 [Apache License 2.0](LICENSE)。
- 合成演示污染数据 `data/samples/pollution.csv` 采用 [CC BY 4.0](DATA_LICENSE)，使用时请保留 `data/samples/SYNTHETIC_DATA.md` 的署名说明。
- `data/samples/weather.csv` 为 NOAA NCEI 的美国政府数据，不由 `DATA_LICENSE` 重新许可；完整署名和排除项见 [NOTICE](NOTICE)。
- 公开资产边界、哈希和模型包组成见 `configs/public_release_assets.json`（该清单覆盖首批核心资产；本次新增的 presentation/、docs/、扩展模型包与 `src/models/wu.py` 等以本 README 为准）。

## 内容总览

```text
app/                 Flask 服务与静态前端源码
sdk/                 可独立构建的 Python SDK 与 CLI
src/                 推理、评估、训练和数据处理源码
src/models/          模型定义（含 wu.py 统一训练模型、zhuyili.py 基线、transfer.py 迁移）
src/training/        训练入口（foundation / tgcn / moe_tgcn / innovation / multistep）
data/samples/        合成污染数据与 NOAA 气象样例
models/checkpoints/  全部真实模型包（multistep_v2 及其种子复现、wind_gated_tcn 系列、消融与验证包）
configs/             风险分级、模型注册与公开资产清单
scripts/             公开资产清单验证器
presentation/        面向用户的展示成品：项目信息表
docs/visualizations/ 模型进展指标图（哑铃图 / ΔR² 热力图 / 站点稳健性 / 误差分布等 8 张）
```

## 展示材料（面向用户）

```text
presentation/项目信息表.html/.md       项目信息表（团队、分工、技术要点）
```

## 模型包

`models/checkpoints/` 收录全部真实（非 smoke）模型包，均为小体积 CPU 可推理制品：

- `multistep_v2`（已验收部署包）及 `multistep_v2_seed42` / `multistep_v2_seed123`（种子复现）
- `multistep_2022_2026`（2022–2026 训练窗版本）
- `wind_gated_tcn` 系列（含种子复现）与 `wind_long_horizon_validation`、`validation_blend`
- `ablation_no_attention` / `ablation_single_expert` / `ablation_static_adj`（消融对照）
- `causal_gat`、`miao_flexible_validation`、`miao_strict_validation`

各包内含 `MODEL_CARD.md`，记录口径、指标与哈希。模型的训练数据哈希仅用于溯源，不代表训练数据可以下载、重分发或由本仓库恢复。现有结果是 validation/development 证据，不是未来 final holdout 成绩。

## 训练

模型定义与训练入口已随本包公开：

- 模型定义：`src/models/wu.py`（统一训练 wu_hybrid 主线）、`src/models/wu_v2.py`、`src/models/zhuyili.py`（基线）、`src/models/transfer.py`（迁移）、`src/models/innovations/`
- 训练入口：`src/training/train_multistep.py`（多步直接预测）、`train_innovation.py`、`train_moe_tgcn.py`、`train_tgcn.py`、`pretrain_foundation.py`、`run_tgcn_night.py`

训练需要真实观测数据（不随本仓库分发）；使用 `data/samples/` 的合成数据可以跑通代码路径但无法复现论文级指标。

## 验证公开资产

使用 Python 3.9+ 运行：

```powershell
python scripts/validate_public_release_assets.py
```

该命令会校验白名单中的文件、许可证文本、必要的 NOAA 署名和已批准模型包的哈希。它不会下载、生成或验证未公开的训练数据。

## SDK

SDK 可独立安装；它是调用已部署 API 的客户端，不携带服务端依赖：

```powershell
python -m pip install ./sdk
qiongyu --help
```

服务端运行和完整实验需要与运行环境匹配的依赖及受控数据；这些不作为本次公开发布包的一部分。

## 贡献与发布

新增任何文件前，先更新并通过公开资产清单验证。不得将本地缓存、真实观测、私有历史、未授权第三方材料或视频 / PPT / 文档生成代码加入本仓库。
