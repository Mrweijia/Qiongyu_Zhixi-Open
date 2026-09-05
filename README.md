# 穹宇智析（公开发布包）

这是从私有集成仓库按 `configs/public_release_assets.json` 白名单生成的独立公开发布包。它提供可公开的应用、SDK、推理与评估代码、合成演示污染数据、NOAA 气象样例以及已验收的 `multistep_v2` 模型包。

这个仓库刻意不包含真实污染观测、训练语料、CNEMC/官方材料、运行时状态、研究或失效权重，或来源尚未核验的 `src/models/wu.py` 与 `src/models/zhuyili.py`。它不是私有仓的 Git 历史导出。

## 许可证与数据来源

- 项目自有代码、配置、文档和已批准模型包采用 [Apache License 2.0](LICENSE)。
- 合成演示污染数据 `data/samples/pollution.csv` 采用 [CC BY 4.0](DATA_LICENSE)，使用时请保留 `data/samples/SYNTHETIC_DATA.md` 的署名说明。
- `data/samples/weather.csv` 为 NOAA NCEI 的美国政府数据，不由 `DATA_LICENSE` 重新许可；完整署名和排除项见 [NOTICE](NOTICE)。
- 公开资产边界、哈希和模型包组成见 `configs/public_release_assets.json`。

## 内容边界

```text
app/                 Flask 服务与静态前端源码
sdk/                 可独立构建的 Python SDK 与 CLI
src/                 可公开的推理、评估、训练和数据处理源码
data/samples/        合成污染数据与 NOAA 气象样例
models/checkpoints/  已批准的 multistep_v2 模型包
configs/             风险分级、模型注册与公开资产清单
scripts/             公开资产清单验证器
```

模型的训练数据哈希仅用于溯源，不代表训练数据可以下载、重分发或由本仓库恢复。现有结果是 validation/development 证据，不是未来 final holdout 成绩。

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

新增任何文件前，先更新并通过公开资产清单验证。不得将本地缓存、真实观测、私有历史、未授权第三方材料或未批准模型制品加入本仓库。
