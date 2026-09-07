# miao flexible Wind-Gated TCN（受限 validation）

## 身份与适用范围

- 模型 ID：`miao-flexible-wind-tcn-v1`；生命周期：批准候选，须显式选择，**不是默认部署模型**。
- 适用：输入存在少量缺失或短暂站点断流、仍希望得到辅助预测的场景。
- 不适用：跨城市精度结论、官方空气质量发布、缺失超过下述边界或无有效图结构的调用。

## 输入、输出与边界

此包使用特征级缩放器，可接收调用方提供的任意非空、唯一站点清单和同行列归一化邻接矩阵；没有把长沙站点 ID 写入权重或 scaler。

- 输入：最近连续 12 小时、每站 8 字段：PM2.5、PM10、NO2、O3、tmp_C、wind_dir、wind_spd、rh。
- 输出：每个输入站点 PM2.5、PM10、NO2、O3 的 T+1、T+2、T+3 小时预测；输出仅供辅助决策。
- 缺失边界：总体缺失不超过 30%，任一站连续全缺失不超过 3 小时；容许整站/整列缺失。超过边界会被拒绝，不会用填补结果伪装为完整观测。
- API 示例：`POST /api/predict {"session_id":"…","model_id":"miao_flexible_validation"}`；响应的 `meta.actual_model`、`meta.capability_sha256` 与 `meta.input_quality` 用于追溯。

## 证据、风险与版本

- 证据：`validation_receipt.json` 为单种子、512 个训练窗口与 128 个 validation 窗口的受限验证；`validation_mse=0.341867`、Persistence MSE=`0.116454`，未达长训晋级，不得宣称优于 Persistence。
- 协议：`miao-4poll-10s-12h-v1`，evaluation access 为 validation；future final holdout 明确为 forbidden。
- 风险：缺失容忍代表接口边界而非跨城市精度保证；模型不应作为监管、医疗或应急处置的唯一依据。
- 文件哈希（SHA-256）：`model.pt` `170ef3de5f13a79285808098868a8109435dd3c5a24f7547f73fe37a21cdb1b4`；`scaler.npz` `d9d73276819d477b613b96b1c8ab2b2e24a11d1e2f608d42d7b12e33b66a0105`；`validation_receipt.json` `69c720bf530a72787e7a7bf09171c64f2e9d20163c000ff828e8552dc5aaad9f`。
