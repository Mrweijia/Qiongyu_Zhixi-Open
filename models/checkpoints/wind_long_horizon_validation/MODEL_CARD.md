# wind TCNResidual 长 horizon 候选（受限 validation，未晋级）

这是从 wind 研究方向独立迁移的 9 站 PM2.5 候选：24 小时输入，预测
1/3/6/12/24 小时。它使用当前审计后数据、`pm25-9s-24h-v1` 协议、
feature-level scaler 和当前 validation，而不是 wind 旧 test 分数或旧权重。

- 证据：512 个训练窗口、128 个 validation 窗口、单种子 2 epoch；完整输入边界和数据哈希见 `validation_receipt.json`。
- 晋级门槛：同一 validation 的平均 MSE 必须至少低于 Persistence 1%。
- 结果：TCN MSE 0.604093，高于 Persistence 0.402010，`advance_to_long_training` 为 `false`。
- 结论：本包只保留为可复验的未晋级候选，不得部署、不得写作最终成绩，也不得据 wind 历史 test 成绩重新晋级。
