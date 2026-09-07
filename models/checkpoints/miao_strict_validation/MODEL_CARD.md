# miao strict Wind-Gated TCN（受限 validation）

## 身份与适用范围

- 模型 ID：`miao-strict-wind-tcn-v1`；生命周期：批准候选，须显式选择，**不是默认部署模型**。
- 适用：输入完整、连续且希望采用更严格可审计边界的辅助预测场景。
- 不适用：任何超出完整输入边界的请求，或把长沙 validation 结果解释为外地精度。

## 输入、输出与边界

此包同样以特征级 scaler 和调用方站点清单工作，但只接受完整连续输入，以换取更稳定、可审计的运行边界。

- 输入：最近连续 12 小时、每站 8 字段：PM2.5、PM10、NO2、O3、tmp_C、wind_dir、wind_spd、rh。
- 输出：每个输入站点 PM2.5、PM10、NO2、O3 的 T+1、T+2、T+3 小时预测；输出仅供辅助决策。
- 缺失边界：总体缺失不超过 2%，不接受连续整站缺口、整站停报或整列特征缺失。此模型拒绝后，系统只会尝试已验证的 flexible 候选，并在响应中明确 `fallback_reason`。
- API 示例：`POST /api/predict {"session_id":"…","model_id":"miao_strict_validation"}`；可用 `"model_id":"auto"` 按“严格优先、灵活兜底”规则选择。

## 证据、风险与版本

- 证据：`validation_receipt.json` 为单种子、512 个训练窗口与 128 个 validation 窗口的受限验证；`validation_mse=0.324376`、Persistence MSE=`0.104388`，未达长训晋级，不得宣称优于 Persistence。
- 协议：`miao-4poll-10s-12h-v1`，evaluation access 为 validation；future final holdout 明确为 forbidden。
- 风险：完整输入要求不等于跨城市泛化保证；模型不应作为监管、医疗或应急处置的唯一依据。
- 文件哈希（SHA-256）：`model.pt` `82016137292aa8dcbf4b34badc153eee1c29b808c2d7382c0af9c822f5c72ef3`；`scaler.npz` `1ec83362b76f96dde3c9612d26807064042597163c9f5457be9b91eba93cbf1e`；`validation_receipt.json` `c54ad97967888c59072a4f0d1e255ca2f9a6a6c67e4032a79fda4820bcbfea95`。
