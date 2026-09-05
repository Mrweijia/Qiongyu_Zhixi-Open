# Web 示例数据说明

`pollution.csv` 是合成演示数据，不来自真实监测记录，也不可还原为真实观测。

- 生成脚本：`../scripts/generate_synthetic_sample.py`
- 生成方式：固定随机种子 `20260903`，覆盖与 `weather.csv` 同一天对齐的 2025-05-17 至 2025-05-18 连续 48 小时。
- 日期选择仅用于让示例上传流程与 NOAA 气象示例重合；日期上的数值仍全部为合成值。
- 结构：`date,hour,type,1335A,...,1344A`，每行包含 PM2.5、PM10、SO2、NO2、O3、CO 中的一个污染物类型。
- 用途：仅用于 Web 演示和接口流程验证。

`weather.csv` 来自 NOAA NCEI（美国国家海洋和大气管理局国家环境信息中心），允许公开使用，使用需保留 NOAA NCEI 署名。
