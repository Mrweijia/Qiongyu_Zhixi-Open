import gradio as gr
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.wu import build_pollution_tensor, build_weather_tensor, build_static_geo_adj, build_dynamic_adj, PRED_IDX, SEL_STATIONS, PRED_NAMES, HardMOESeason

MODEL_PATH = ROOT_DIR / "models" / "checkpoints" / "tgcn_four_out.pt"
OUTPUT_DIR = ROOT_DIR / "outputs" / "runtime"

# 定义加载模型的函数
def load_model():
    model = HardMOESeason(in_f=10, poll_dim=6, wx_dim=4, attn_dim=8, n_out=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 设置为评估模式
    return model

# 使用CPU（模型会在CPU上运行）
device = 'cpu'
model = load_model().to(device)

# 标准化函数
def standardize_input(X):
    mu, sd = X.mean((0,1), keepdims=True), X.std((0,1), keepdims=True) + 1e-6
    return (X - mu) / sd, mu, sd

# 预测函数
def predict(pollution_file, weather_file):
    print("文件加载中...")
    pollution_df = pd.read_csv(pollution_file.name)
    weather_df = pd.read_csv(weather_file.name)

    # 获取数据的最后12小时（假设数据按时间排序）
    pollution_df['datetime'] = pd.to_datetime(pollution_df['date'].astype(str) + ' ' + pollution_df['hour'].astype(str), format="%Y%m%d %H")
    weather_df['datetime'] = pd.to_datetime(weather_df['DATE'], format="%Y-%m-%d %H:%M:%S")

    # 获取最后12小时的数据
    pollution_df = pollution_df[pollution_df['datetime'] >= pollution_df['datetime'].max() - pd.Timedelta(hours=12)]
    weather_df = weather_df[weather_df['datetime'] >= weather_df['datetime'].max() - pd.Timedelta(hours=12)]

    # 生成张量数据
    Xp, times = build_pollution_tensor(pollution_df)
    Xw = build_weather_tensor(weather_df, times)

    # 合并和标准化数据
    X = np.concatenate([Xp, Xw], axis=2)
    X, mu, sd = standardize_input(X)

    # 邻接矩阵
    A_static = build_static_geo_adj()
    A_seq = build_dynamic_adj(A_static, Xw)

    # 将输入数据转为Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        predictions = model(X_tensor, A_seq)
        predictions = predictions.cpu().numpy()

    # 反归一化
    pred_denorm = predictions[0] * sd[0, 0, PRED_IDX] + mu[0, 0, PRED_IDX]

    # 格式化输出为字典
    result = {}
    for i, station in enumerate(SEL_STATIONS):
        result[station] = {PRED_NAMES[j]: pred_denorm[i, j] for j in range(len(PRED_IDX))}

    # 将结果转换为DataFrame以便生成CSV文件
    result_df = pd.DataFrame(result).transpose()
    result_df['Station'] = result_df.index
    result_df = result_df[['Station'] + PRED_NAMES]  # 调整列顺序

    # 保存结果为CSV文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / 'forecast_results.csv'
    result_df.to_csv(output_csv, index=False)

    # 确保文件存在
    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"{output_csv} not found.")

    # 返回HTML结果和CSV文件路径
    return result_df.to_html(), output_csv
    #return result_df, output_csv

# 设置Gradio接口
iface = gr.Interface(
    fn=predict,
    inputs=[gr.File(label="污染数据CSV文件"),gr.File(label="天气数据CSV文件")],
    outputs=[gr.HTML(label="预测结果"), gr.File(label="下载CSV文件")],  # 确保这两者匹配
    live=True,
    title="空气质量预测模型",
    description="上传包含过去12小时数据的CSV文件，模型将预测未来所有站点的空气污染物浓度。",
)

iface.launch()
