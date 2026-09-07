import gradio as gr
import pandas as pd
import numpy as np
import torch
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.zhuyili import HardMOESeason, build_pollution_tensor, build_weather_tensor, get_season_index, build_static_geo_adj, build_dynamic_adj

MODEL_PATH = ROOT_DIR / "models" / "checkpoints" / "tgcn_pm25.pt"

# 设备设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型加载函数
def load_model():
    model = HardMOESeason(in_f=10, poll_dim=6, wx_dim=4, attn_dim=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # 加载训练好的模型
    model.eval()  # 设置为评估模式
    return model


model = load_model()  # 加载模型

# 处理输入的函数
def process_input(pollution_file, weather_file):
    # 读取上传的CSV文件
    poll_df = pd.read_csv(pollution_file.name)
    wx_df = pd.read_csv(weather_file.name)

    # 构建污染数据张量和天气数据张量
    Xp, times = build_pollution_tensor(poll_df)
    Xw = build_weather_tensor(wx_df, times)
    X = np.concatenate([Xp, Xw], axis=2)

    # 数据归一化
    mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
    X = (X - mu) / sd

    # 构建邻接矩阵
    A_static = build_static_geo_adj()
    A_seq = build_dynamic_adj(A_static, Xw)

    # 准备输入数据进行预测
    last = torch.tensor(X[-12:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    last_season = get_season_index(pd.Timestamp(times[-1]))

    with torch.no_grad():
        predictions = model(last, A_seq, torch.tensor([last_season])).cpu().numpy().flatten()

    # 转换为DataFrame并返回
    result_df = pd.DataFrame(predictions, columns=["Prediction"],
                             index=[f"Station {i + 1}" for i in range(len(predictions))])
    return result_df, result_df.to_csv(index=True)

# Gradio界面构建
def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Pollution and Weather CSV Files")
                pollution_file = gr.File(label="Upload Pollution Data CSV")
                weather_file = gr.File(label="Upload Weather Data CSV")

                # 输出元素
                output_table = gr.DataFrame(label="Processed Data")
                output_csv = gr.File(label="Download Result CSV")

                submit_button = gr.Button("Submit")

                submit_button.click(fn=process_input, inputs=[pollution_file, weather_file],
                                    outputs=[output_table, output_csv])

    return demo

# 启动Gradio应用
gr.Interface(fn=process_input, inputs=[gr.File(), gr.File()], outputs=[gr.DataFrame(), gr.File()]).launch(server_port=7861, share=True)
