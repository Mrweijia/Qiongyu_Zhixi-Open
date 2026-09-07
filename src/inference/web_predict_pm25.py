import os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "checkpoints" / "tgcn_pm25.pt"

# ==== 配置参数 ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEL_STATIONS = [
    "1335A","1336A","1337A","1338A","1339A",
    "1340A","1341A","1342A","1343A","1344A"
]
POLL_TYPES = ["PM2.5","PM10","SO2","NO2","O3","CO"]
WX_TYPES = ["tmp_C", "wind_dir", "wind_spd", "rh"]  # 气象特征顺序
INPUT_STEPS = 12

STATION_COORDS = {
    "1335A": (113.0833,28.2325), "1336A": (112.8872,28.2189),
    "1337A": (113.0792,28.2053), "1338A": (112.9394,28.1900),
    "1339A": (113.0178,28.1322), "1340A": (112.9792,28.2597),
    "1341A": (113.0014,28.1944), "1342A": (112.9840,28.1178),
    "1343A": (112.8908,28.1308), "1344A": (112.9581,28.3611)
}
WEATHER_STATIONS = {
    "57687099999": (28.116666,112.783333),
    "592871999999": (28.189158,113.219633)
}

# ======= 网络结构（与训练保持一致）=======
class FeatureAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels))
    def forward(self, x):
        return x * self.alpha

class TemporalSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, 2, batch_first=True)
    def forward(self, x):
        B, T, N, F = x.shape
        x_flat = x.permute(0,2,1,3).reshape(B*N, T, F)
        out, _ = self.attn(x_flat, x_flat, x_flat)
        out = out.reshape(B, N, T, F).permute(0,2,1,3)
        return out

class CrossAttention(nn.Module):
    def __init__(self, poll_dim, wx_dim, attn_dim=8):
        super().__init__()
        self.poll_proj = nn.Linear(poll_dim, attn_dim)
        self.wx_proj = nn.Linear(wx_dim, attn_dim)
        self.cross = nn.MultiheadAttention(attn_dim, 2, batch_first=True)
        self.fc = nn.Linear(poll_dim+attn_dim, poll_dim+attn_dim)
        self.attn_dim = attn_dim
    def forward(self, poll, wx):
        B,T,N,P = poll.shape
        _,_,_,W = wx.shape
        poll_f = poll.permute(0,2,1,3).reshape(B*N,T,P)
        wx_f   = wx.permute(0,2,1,3).reshape(B*N,T,W)
        poll_f_proj = self.poll_proj(poll_f)
        wx_f_proj = self.wx_proj(wx_f)
        attn_out, _ = self.cross(poll_f_proj, wx_f_proj, wx_f_proj)
        attn_out = attn_out.reshape(B,N,T,self.attn_dim).permute(0,2,1,3)
        concat = torch.cat([poll, attn_out], dim=-1)
        return self.fc(concat)

class GraphConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.lin=nn.Linear(in_ch,out_ch)
    def forward(self,X,A):
        return torch.relu(self.lin(A@X))

class TGCN_with_Attn(nn.Module):
    def __init__(self,in_f,g_h=32,gru_h=32, poll_dim=6, wx_dim=4, attn_dim=8):
        super().__init__()
        self.feature_attn = FeatureAttention(in_f)
        self.temporal_attn = TemporalSelfAttention(in_f)
        self.cross_attn = CrossAttention(poll_dim=poll_dim, wx_dim=wx_dim, attn_dim=attn_dim)
        self.gcn = GraphConv(poll_dim+attn_dim, g_h)
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        self.fc  = nn.Linear(gru_h, 1)
        self.poll_dim = poll_dim
        self.wx_dim = wx_dim
    def forward(self, x, A_seq):
        poll = x[...,:self.poll_dim]
        wx   = x[...,-self.wx_dim:]
        x_fa = self.feature_attn(x)
        x_ta = self.temporal_attn(x_fa)
        x_cross = self.cross_attn(poll, wx)
        B,T,N,F_new = x_cross.shape
        gcn_out = []
        for t in range(T):
            A_t = A_seq[t]
            h_t = [self.gcn(x_cross[b,t],A_t) for b in range(B)]
            gcn_out.append(torch.stack(h_t))
        gcn_out = torch.stack(gcn_out,1)
        bnf = gcn_out.permute(0,2,1,3).reshape(B*N,T,-1)
        _,h_n=self.gru(bnf)
        return self.fc(h_n.squeeze(0)).reshape(B,N)

class HardMOESeason(nn.Module):
    def __init__(self, in_f, g_h=32, gru_h=32, poll_dim=6, wx_dim=4, attn_dim=8, n_exp=4):
        super().__init__()
        self.experts = nn.ModuleList([
            TGCN_with_Attn(in_f, g_h, gru_h, poll_dim, wx_dim, attn_dim) for _ in range(n_exp)
        ])
    def forward(self, x, A_seq, season_ids):
        outs = []
        for i in range(x.shape[0]):
            s = int(season_ids[i])
            outs.append(self.experts[s](x[i:i+1],A_seq))
        return torch.cat(outs,0)

# ========= 构建邻接矩阵 =========
def haversine(lat1,lon1,lat2,lon2):
    R=6371.0
    φ1,φ2=map(math.radians,(lat1,lat2))
    dφ=math.radians(lat2-lat1); dλ=math.radians(lon2-lon1)
    a=math.sin(dφ/2)**2+math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_static_geo_adj():
    N=len(SEL_STATIONS)
    A=np.zeros((N,N),dtype=np.float32)
    coords=[STATION_COORDS[s] for s in SEL_STATIONS]
    for i,(lon1,lat1) in enumerate(coords):
        for j,(lon2,lat2) in enumerate(coords):
            d=haversine(lat1,lon1,lat2,lon2)
            if d<=50:
                A[i,j]=math.exp(-d*d/(2*20**2))
    np.fill_diagonal(A,1.0)
    D_inv=np.diag(1/np.sqrt(A.sum(1)+1e-6))
    return D_inv @ A @ D_inv

def build_dynamic_adj(static_A, Xw):
    T,N,_ = Xw.shape
    coords = [STATION_COORDS[s] for s in SEL_STATIONS]
    ang_ij = np.zeros((N,N), dtype=np.float32)
    for i, (lon1,lat1) in enumerate(coords):
        for j, (lon2,lat2) in enumerate(coords):
            ang_ij[i,j] = math.atan2(lat2-lat1, lon2-lon1)
    A_seq = np.zeros((T,N,N), dtype=np.float32)
    for t in range(T):
        flow_rad = np.deg2rad((Xw[t,:,1] + 180) % 360)
        align = np.maximum(0, np.cos(flow_rad[None, :] - ang_ij))
        A_t = static_A * align
        np.fill_diagonal(A_t, 1.0)
        row_sum = A_t.sum(axis=1, keepdims=True) + 1e-6
        A_seq[t] = A_t / row_sum
    return torch.tensor(A_seq, device=DEVICE)

def get_season_index(ts):
    m = ts.month
    if m in [12,1,2]: return 0
    if m in [3,4,5]:  return 1
    if m in [6,7,8]:  return 2
    return 3

# ========= 交互式表格输入、批量推理 =========
def predict_by_tables(poll_table, wx_table, input_time, input_month):
    """
    poll_table: DataFrame, shape=(INPUT_STEPS, 10*6) 每小时所有站点的6类污染物
    wx_table: DataFrame, shape=(INPUT_STEPS, 10*4) 每小时所有站点的4类气象
    input_time: 预测时刻（字符串如'2025-07-15 09:00'）
    input_month: 预测时刻月份
    """
    try:
        Xp = np.array(poll_table).reshape(INPUT_STEPS, len(SEL_STATIONS), len(POLL_TYPES)).astype(np.float32)
        Xw = np.array(wx_table).reshape(INPUT_STEPS, len(SEL_STATIONS), len(WX_TYPES)).astype(np.float32)
        # 标准化（与训练一致，建议保存均值/方差，下面假设用输入本身的统计量归一化）
        mu = np.mean(np.concatenate([Xp,Xw],axis=2), axis=(0,1), keepdims=True)
        sd = np.std(np.concatenate([Xp,Xw],axis=2), axis=(0,1), keepdims=True) + 1e-6
        X = (np.concatenate([Xp, Xw], axis=2) - mu) / sd
        # 构建动态邻接
        A_static = build_static_geo_adj()
        A_seq = build_dynamic_adj(A_static, Xw)
        # 构造模型
        model = HardMOESeason(
            in_f=X.shape[2], poll_dim=Xp.shape[2], wx_dim=Xw.shape[2], attn_dim=8
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        # 预测
        input_month = int(input_month)
        last_season = 0 if input_month in [12,1,2] else 1 if input_month in [3,4,5] else 2 if input_month in [6,7,8] else 3
        last = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pr = model(last, A_seq, torch.tensor([last_season])).cpu().numpy().flatten()
        # 反归一化，只输出PM2.5
        pm25_mu = Xp.mean((0,1))
        pm25_sd = Xp.std((0,1))+1e-6
        pm25_pred = pr * pm25_sd[0] + pm25_mu[0]
        df_out = pd.DataFrame({
            "站点": SEL_STATIONS,
            "PM2.5预测值(μg/m³)": pm25_pred.round(2)
        })
        return df_out
    except Exception as e:
        return f"❌ 数据格式错误或模型异常: {str(e)}"

# ==== Gradio 网页构建 ====
table_header_poll = []
for h in range(INPUT_STEPS):
    for s in SEL_STATIONS:
        for p in POLL_TYPES:
            table_header_poll.append(f"{s}-{p}-t{h+1}")
table_header_wx = []
for h in range(INPUT_STEPS):
    for s in SEL_STATIONS:
        for w in WX_TYPES:
            table_header_wx.append(f"{s}-{w}-t{h+1}")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber")) as demo:
    gr.Markdown("""
    # 🌏 T-GCN PM2.5批量预测智能系统
    **功能说明**  
    - 支持自定义输入任意12小时污染物与气象历史数据，10个站点一键批量预测下一小时PM2.5。  
    - 只需复制/填写下方表格数据，即可获得预测结果，适合教学、竞赛或实际业务场景。
    """)
    with gr.Accordion("1. 输入：近12小时污染物数据（每站点每小时6项）", open=True):
        poll_table = gr.Dataframe(headers=[f"{s}-{p}" for s in SEL_STATIONS for p in POLL_TYPES], 
                                  datatype="number", 
                                  row_count=INPUT_STEPS, 
                                  label="污染物(μg/m³) 输入表格（10站*6污染物）")
    with gr.Accordion("2. 输入：近12小时气象数据（每站点每小时4项）", open=False):
        wx_table = gr.Dataframe(headers=[f"{s}-{w}" for s in SEL_STATIONS for w in WX_TYPES],
                               datatype="number",
                               row_count=INPUT_STEPS,
                               label="气象输入表格（10站*4气象）")
    with gr.Row():
        input_time = gr.Textbox(label="预测时刻（如2025-07-15 09:00）", value="2025-07-15 09:00")
        input_month = gr.Textbox(label="预测月份", value="7")
    with gr.Row():
        btn = gr.Button("开始预测（批量PM2.5）")
    output = gr.Dataframe(label="预测结果（各站点PM2.5 μg/m³）", headers=["站点","PM2.5预测值(μg/m³)"])
    btn.click(fn=predict_by_tables, inputs=[poll_table, wx_table, input_time, input_month], outputs=output)

    gr.Markdown("""
    > 🧑‍💻 站点顺序/污染物顺序/气象顺序须与训练保持一致，默认12小时窗口，结果更可靠。<br>
    > 🌟 建议输入真实或仿真的历史序列，可用于城市空气质量管理、科学研究和AI模型竞赛展示。
    """)

if __name__ == "__main__":
    demo.launch(share=True)
