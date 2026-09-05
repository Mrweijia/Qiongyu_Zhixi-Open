"""
MoE-TGCN 空气质量预测模型
结合气象条件(温度、风向、风速、湿度)和时间信息进行无监督专家分配
"""

import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import gumbel_softmax
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "moe_tgcn"

# ─── Config ────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 监测站点和污染物类型
SEL_STATIONS = [
    "1335A", "1336A", "1337A", "1338A", "1339A",
    "1340A", "1341A", "1342A", "1343A", "1344A"
]
POLL_TYPES = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]

# 时空参数
START_DATE = "2024-12-01"
END_DATE = "2025-02-28"
INPUT_STEPS = 12  # 输入时间步长(小时)
PRED_HORIZON = 1  # 预测步长(小时)

# 模型参数
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EXPERTS = 4  # 专家数量

# 地理参数
DIST_THRESHOLD_KM = 50
GAUSS_SIGMA_KM = 20

# 站点坐标
STATION_COORDS = {
    "1335A": (113.0833, 28.2325), "1336A": (112.8872, 28.2189),
    "1337A": (113.0792, 28.2053), "1338A": (112.9394, 28.1900),
    "1339A": (113.0178, 28.1322), "1340A": (112.9792, 28.2597),
    "1341A": (113.0014, 28.1944), "1342A": (112.9840, 28.1178),
    "1343A": (112.8908, 28.1308), "1344A": (112.9581, 28.3611)
}

# 气象站坐标
WEATHER_STATIONS = {
    "57687099999": (28.116666, 112.783333),
    "592871999999": (28.189158, 113.219633)
}


# ────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """计算两个经纬度坐标之间的距离(km)"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ─── 数据加载与处理 ────────────────────────────────────────────────────────
def load_pollution(path):
    """加载污染数据"""
    df = pd.read_csv(path, dtype=str)
    df['date_num'] = pd.to_numeric(df['date'], errors='coerce')
    df['hour_num'] = pd.to_numeric(df['hour'], errors='coerce')
    df.dropna(subset=['date_num', 'hour_num'], inplace=True)
    # df['datetime'] = (
    #         pd.to_datetime(df['date_num'].astype(int).astype(str), format="%Y%m%d", errors='coerce') +
    #         pd.to_timedelta(df['hour_num'].astype(int), unit='h')
    #
    # df.dropna(subset=['datetime'], inplace=True)
    df['datetime'] = (
            pd.to_datetime(df['date_num'].astype(int).astype(str),
                           format="%Y%m%d", errors='coerce')
            + pd.to_timedelta(df['hour_num'].astype(int), unit='h')
    )
    df.dropna(subset=['datetime'], inplace=True)

    val_cols = [c for c in df.columns if c not in ['date', 'hour', 'date_num', 'hour_num', 'datetime', 'type']]
    long = df.melt(id_vars=['datetime', 'type'], value_vars=val_cols, var_name='site', value_name='value')
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    return long.dropna(subset=['value'])


def build_pollution_tensor(df):
    """构建污染数据张量"""
    df = df[df['site'].isin(SEL_STATIONS) & df['type'].isin(POLL_TYPES)]
    tidy = df.pivot_table(index=['datetime', 'site'], columns='type', values='value').reset_index()

    mask = (tidy['datetime'] >= pd.to_datetime(START_DATE)) & \
           (tidy['datetime'] <= pd.to_datetime(END_DATE))
    tidy = tidy.loc[mask].sort_values(['datetime', 'site'])
    tidy[POLL_TYPES] = tidy.groupby('site')[POLL_TYPES].transform(lambda g: g.ffill().bfill())

    times = sorted(tidy['datetime'].unique())
    tmap = {t: i for i, t in enumerate(times)}
    smap = {s: i for i, s in enumerate(SEL_STATIONS)}

    T, N, P = len(times), len(SEL_STATIONS), len(POLL_TYPES)
    Xp = np.zeros((T, N, P), dtype=np.float32)

    for _, r in tidy.iterrows():
        ti, si = tmap[r['datetime']], smap[r['site']]
        for pi, pol in enumerate(POLL_TYPES):
            Xp[ti, si, pi] = r.get(pol, np.nan)

    return np.nan_to_num(Xp, nan=0.0), times


def load_weather(path):
    """加载气象数据"""
    df = pd.read_csv(path, dtype=str)
    df['datetime'] = pd.to_datetime(df['DATE'], errors='coerce')
    df.dropna(subset=['datetime', 'STATION'], inplace=True)

    # 温度处理
    df['tmp_C'] = pd.to_numeric(df['TMP'].str.split(',').str[0], errors='coerce') / 10.0

    # 风向风速处理
    def parse_wnd(s):
        p = str(s).split(',')
        d = float(p[0]) if p[0].replace('.', '', 1).isdigit() else np.nan
        sp = float(p[3]) / 10.0 if len(p) > 3 and p[3].isdigit() else np.nan
        return d, sp

    df['wind_dir'], df['wind_spd'] = zip(*df['WND'].apply(
        lambda x: parse_wnd(x) if isinstance(x, str) else (np.nan, np.nan)))

    # 相对湿度处理
    if 'DEWP' in df.columns:
        df['dew_C'] = pd.to_numeric(df['DEWP'].str.split(',').str[0], errors='coerce') / 10.0

        def calc_rh(t, td):
            if np.isnan(t) or np.isnan(td): return np.nan
            a, b = 17.27, 237.7
            return 100 * math.exp(a * td / (b + td) - a * t / (b + t))

        df['rh'] = df.apply(lambda r: calc_rh(r['tmp_C'], r['dew_C']), axis=1)
    else:
        df['rh'] = np.nan

    df = df.sort_values(['STATION', 'datetime'])
    df[['tmp_C', 'wind_dir', 'wind_spd', 'rh']] = \
        df.groupby('STATION')[['tmp_C', 'wind_dir', 'wind_spd', 'rh']].transform(
            lambda s: s.interpolate().ffill().bfill())

    return df[['datetime', 'STATION', 'tmp_C', 'wind_dir', 'wind_spd', 'rh']]


def build_weather_tensor(wx, times):
    """构建气象数据张量"""
    stations_wx = list(WEATHER_STATIONS.keys())
    M, N = len(stations_wx), len(SEL_STATIONS)

    # 距离权重矩阵
    D = np.zeros((N, M), dtype=np.float32)
    mon_xy = [STATION_COORDS[s] for s in SEL_STATIONS]
    wx_xy = [WEATHER_STATIONS[s] for s in stations_wx]

    for i, (lon, lat) in enumerate(mon_xy):
        ds = np.array([haversine(lat, lon, lat2, lon2) for lat2, lon2 in wx_xy], dtype=np.float32)
        inv = 1 / (ds + 1e-6)
        D[i] = inv / inv.sum()

    # 构建气象特征
    W_list = []
    for t in times:
        df_t = wx[wx['datetime'] == t]
        Wm = np.zeros((M, 4), dtype=np.float32)
        for m, st in enumerate(stations_wx):
            r = df_t[df_t['STATION'] == st]
            if not r.empty:
                row = r.iloc[0]
                Wm[m] = [row['tmp_C'], row['wind_dir'], row['wind_spd'], row['rh']]
        W_list.append(Wm)

    W_np = np.stack(W_list, axis=0)  # [T, M, 4]
    Xw = np.stack([D.dot(W_np[t]) for t in range(len(times))], axis=0)  # [T, N, 4]
    return np.nan_to_num(Xw, nan=0.0)


def build_weather_time_features(wx, times):
    """构建用于门控的气象时间特征"""
    features = []
    for t in times:
        df_t = wx[wx['datetime'] == t]
        if not df_t.empty:
            row = df_t.iloc[0]  # 使用第一个气象站数据
            hour = t.hour / 23.0  # 归一化到[0,1]
            features.append([
                row['tmp_C'] / 40.0,  # 假设最大40°C
                row['wind_dir'] / 360.0,
                row['wind_spd'] / 20.0,  # 假设最大20m/s
                row['rh'] / 100.0,
                hour
            ])
        else:
            features.append([0.0] * 5)  # 缺失值填充
    return np.array(features, dtype=np.float32)  # [T, 5]


# ─── 模型定义 ──────────────────────────────────────────────────────────────
class GraphAttentionConv(nn.Module):
    """图注意力卷积层"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, X, A):
        H = self.lin(X)
        S = (H @ H.T) / math.sqrt(H.size(1))
        S = S.masked_fill(A <= 0, float('-inf'))
        α = torch.softmax(S, dim=1)
        return α @ H


class Expert(nn.Module):
    """专家模块(每个专家是一个独立的T-GCN变体)"""

    def __init__(self, in_f, g_h=16, gru_h=16):
        super().__init__()
        self.gcn = GraphAttentionConv(in_f, g_h)
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        self.fc = nn.Linear(gru_h, 1)

    def forward(self, x, A):
        B, T, N, _ = x.shape
        gcn_out = []
        for t in range(T):
            h_t = [self.gcn(x[b, t], A) for b in range(B)]
            gcn_out.append(torch.stack(h_t))
        gcn_out = torch.stack(gcn_out, 1)
        bnf = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, h_n = self.gru(bnf)
        return self.fc(h_n.squeeze(0)).reshape(B, N)


class MoE_TGCN(nn.Module):
    """MoE-TGCN主模型"""

    def __init__(self, in_f, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(in_f) for _ in range(num_experts)])

        # 门控网络(输入: 温度, 风向, 风速, 湿度, 小时)
        self.gate_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts)
        )

        # 专家负载均衡参数
        self.load_balance_loss = 0
        self.alpha = 0.01  # 负载均衡系数

    def forward(self, x, A, weather_time_features):
        """
        Args:
            x: 输入数据 [B, T, N, F]
            A: 邻接矩阵 [N, N]
            weather_time_features: [B, 5] (temp, wind_dir, wind_spd, rh, hour)
        """
        # 计算门控权重(使用Gumbel-Softmax)
        logits = self.gate_net(weather_time_features)
        gates = gumbel_softmax(logits, tau=0.5, dim=-1)  # [B, num_experts]

        # 计算专家负载均衡损失
        expert_activations = gates.mean(0)
        self.load_balance_loss = self.alpha * (expert_activations.std() / expert_activations.mean())

        # 各专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x, A))  # 每个输出 [B, N]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [B, N, num_experts]

        # 加权组合
        output = torch.einsum('bnk,bk->bn', expert_outputs, gates)  # [B, N]

        return output


# ─── 数据集定义 ─────────────────────────────────────────────────────────────
class WindowDataset(torch.utils.data.Dataset):
    """时空滑动窗口数据集"""

    def __init__(self, X, X_weather_time):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_weather_time = torch.tensor(X_weather_time, dtype=torch.float32)
        self.L = X.shape[0] - INPUT_STEPS - PRED_HORIZON + 1

    def __len__(self):
        return max(self.L, 0)

    def __getitem__(self, i):
        x = self.X[i:i + INPUT_STEPS]
        weather_time = self.X_weather_time[i + INPUT_STEPS - 1]  # 使用最后一个时间步的气象数据
        y = self.X[i + INPUT_STEPS + PRED_HORIZON - 1, :, 0]
        return x, weather_time, y


# ─── 主函数 ────────────────────────────────────────────────────────────────
def main():
    # 1) 加载污染数据
    pollution_path = DATA_DIR / "pollution.csv"
    weather_path = DATA_DIR / "weather.csv"
    if not pollution_path.exists():
        print(f"❌ 找不到 pollution.csv，请检查 {DATA_DIR}")
        return
    poll_df = load_pollution(pollution_path)
    Xp, times = build_pollution_tensor(poll_df)
    print("污染数据形状:", Xp.shape)  # [T, N, P]

    # 2) 加载气象数据
    if not weather_path.exists():
        print(f"❌ 找不到 weather.csv，请检查 {DATA_DIR}")
        return
    wx_df = load_weather(weather_path)
    Xw = build_weather_tensor(wx_df, times)
    X_weather_time = build_weather_time_features(wx_df, times)
    print("气象数据形状:", Xw.shape)  # [T, N, 4]
    print("气象时间特征形状:", X_weather_time.shape)  # [T, 5]

    # 3) 数据合并与标准化
    X = np.concatenate([Xp, Xw], axis=2)  # [T, N, 10]
    mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6
    X = (X - mu) / sd
    print("合并后数据形状:", X.shape)

    # 4) 构建邻接矩阵
    def build_geo_adj(sts):
        N = len(sts)
        A = np.zeros((N, N), dtype=np.float32)
        xy = [STATION_COORDS[s] for s in sts]
        for i in range(N):
            for j in range(N):
                d = haversine(xy[i][1], xy[i][0], xy[j][1], xy[j][0])
                if d <= DIST_THRESHOLD_KM:
                    A[i, j] = math.exp(-d * d / (2 * GAUSS_SIGMA_KM ** 2))
        np.fill_diagonal(A, 1.0)
        D_inv = np.diag(1 / np.sqrt(A.sum(1) + 1e-6))
        return D_inv @ A @ D_inv

    A = torch.tensor(build_geo_adj(SEL_STATIONS), device=DEVICE)

    # 5) 准备数据集
    ds = WindowDataset(X, X_weather_time)
    ntr = int(0.8 * len(ds))
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [ntr, len(ds) - ntr])
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, batch_size=BATCH_SIZE)
    print(f"训练样本: {len(tr_ds)}, 验证样本: {len(vl_ds)}")

    # 6) 初始化模型
    model = MoE_TGCN(in_f=X.shape[2], num_experts=NUM_EXPERTS).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    tr_losses, vl_losses = [], []

    # 7) 训练循环
    print("\n开始训练...")
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0
        for x, weather_time, y in tr_ld:
            x, weather_time, y = x.to(DEVICE), weather_time.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            p = model(x, A, weather_time)
            loss = loss_fn(p, y) + model.load_balance_loss  # 添加负载均衡损失
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
        tr_losses.append(tl / len(tr_ld.dataset))

        # 验证循环
        model.eval()
        vl = 0
        with torch.no_grad():
            for x, weather_time, y in vl_ld:
                x, weather_time, y = x.to(DEVICE), weather_time.to(DEVICE), y.to(DEVICE)
                p = model(x, A, weather_time)
                vl += loss_fn(p, y).item() * x.size(0)
        vl_losses.append(vl / len(vl_ld.dataset))

        print(f"Epoch {ep}/{EPOCHS} | Train Loss: {tr_losses[-1]:.4f} | Val Loss: {vl_losses[-1]:.4f}")

    # 8) 评估与可视化
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 损失曲线
    plt.figure()
    plt.plot(tr_losses, label='Train')
    plt.plot(vl_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png', dpi=150)
    plt.close()

    # 验证集指标
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, weather_time, y in vl_ld:
            x, weather_time, y = x.to(DEVICE), weather_time.to(DEVICE), y.to(DEVICE)
            p = model(x, A, weather_time)
            all_preds.append(p.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(all_preds, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)
    y_true = val_targets.flatten()
    y_pred = val_preds.flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📊 最终验证指标:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # 专家激活分析
    def analyze_experts():
        """分析不同条件下专家的激活情况"""
        test_conditions = [
            {'name': 'Morning_Rush', 'tmp_C': 15, 'wind_dir': 90, 'wind_spd': 3, 'rh': 70, 'hour': 8},
            {'name': 'Hot_Noon', 'tmp_C': 35, 'wind_dir': 180, 'wind_spd': 2, 'rh': 40, 'hour': 12},
            {'name': 'Rainy_Night', 'tmp_C': 10, 'wind_dir': 270, 'wind_spd': 5, 'rh': 95, 'hour': 2},
            {'name': 'Windy_Evening', 'tmp_C': 20, 'wind_dir': 45, 'wind_spd': 10, 'rh': 60, 'hour': 18}
        ]

        print("\n🔍 专家激活分析:")
        model.eval()
        with torch.no_grad():
            for cond in test_conditions:
                features = torch.tensor([
                    cond['tmp_C'] / 40.0,
                    cond['wind_dir'] / 360.0,
                    cond['wind_spd'] / 20.0,
                    cond['rh'] / 100.0,
                    cond['hour'] / 23.0
                ], device=DEVICE).unsqueeze(0)

                gates = torch.softmax(model.gate_net(features), -1)
                print(f"{cond['name']}: {gates.cpu().numpy()[0].round(4)}")

    analyze_experts()

    print(f"\n✅ 训练完成! 结果保存在 {OUTPUT_DIR} 目录")


if __name__ == "__main__":
    main()
