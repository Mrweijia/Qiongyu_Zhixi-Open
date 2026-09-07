import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
# 设置字体为支持中文的字体
rcParams['font.family'] = 'Microsoft YaHei'  # 或者使用 'SimHei' 或其他支持中文的字体

# 确保正确显示中文
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "moe_tgcn"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
random.seed(53)
np.random.seed(53)
torch.manual_seed(53)
torch.cuda.manual_seed_all(53)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Config ────────────────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SEL_STATIONS = ["1335A","1336A","1337A","1338A","1339A",
                "1340A","1341A","1342A","1343A","1344A"]
POLL_TYPES   = ["PM2.5","PM10","SO2","NO2","O3","CO"]

# 仅预测 PM2.5、PM10、NO2、O3
PRED_IDX   = [0, 1, 3, 4]
PRED_NAMES = [POLL_TYPES[i] for i in PRED_IDX]

START_DATE   = "2023-01-01"
END_DATE     = "2025-05-16"
INPUT_STEPS  = 12
PRED_HORIZON = 1
EPOCHS       = 15
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DIST_THRESHOLD_KM = 50
GAUSS_SIGMA_KM    = 20

STATION_COORDS = {
    "1335A":(113.0833,28.2325),"1336A":(112.8872,28.2189),
    "1337A":(113.0792,28.2053),"1338A":(112.9394,28.1900),
    "1339A":(113.0178,28.1322),"1340A":(112.9792,28.2597),
    "1341A":(113.0014,28.1944),"1342A":(112.9840,28.1178),
    "1343A":(112.8908,28.1308),"1344A":(112.9581,28.3611)
}
WEATHER_STATIONS = {
    "57687099999":  (28.116666,112.783333),
    "592871999999": (28.189158,113.219633)
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def load_pollution(path):
    df = pd.read_csv(path, dtype=str)
    df['date_num'] = pd.to_numeric(df['date'], errors='coerce')
    df['hour_num'] = pd.to_numeric(df['hour'], errors='coerce')
    df.dropna(subset=['date_num', 'hour_num'], inplace=True)
    df['datetime'] = (
        pd.to_datetime(df['date_num'].astype(int).astype(str),
                       format="%Y%m%d", errors='coerce')
        + pd.to_timedelta(df['hour_num'].astype(int), unit='h')
    )
    df.dropna(subset=['datetime'], inplace=True)
    val_cols = [c for c in df.columns if c not in
                ['date', 'hour', 'date_num', 'hour_num', 'datetime', 'type']]
    long = df.melt(id_vars=['datetime','type'], value_vars=val_cols,
                   var_name='site', value_name='value')
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    return long.dropna(subset=['value'])


def build_pollution_tensor(df):
    df = df[df['site'].isin(SEL_STATIONS) & df['type'].isin(POLL_TYPES)]
    tidy = df.pivot_table(index=['datetime','site'],
                          columns='type', values='value').reset_index()
    # Do not clip inference input to the historical training date window. The
    # online service must also accept current and future observation dates.
    tidy = tidy.sort_values(['datetime','site'])
    tidy[POLL_TYPES] = tidy.groupby('site')[POLL_TYPES] \
                           .transform(lambda g: g.ffill().bfill())
    times = sorted(tidy['datetime'].unique())
    tmap  = {t:i for i,t in enumerate(times)}
    smap  = {s:i for i,s in enumerate(SEL_STATIONS)}
    T,N,P = len(times), len(SEL_STATIONS), len(POLL_TYPES)
    Xp = np.zeros((T,N,P), dtype=np.float32)
    for _,r in tidy.iterrows():
        ti,si = tmap[r['datetime']], smap[r['site']]
        for pi,pol in enumerate(POLL_TYPES):
            Xp[ti,si,pi] = r.get(pol, np.nan)
    return np.nan_to_num(Xp, nan=0.0), times


def load_weather(path):
    df = pd.read_csv(path, dtype=str)
    df['datetime'] = pd.to_datetime(df['DATE'], errors='coerce')
    df.dropna(subset=['datetime','STATION'], inplace=True)
    df['tmp_C'] = pd.to_numeric(df['TMP'].str.split(',').str[0], errors='coerce')/10.0
    def parse_wnd(s):
        p = str(s).split(',')
        d = float(p[0]) if p[0].replace('.','',1).isdigit() else np.nan
        sp= float(p[3])/10.0 if len(p)>3 and p[3].isdigit() else np.nan
        return d, sp
    df['wind_dir'], df['wind_spd'] = zip(*df['WND'].apply(
        lambda x: parse_wnd(x) if isinstance(x,str) else (np.nan,np.nan)
    ))
    if 'DEWP' in df.columns:
        df['dew_C'] = pd.to_numeric(df['DEWP'].str.split(',').str[0], errors='coerce')/10.0
        def calc_rh(t,td):
            if np.isnan(t) or np.isnan(td): return np.nan
            a,b = 17.27, 237.7
            return 100 * math.exp(a*td/(b+td) - a*t/(b+t))
        df['rh'] = df.apply(lambda r: calc_rh(r['tmp_C'], r['dew_C']), axis=1)
    else:
        df['rh'] = np.nan
    df = df.sort_values(['STATION','datetime'])
    df[['tmp_C','wind_dir','wind_spd','rh']] = \
        df.groupby('STATION')[['tmp_C','wind_dir','wind_spd','rh']]\
          .transform(lambda s: s.interpolate().ffill().bfill())
    return df[['datetime','STATION','tmp_C','wind_dir','wind_spd','rh']]


def build_weather_tensor(wx, times):
    stations_wx = list(WEATHER_STATIONS.keys())
    M,N = len(stations_wx), len(SEL_STATIONS)
    D = np.zeros((N,M), dtype=np.float32)
    mon_xy = [STATION_COORDS[s] for s in SEL_STATIONS]
    wx_xy  = [WEATHER_STATIONS[s] for s in stations_wx]
    for i,(lon,lat) in enumerate(mon_xy):
        ds = np.array([haversine(lat,lon,la,lo) for la,lo in wx_xy], dtype=np.float32)
        inv = 1/(ds+1e-6)
        D[i] = inv / inv.sum()
    W_list=[]
    for t in times:
        df_t = wx[wx['datetime']==t]
        Wm   = np.zeros((M,4), dtype=np.float32)
        for m,st in enumerate(stations_wx):
            r = df_t[df_t['STATION']==st]
            if not r.empty:
                row = r.iloc[0]
                Wm[m] = [row['tmp_C'], row['wind_dir'], row['wind_spd'], row['rh']]
        W_list.append(Wm)
    W_np = np.stack(W_list, axis=0)
    Xw   = np.stack([D.dot(W_np[t]) for t in range(len(times))], axis=0)
    return np.nan_to_num(Xw, nan=0.0)


def build_static_geo_adj():
    N = len(SEL_STATIONS)
    A = np.zeros((N,N), dtype=np.float32)
    coords = [STATION_COORDS[s] for s in SEL_STATIONS]
    for i,(lon1,lat1) in enumerate(coords):
        for j,(lon2,lat2) in enumerate(coords):
            d = haversine(lat1,lon1,lat2,lon2)
            if d <= DIST_THRESHOLD_KM:
                A[i,j] = math.exp(-d*d/(2*GAUSS_SIGMA_KM**2))
    np.fill_diagonal(A,1.0)
    D_inv = np.diag(1/np.sqrt(A.sum(axis=1)+1e-6))
    return D_inv @ A @ D_inv


def build_dynamic_adj(static_A, Xw):
    T,N,_ = Xw.shape
    coords = [STATION_COORDS[s] for s in SEL_STATIONS]
    ang_ij = np.zeros((N,N), dtype=np.float32)
    for i,(lon1,lat1) in enumerate(coords):
        for j,(lon2,lat2) in enumerate(coords):
            ang_ij[i,j] = math.atan2(lat2-lat1, lon2-lon1)
    A_seq = np.zeros((T,N,N), dtype=np.float32)
    for t in range(T):
        flow_rad = np.deg2rad((Xw[t,:,1] + 180) % 360)
        align    = np.maximum(0, np.cos(flow_rad[None,:] - ang_ij))
        A_t = static_A * align
        np.fill_diagonal(A_t,1.0)
        row_sum = A_t.sum(axis=1, keepdims=True) + 1e-6
        A_seq[t] = A_t / row_sum
    return torch.tensor(A_seq, device=DEVICE)


# ===== Attention + Model + Dataset =====

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
        B,T,N,F = x.shape
        x_flat = x.permute(0,2,1,3).reshape(B*N, T, F)
        out,_  = self.attn(x_flat, x_flat, x_flat)
        out    = out.reshape(B, N, T, F).permute(0,2,1,3)
        return out


class CrossAttention(nn.Module):
    def __init__(self, poll_dim, wx_dim, attn_dim=8):
        super().__init__()
        self.poll_proj = nn.Linear(poll_dim, attn_dim)
        self.wx_proj   = nn.Linear(wx_dim, attn_dim)
        self.cross     = nn.MultiheadAttention(attn_dim, 2, batch_first=True)
        self.fc        = nn.Linear(poll_dim+attn_dim, poll_dim+attn_dim)
        self.attn_dim  = attn_dim
    def forward(self, poll, wx):
        B,T,N,P = poll.shape
        _,_,_,W = wx.shape
        p_flat = poll.permute(0,2,1,3).reshape(B*N, T, P)
        w_flat = wx.permute(0,2,1,3).reshape(B*N, T, W)
        p_proj = self.poll_proj(p_flat)
        w_proj = self.wx_proj(w_flat)
        attn_out,_ = self.cross(p_proj, w_proj, w_proj)
        attn_out   = attn_out.reshape(B, N, T, self.attn_dim).permute(0,2,1,3)
        concat     = torch.cat([poll, attn_out], dim=-1)
        return self.fc(concat)


def get_season_index(ts: pd.Timestamp):
    m = ts.month
    if m in [12,1,2]: return 0
    if m in [3,4,5]:  return 1
    if m in [6,7,8]:  return 2
    return 3


class WindowDatasetWithSeason(torch.utils.data.Dataset):
    def __init__(self, X, times):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.L = X.shape[0] - INPUT_STEPS - PRED_HORIZON + 1
        self.times = times
        self.season_ids = [
            get_season_index(times[i+INPUT_STEPS+PRED_HORIZON-1])
            for i in range(self.L)
        ]
    def __len__(self):
        return max(self.L, 0)
    def __getitem__(self, i):
        x = self.X[i:i+INPUT_STEPS]
        y = self.X[i+INPUT_STEPS+PRED_HORIZON-1, :, PRED_IDX]
        return x, y, self.season_ids[i]


class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, X, A):
        return torch.relu(self.lin(A @ X))


class TGCN_with_Attn(nn.Module):
    def __init__(self, in_f, g_h=32, gru_h=32, poll_dim=6, wx_dim=4, attn_dim=8, n_out=4):
        super().__init__()
        self.feature_attn  = FeatureAttention(in_f)
        self.temporal_attn = TemporalSelfAttention(in_f)
        self.cross_attn    = CrossAttention(poll_dim, wx_dim, attn_dim)
        self.gcn           = GraphConv(poll_dim+attn_dim, g_h)
        self.gru           = nn.GRU(g_h, gru_h, batch_first=True)
        self.fc            = nn.Linear(gru_h, n_out)
        self.poll_dim      = poll_dim
        self.wx_dim        = wx_dim

    def forward(self, x, A_seq):
        poll    = x[...,:self.poll_dim]
        wx      = x[...,-self.wx_dim:]
        x_fa    = self.feature_attn(x)
        x_ta    = self.temporal_attn(x_fa)
        # Keep the trained checkpoint architecture: the graph convolution takes
        # the pollution/weather cross-attention output directly.
        x_cross = self.cross_attn(poll, wx)
        B,T,N,F = x_cross.shape

        gcn_out = []
        for t in range(T):
            A_t = A_seq[t]
            h_t = [self.gcn(x_cross[b,t], A_t) for b in range(B)]
            gcn_out.append(torch.stack(h_t))
        gcn_out = torch.stack(gcn_out, dim=1)  # (B,T,N,g_h)

        bnf = gcn_out.permute(0,2,1,3).reshape(B*N, T, -1)
        _, h_n = self.gru(bnf)
        out    = self.fc(h_n.squeeze(0))
        return out.reshape(B, N, -1)


class HardMOESeason(nn.Module):
    def __init__(self, in_f, g_h=32, gru_h=32, poll_dim=6, wx_dim=4,
                 attn_dim=8, n_exp=4, n_out=4):
        super().__init__()
        self.experts = nn.ModuleList([
            TGCN_with_Attn(in_f, g_h, gru_h, poll_dim, wx_dim, attn_dim, n_out)
            for _ in range(n_exp)
        ])
    def forward(self, x, A_seq, season_ids):
        outs = []
        for i in range(x.shape[0]):
            s = int(season_ids[i])
            outs.append(self.experts[s](x[i:i+1], A_seq))
        return torch.cat(outs, dim=0)


def main():
    pollution_path = DATA_DIR / "pollution.csv"
    weather_path = DATA_DIR / "weather.csv"
    if not (pollution_path.exists() and weather_path.exists()):
        print(f"❌ 找不到输入数据，请检查 {DATA_DIR}")
        sys.exit(1)

    # 数据加载
    poll_df = load_pollution(pollution_path)
    Xp, times = build_pollution_tensor(poll_df)
    wx_df = load_weather(weather_path)
    Xw = build_weather_tensor(wx_df, times)

    # 合并 & 标准化
    X  = np.concatenate([Xp, Xw], axis=2)
    mu, sd = X.mean((0,1), keepdims=True), X.std((0,1), keepdims=True) + 1e-6
    X = (X - mu) / sd

    # 邻接
    A_static = build_static_geo_adj()
    A_seq    = build_dynamic_adj(A_static, Xw)

    # DataLoader
    ds   = WindowDatasetWithSeason(X, times)
    ntr  = int(0.8 * len(ds))
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [ntr, len(ds)-ntr])
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, batch_size=BATCH_SIZE)

    # 模型 & 优化
    model = HardMOESeason(in_f=X.shape[2],
                          poll_dim=Xp.shape[2],
                          wx_dim=Xw.shape[2],
                          attn_dim=8,
                          n_out=len(PRED_IDX)
                         ).to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn= nn.MSELoss()

    # 训练 & 验证
    tr_losses, vl_losses = [], []
    for ep in range(1, EPOCHS+1):
        model.train()
        tot_tr = 0.0
        for x,y,season in tr_ld:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            p = model(x, A_seq, season)
            l = loss_fn(p, y); l.backward(); opt.step()
            tot_tr += l.item()*x.size(0)
        tr_losses.append(tot_tr/len(tr_ld.dataset))

        model.eval()
        tot_vl = 0.0
        with torch.no_grad():
            for x,y,season in vl_ld:
                x,y = x.to(DEVICE), y.to(DEVICE)
                tot_vl += loss_fn(model(x, A_seq, season), y).item()*x.size(0)
        vl_losses.append(tot_vl/len(vl_ld.dataset))

        print(f"Epoch {ep}/{EPOCHS}  train MSE {tr_losses[-1]:.4f}  val MSE {vl_losses[-1]:.4f}")

    # 保存 loss 曲线
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(tr_losses, label='train')
    plt.plot(vl_losses, label='val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png', dpi=150)

    # 验证评估
    all_p, all_t = [], []
    model.eval()
    with torch.no_grad():
        for x,y,season in vl_ld:
            x,y = x.to(DEVICE), y.to(DEVICE)
            all_p.append(model(x, A_seq, season).cpu().numpy())
            all_t.append(y.cpu().numpy())

    preds   = np.concatenate(all_p, axis=0)
    targets = np.concatenate(all_t, axis=0)
    y_true  = targets.reshape(-1, len(PRED_IDX))
    y_pred  = preds.reshape(-1, len(PRED_IDX))

    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊 Final val → MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, 联合R²={r2:.4f}")
    for i,name in enumerate(PRED_NAMES):
        print(f"{name} R² = {r2_score(y_true[:,i], y_pred[:,i]):.4f}")

    # ====== 最终预测 & 反归一化科研柱状图 ======
    last_seq    = torch.tensor(X[-INPUT_STEPS:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    last_season = get_season_index(pd.Timestamp(times[-1]))
    with torch.no_grad():
        pr = model(last_seq, A_seq, torch.tensor([last_season])).cpu().numpy()[0]

    gt = X[-1][:, PRED_IDX]
    # 反归一化
    mu_poll = mu[0,0,PRED_IDX]
    sd_poll = sd[0,0,PRED_IDX]
    gt_denorm = gt * sd_poll + mu_poll
    pr_denorm = pr * sd_poll + mu_poll

    idx = np.arange(len(SEL_STATIONS))
    K   = len(PRED_IDX)
    width = 0.7 / K

    fig, ax = plt.subplots(figsize=(14,6))
    cmap = plt.colormaps['tab10']
    for i,pname in enumerate(PRED_NAMES):
        ax.bar(idx + i*width,         gt_denorm[:,i],  width*0.95,
               color=cmap(i), edgecolor='black', alpha=0.85, label=f'{pname} GT')
        ax.bar(idx + i*width + K*width, pr_denorm[:,i], width*0.95,
               color=cmap(i), edgecolor='black', alpha=0.45, label=f'{pname} Pred')

    ax.set_xticks(idx + K*width/1.5)
    ax.set_xticklabels(SEL_STATIONS, fontsize=13)
    ax.set_xlabel('站点编号', fontsize=15)
    ax.set_ylabel('浓度 (µg/m³)', fontsize=15)
    ax.set_title('各站点多污染物预测对比（反归一化）', fontsize=16, weight='bold')

    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1.0)

    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    dedup = OrderedDict()
    for h,l in zip(handles,labels):
        if l not in dedup:
            dedup[l] = h
    ax.legend(dedup.values(), dedup.keys(), ncol=2, fontsize=13, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'forecast_bar_denorm.png', dpi=300, bbox_inches='tight')
    plt.show()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "tgcn_four_out.pt")
    print(f"✅ Done. 结果已保存在 {OUTPUT_DIR}，模型已保存到 {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()
