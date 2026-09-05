import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "tgcn"

random.seed(53)
np.random.seed(53)
torch.manual_seed(53)
torch.cuda.manual_seed_all(53)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SEL_STATIONS = [
    "1335A","1336A","1337A","1338A","1339A",
    "1340A","1341A","1342A","1343A","1344A"
]
POLL_TYPES   = ["PM2.5","PM10","SO2","NO2","O3","CO"]
GROUP1 = [0, 1, 3]   # PM2.5, PM10, NO2
GROUP2 = [2, 4, 5]   # SO2, O3, CO
START_DATE   = "2024-11-01"
END_DATE     = "2025-03-16"
INPUT_STEPS  = 12
PRED_HORIZON = 1
EPOCHS       = 12
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

def haversine(lat1,lon1,lat2,lon2):
    R=6371.0
    φ1,φ2=map(math.radians,(lat1,lat2))
    dφ=math.radians(lat2-lat1); dλ=math.radians(lon2-lon1)
    a=math.sin(dφ/2)**2+math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_pollution(path):
    df=pd.read_csv(path,dtype=str)
    df['date_num']=pd.to_numeric(df['date'],errors='coerce')
    df['hour_num']=pd.to_numeric(df['hour'],errors='coerce')
    df.dropna(subset=['date_num','hour_num'],inplace=True)
    df['datetime'] = (
        pd.to_datetime(df['date_num'].astype(int).astype(str),
                       format="%Y%m%d",errors='coerce')
        + pd.to_timedelta(df['hour_num'].astype(int),unit='h')
    )
    df.dropna(subset=['datetime'],inplace=True)
    val_cols=[c for c in df.columns if c not in
              ['date','hour','date_num','hour_num','datetime','type']]
    long=df.melt(id_vars=['datetime','type'],value_vars=val_cols,
                 var_name='site',value_name='value')
    long['value']=pd.to_numeric(long['value'],errors='coerce')
    return long.dropna(subset=['value'])

def build_pollution_tensor(df):
    df=df[df['site'].isin(SEL_STATIONS)&df['type'].isin(POLL_TYPES)]
    tidy=df.pivot_table(index=['datetime','site'],
                        columns='type',values='value').reset_index()
    mask=(tidy['datetime']>=pd.to_datetime(START_DATE)) & \
         (tidy['datetime']<=pd.to_datetime(END_DATE))
    tidy=tidy.loc[mask].sort_values(['datetime','site'])
    tidy[POLL_TYPES]=tidy.groupby('site')[POLL_TYPES] \
                         .transform(lambda g: g.ffill().bfill())
    times=sorted(tidy['datetime'].unique())
    tmap={t:i for i,t in enumerate(times)}
    smap={s:i for i,s in enumerate(SEL_STATIONS)}
    T,N,P=len(times),len(SEL_STATIONS),len(POLL_TYPES)
    Xp=np.zeros((T,N,P),dtype=np.float32)
    for _,r in tidy.iterrows():
        ti,si=tmap[r['datetime']],smap[r['site']]
        for pi,pol in enumerate(POLL_TYPES):
            Xp[ti,si,pi]=r.get(pol,np.nan)
    return np.nan_to_num(Xp,nan=0.0), times

def load_weather(path):
    df=pd.read_csv(path,dtype=str)
    df['datetime']=pd.to_datetime(df['DATE'],errors='coerce')
    df.dropna(subset=['datetime','STATION'],inplace=True)
    df['tmp_C']=pd.to_numeric(df['TMP'].str.split(',').str[0], errors='coerce')/10.0
    def parse_wnd(s):
        p=str(s).split(',')
        d=float(p[0]) if p[0].replace('.','',1).isdigit() else np.nan
        sp=float(p[3])/10.0 if len(p)>3 and p[3].isdigit() else np.nan
        return d,sp
    df['wind_dir'],df['wind_spd']=zip(*df['WND'].apply(
        lambda x: parse_wnd(x) if isinstance(x,str) else (np.nan,np.nan)
    ))
    if 'DEWP' in df.columns:
        df['dew_C']=pd.to_numeric(df['DEWP'].str.split(',').str[0], errors='coerce')/10.0
        def calc_rh(t,td):
            if np.isnan(t) or np.isnan(td): return np.nan
            a,b=17.27,237.7
            return 100*math.exp(a*td/(b+td)-a*t/(b+t))
        df['rh']=df.apply(lambda r:calc_rh(r['tmp_C'],r['dew_C']),axis=1)
    else:
        df['rh']=np.nan
    df=df.sort_values(['STATION','datetime'])
    df[['tmp_C','wind_dir','wind_spd','rh']]= \
        df.groupby('STATION')[['tmp_C','wind_dir','wind_spd','rh']]\
          .transform(lambda s: s.interpolate().ffill().bfill())
    return df[['datetime','STATION','tmp_C','wind_dir','wind_spd','rh']]

def build_weather_tensor(wx, times):
    stations_wx=list(WEATHER_STATIONS.keys())
    M,N=len(stations_wx),len(SEL_STATIONS)
    D=np.zeros((N,M),dtype=np.float32)
    mon_xy=[STATION_COORDS[s] for s in SEL_STATIONS]
    wx_xy =[WEATHER_STATIONS[s] for s in stations_wx]
    for i,(lon,lat) in enumerate(mon_xy):
        ds=np.array([haversine(lat,lon,la,lo) for la,lo in wx_xy],dtype=np.float32)
        inv=1/(ds+1e-6); D[i]=inv/inv.sum()
    W_list=[]
    for t in times:
        df_t=wx[wx['datetime']==t]
        Wm=np.zeros((M,4),dtype=np.float32)
        for m,st in enumerate(stations_wx):
            r=df_t[df_t['STATION']==st]
            if not r.empty:
                row=r.iloc[0]
                Wm[m]=[row['tmp_C'],row['wind_dir'],row['wind_spd'],row['rh']]
        W_list.append(Wm)
    W_np=np.stack(W_list,axis=0)       # (T,M,4)
    Xw=np.stack([D.dot(W_np[t]) for t in range(len(times))],axis=0)
    return np.nan_to_num(Xw,nan=0.0)    # (T,N,4)

def build_static_geo_adj():
    N=len(SEL_STATIONS)
    A=np.zeros((N,N),dtype=np.float32)
    coords=[STATION_COORDS[s] for s in SEL_STATIONS]
    for i,(lon1,lat1) in enumerate(coords):
        for j,(lon2,lat2) in enumerate(coords):
            d=haversine(lat1,lon1,lat2,lon2)
            if d<=DIST_THRESHOLD_KM:
                A[i,j]=math.exp(-d*d/(2*GAUSS_SIGMA_KM**2))
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

class WindowDatasetWithGroups(torch.utils.data.Dataset):
    """每个样本预测多污染物并归一化"""
    def __init__(self, X, times, mu, sd):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.L = X.shape[0] - INPUT_STEPS - PRED_HORIZON + 1
        self.times = times
        self.mu = mu
        self.sd = sd
    def __len__(self): return max(self.L, 0)
    def __getitem__(self, i):
        x = self.X[i:i+INPUT_STEPS]
        y = self.X[i+INPUT_STEPS+PRED_HORIZON-1, :, :len(POLL_TYPES)]
        # 归一化y
        y_norm = (y - self.mu[0, 0, :]) / self.sd[0, 0, :]
        return x, y_norm

class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, X, A):
        return torch.relu(self.lin(A @ X))

class TGCN(nn.Module):
    def __init__(self, in_f, g_h=32, gru_h=32, out_dim=6):
        super().__init__()
        self.gcn = GraphConv(in_f, g_h)
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        # 分组专家输出
        self.group1_fc = nn.Linear(gru_h, len(GROUP1))
        self.group2_fc = nn.ModuleList([nn.Linear(gru_h, 1) for _ in GROUP2])
        self.out_dim = out_dim
    def forward(self, x, A_seq):
        B, T, N, F = x.shape
        gcn_out = []
        for t in range(T):
            A_t = A_seq[t]
            h_t = [self.gcn(x[b, t], A_t) for b in range(B)]
            gcn_out.append(torch.stack(h_t))
        gcn_out = torch.stack(gcn_out, 1)           # (B,T,N,g_h)
        bnf = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, h_n = self.gru(bnf)
        h_final = h_n.squeeze(0).reshape(B, N, -1)  # (B,N,gru_h)
        # group1联合预测
        group1_pred = self.group1_fc(h_final)       # (B,N,3)
        # group2分别预测
        group2_pred = [fc(h_final).squeeze(-1) for fc in self.group2_fc]  # [(B,N),...]
        group2_pred = torch.stack(group2_pred, -1)  # (B,N,3)
        # 按原顺序组装输出
        y_pred = torch.zeros(B, N, self.out_dim, device=x.device)
        for idx, g in enumerate(GROUP1):
            y_pred[:, :, g] = group1_pred[:, :, idx]
        for idx, g in enumerate(GROUP2):
            y_pred[:, :, g] = group2_pred[:, :, idx]
        return y_pred

def main():
    pollution_path = DATA_DIR / "pollution.csv"
    weather_path = DATA_DIR / "weather.csv"
    if not (pollution_path.exists() and weather_path.exists()):
        print(f"❌ 找不到输入数据，请检查 {DATA_DIR}")
        sys.exit(1)

    poll_df = load_pollution(pollution_path)
    Xp, times = build_pollution_tensor(poll_df)
    wx_df = load_weather(weather_path)
    Xw = build_weather_tensor(wx_df, times)

    X = np.concatenate([Xp, Xw], axis=2)
    # ---- 输入归一化（按所有特征） ----
    mu, sd = X.mean((0,1), keepdims=True), X.std((0,1), keepdims=True) + 1e-6
    X = (X - mu) / sd

    # ---- 标签归一化（每种污染物独立）----
    y_mu = Xp.mean((0,1), keepdims=True)     # (1,1,6)
    y_sd = Xp.std((0,1), keepdims=True) + 1e-6

    A_static = build_static_geo_adj()
    A_seq = build_dynamic_adj(A_static, Xw)

    ds = WindowDatasetWithGroups(X, times, y_mu, y_sd)
    ntr = int(0.8 * len(ds))
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [ntr, len(ds)-ntr])
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, batch_size=BATCH_SIZE)

    model = TGCN(in_f=X.shape[2], out_dim=len(POLL_TYPES)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    tr_losses, vl_losses = [], []

    for ep in range(1, EPOCHS+1):
        model.train(); tl=0
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            p = model(x, A_seq)
            l = loss_fn(p, y)
            l.backward(); opt.step()
            tl += l.item() * x.size(0)
        tr_losses.append(tl / len(tr_ld.dataset))

        model.eval(); vl=0
        with torch.no_grad():
            for x, y in vl_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vl += loss_fn(model(x, A_seq), y).item() * x.size(0)
        vl_losses.append(vl / len(vl_ld.dataset))

        print(f"Epoch {ep}/{EPOCHS}  train MSE {tr_losses[-1]:.4f}  val MSE {vl_losses[-1]:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(tr_losses, label='train'); plt.plot(vl_losses, label='val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png', dpi=150)

    all_p, all_t = [], []
    model.eval()
    with torch.no_grad():
        for x, y in vl_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x, A_seq)
            all_p.append(p.cpu().numpy())
            all_t.append(y.cpu().numpy())
    preds = np.concatenate(all_p, axis=0)
    targets = np.concatenate(all_t, axis=0)
    # ---- 反归一化（按物质）----
    preds_real   = preds * y_sd + y_mu    # (n, N, K)
    targets_real = targets * y_sd + y_mu

    y_true = targets_real.reshape(-1, len(POLL_TYPES))
    y_pred = preds_real.reshape(-1, len(POLL_TYPES))

    for k, name in enumerate(POLL_TYPES):
        mse  = mean_squared_error(y_true[:, k], y_pred[:, k])
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(y_true[:, k], y_pred[:, k])
        r2   = r2_score(y_true[:, k], y_pred[:, k])
        print(f"{name:6} - MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    print(f"✅ Done. 结果保存在 {OUTPUT_DIR}")

if __name__=="__main__":
    main()
