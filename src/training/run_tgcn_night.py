# run_tgcn_with_humidity.py
"""
T-GCN 验证脚本：在原有污染 + 气象（温度、风向/速）之外，仅引入相对湿度(rh)特征。

Usage:
    python run_tgcn_with_humidity.py
"""

import os, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "tgcn"

# ─── Config ────────────────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

SEL_STATIONS = [
    "1335A","1336A","1337A","1338A","1339A",
    "1340A","1341A","1342A","1343A","1344A"
]

POLL_TYPES   = ["PM2.5","PM10","SO2","NO2","O3","CO"]
START_DATE   = "2024-12-01"
END_DATE     = "2025-02-28"
INPUT_STEPS  = 12
PRED_HORIZON = 1
EPOCHS       = 10
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4

DIST_THRESHOLD_KM = 50
GAUSS_SIGMA_KM    = 20
# ────────────────────────────────────────────────────────────────────────────

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

# ─── 1. Pollution ──────────────────────────────────────────────────────────
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
    long=df.melt(id_vars=['datetime','type'],
                 value_vars=val_cols,
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

# ─── 2. Weather (temp, wind, humidity) ────────────────────────────────────
def load_weather(path):
    df=pd.read_csv(path,dtype=str)
    df['datetime']=pd.to_datetime(df['DATE'],errors='coerce')
    df.dropna(subset=['datetime','STATION'],inplace=True)

    # temperature
    df['tmp_C']=pd.to_numeric(
        df['TMP'].str.split(',').str[0],errors='coerce')/10.0

    # wind direction & speed
    def parse_wnd(s):
        p=str(s).split(',')
        d = float(p[0]) if p[0].replace('.','',1).isdigit() else np.nan
        sp= float(p[3])/10.0 if len(p)>3 and p[3].isdigit() else np.nan
        return d,sp
    df['wind_dir'],df['wind_spd']=zip(*df['WND'].apply(
        lambda x: parse_wnd(x) if isinstance(x,str) else (np.nan,np.nan)
    ))

    # relative humidity
    if 'DEWP' in df.columns:
        df['dew_C']=pd.to_numeric(
            df['DEWP'].str.split(',').str[0],errors='coerce')/10.0
        def calc_rh(t,td):
            if np.isnan(t) or np.isnan(td): return np.nan
            a,b=17.27,237.7
            return 100*math.exp(a*td/(b+td)-a*t/(b+t))
        df['rh']=df.apply(lambda r: calc_rh(r['tmp_C'],r['dew_C']),axis=1)
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

    # distance weights
    D=np.zeros((N,M),dtype=np.float32)
    mon_xy=[STATION_COORDS[s] for s in SEL_STATIONS]
    wx_xy =[WEATHER_STATIONS[s] for s in stations_wx]
    for i,(lon,lat) in enumerate(mon_xy):
        ds=np.array([haversine(lat,lon,lat2,lon2) for lat2,lon2 in wx_xy],dtype=np.float32)
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

    W_np=np.stack(W_list,axis=0)        # (T,M,4)
    Xw=np.stack([D.dot(W_np[t]) for t in range(len(times))],axis=0)
    return np.nan_to_num(Xw,nan=0.0)     # (T,N,4)

# ─── Dataset & Model ──────────────────────────────────────────────────────
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self,X):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.L=X.shape[0]-INPUT_STEPS-PRED_HORIZON+1
    def __len__(self): return max(self.L,0)
    def __getitem__(self,i):
        x=self.X[i:i+INPUT_STEPS]
        y=self.X[i+INPUT_STEPS+PRED_HORIZON-1,:,0]
        return x,y

class GraphAttentionConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.lin=nn.Linear(in_ch,out_ch,bias=False)
    def forward(self,X,A):
        H=self.lin(X)
        S=(H@H.T)/math.sqrt(H.size(1))
        S=S.masked_fill(A<=0,float('-inf'))
        α=torch.softmax(S,dim=1)
        return α@H

class TGCN(nn.Module):
    def __init__(self,in_f,g_h=32,gru_h=32):
        super().__init__()
        self.gcn=GraphAttentionConv(in_f,g_h)
        self.gru=nn.GRU(g_h,gru_h,batch_first=True)
        self.fc =nn.Linear(gru_h,1)
    def forward(self,x,A):
        B,T,N,F=x.shape
        gcn_out=[]
        for t in range(T):
            h_t=[self.gcn(x[b,t],A) for b in range(B)]
            gcn_out.append(torch.stack(h_t))
        gcn_out=torch.stack(gcn_out,1)
        bnf=gcn_out.permute(0,2,1,3).reshape(B*N,T,-1)
        _,h_n=self.gru(bnf)
        return self.fc(h_n.squeeze(0)).reshape(B,N)

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    pollution_path = DATA_DIR / "pollution.csv"
    weather_path = DATA_DIR / "weather.csv"
    if not (pollution_path.exists() and weather_path.exists()):
        print(f"❌ 找不到输入数据，请检查 {DATA_DIR}")
        sys.exit(1)

    # 1) Pollution tensor
    poll_df=load_pollution(pollution_path)
    Xp,times=build_pollution_tensor(poll_df)
    print("Xp:",Xp.shape)  # (T,10,6)

    # 2) Weather tensor
    wx_df=load_weather(weather_path)
    Xw=build_weather_tensor(wx_df,times)
    print("Xw:",Xw.shape)  # (T,10,4)

    # 3) Concat + norm
    X=np.concatenate([Xp,Xw],axis=2)  # (T,10,10)
    mu,sd = X.mean((0,1),keepdims=True), X.std((0,1),keepdims=True)+1e-6
    X=(X-mu)/sd
    print("X:",X.shape)

    # 4) Geo-adj
    def build_geo_adj(sts):
        N=len(sts); A=np.zeros((N,N),dtype=np.float32)
        xy=[STATION_COORDS[s] for s in sts]
        for i in range(N):
            for j in range(N):
                d=haversine(xy[i][1],xy[i][0],xy[j][1],xy[j][0])
                if d<=DIST_THRESHOLD_KM:
                    A[i,j]=math.exp(-d*d/(2*GAUSS_SIGMA_KM**2))
        np.fill_diagonal(A,1.0)
        D_inv=np.diag(1/np.sqrt(A.sum(1)+1e-6))
        return D_inv@A@D_inv
    A=torch.tensor(build_geo_adj(SEL_STATIONS),device=DEVICE)

    # 5) DataLoader
    ds=WindowDataset(X)
    ntr=int(0.8*len(ds))
    tr_ds,vl_ds=torch.utils.data.random_split(ds,[ntr,len(ds)-ntr])
    tr_ld=torch.utils.data.DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True)
    vl_ld=torch.utils.data.DataLoader(vl_ds,batch_size=BATCH_SIZE)

    # 6) Model & optim
    model=TGCN(in_f=X.shape[2]).to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    loss_fn=nn.MSELoss()
    tr_losses,vl_losses=[],[]

    # 7) Train
    for ep in range(1,EPOCHS+1):
        model.train(); tl=0
        for x,y in tr_ld:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            p=model(x,A)
            l=loss_fn(p,y)
            l.backward(); opt.step()
            tl+=l.item()*x.size(0)
        tr_losses.append(tl/len(tr_ld.dataset))

        model.eval(); vl=0
        with torch.no_grad():
            for x,y in vl_ld:
                x,y=x.to(DEVICE),y.to(DEVICE)
                vl+=loss_fn(model(x,A),y).item()*x.size(0)
        vl_losses.append(vl/len(vl_ld.dataset))

        print(f"Epoch {ep}/{EPOCHS}  train MSE {tr_losses[-1]:.4f}  val MSE {vl_losses[-1]:.4f}")

    # 8) Plot losses
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(tr_losses,label='train'); plt.plot(vl_losses,label='val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png',dpi=150)

    # 9) Full validation metrics
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x,y in vl_ld:
            x,y = x.to(DEVICE), y.to(DEVICE)
            p = model(x,A)
            all_preds.append(p.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    val_preds   = np.concatenate(all_preds, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

    y_true = val_targets.flatten()
    y_pred = val_preds.flatten()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("\n📊 Validation metrics:")
    print(f"  Loss (MSE): {mse:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  R²        : {r2:.4f}\n")

    # 10) Last forecast bar
    last=torch.tensor(X[-INPUT_STEPS:],dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred=model(last,A).cpu().numpy().flatten()
    gt=X[-1,:,0]
    plt.figure()
    idx=np.arange(len(gt))
    plt.bar(idx-0.2,gt,0.4,label='GT')
    plt.bar(idx+0.2,pred,0.4,label='Pred')
    plt.xticks(idx,SEL_STATIONS)
    plt.ylabel("PM2.5 (std)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'forecast_bar.png',dpi=150)

    print(f"✅ Done. Results in {OUTPUT_DIR}")

if __name__=="__main__":
    main()
