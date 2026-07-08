#!/usr/bin/env python3
"""Quick comparison: baseline vs 3,4,7,8,9. 20% of data, 1 seed each. ~2 min."""
import os, sys; sys.path.insert(0, "/tmp/Reinforcement_Trading_Part_2")
import pandas as pd, numpy as np, joblib, warnings, random
warnings.filterwarnings('ignore')
from pathlib import Path; from tqdm import tqdm
from features import add_stationary_features, atr
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Loading data...")
import pickle as _pk
all_raw = _pk.load(open(Path("cache_data.joblib"),"rb"))
all_raw = {t:d for t,d in all_raw.items() if len(d)>=200}
print(f"Tickers: {len(all_raw)}")

all_t = sorted(all_raw.keys()); random.shuffle(all_t)
val_t = set(all_t[:100])  # 100 val
train_t = set(all_t[100:600])  # 500 train
print(f"Train: {len(train_t)} tickers, Val: {len(val_t)} tickers")

def build(t):
    d=all_raw[t].copy().rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    d.index=pd.to_datetime(d.index).tz_localize("America/New_York");d.index.name=None
    ft,fcols=add_stationary_features(d)
    p=d["Close"];v=d["Volume"];h=d["High"];l=d["Low"];o=d["Open"];aa=atr(d,14).replace(0,np.nan)
    up=(p.diff()>0).astype(float);ft["extra_str_up"]=up*(up.groupby((up!=up.shift()).cumsum()).cumcount()+1)
    dn=(p.diff()<0).astype(float);ft["extra_str_dn"]=dn*(dn.groupby((dn!=dn.shift()).cumsum()).cumcount()+1)
    h20=h.rolling(20,5).max();l20=l.rolling(20,5).min();ft["extra_pos_rng"]=((p-l20)/(h20-l20).replace(0,np.nan)).shift(1)
    v20mn=v.rolling(20,5).min();v20mx=v.rolling(20,5).max();ft["extra_v_pos"]=((v-v20mn)/(v20mx-v20mn).replace(0,np.nan)).shift(1)
    ft["extra_vr"]=p.diff().rolling(5,3).std()/p.diff().rolling(20,10).std().replace(0,np.nan).shift(1)
    ft["extra_pv_corr"]=p.rolling(10,5).corr(v).shift(1)
    ft["extra_dh"]=((p/h.rolling(50,20).max()-1)*100).shift(1);ft["extra_dl"]=((l.rolling(50,20).min()/p-1)*100).shift(1)
    ft["extra_gap"]=(o/p.shift(1)-1).shift(1);ft["extra_r_atr"]=((h-l)/aa.replace(0,np.nan)).shift(1)
    ft["extra_body"]=((p-o).abs()/(h-l).replace(0,np.nan)).shift(1)
    ft["spy_ret"]=0;ft["vix"]=0;ft["tnx"]=0;ft["expert_up"]=0.5;ft["sector_id"]=0
    ft["gap_x_vr"]=ft["extra_gap"]*ft["extra_vr"];ft["ret1_x_spy"]=ft["ret1_atr"]*ft["spy_ret"] if "ret1_atr" in ft.columns else 0
    ft["vr_x_pv"]=ft["extra_vr"]*ft["extra_pv_corr"];ft["gap_x_body"]=ft["extra_gap"]*ft["extra_body"]
    ft["ret1_x_macd"]=ft["ret1_atr"]*ft["macd_hist_atr"] if "macd_hist_atr" in ft.columns else 0
    ALLC=["extra_str_up","extra_str_dn","extra_pos_rng","extra_v_pos","extra_vr","extra_pv_corr",
          "extra_dh","extra_dl","extra_gap","extra_r_atr","extra_body","spy_ret","expert_up",
          "gap_x_vr","ret1_x_spy","vr_x_pv","gap_x_body","ret1_x_macd"]
    ft=ft.iloc[60:].dropna(subset=fcols+ALLC)
    if len(ft)<5: return None
    ret=ft["Close"].pct_change().shift(-1).values*100
    X=np.nan_to_num(ft[fcols+ALLC].values,nan=0,posinf=0,neginf=0)
    return X, ret

print("Building features...")
X_tr=[];y_tr=[];ret_tr=[]
for t in tqdm(train_t,desc="Train"):
    r=build(t)
    if r is not None: X_tr.append(r[0]);y_tr.append((r[1]>0).astype(float));ret_tr.append(r[1])
X_tr=np.vstack(X_tr);y_tr=np.concatenate(y_tr);ret_tr=np.nan_to_num(np.concatenate(ret_tr),nan=0)
X_te=[];y_te=[];ret_te=[]
for t in tqdm(val_t,desc="Val"):
    r=build(t)
    if r is not None: X_te.append(r[0]);y_te.append((r[1]>0).astype(float));ret_te.append(r[1])
X_te=np.vstack(X_te);y_te=np.concatenate(y_te)
sc=StandardScaler();X_tr_s=sc.fit_transform(X_tr);X_te_s=sc.transform(X_te)
print(f"Train: {X_tr_s.shape}, Val: {X_te_s.shape}")

def try_method(name,do_pca=0,do_noise=0,do_meta=0,sample_w=None):
    Xt=X_tr_s.copy();Xv=X_te_s.copy()
    if do_pca:
        p=PCA(5,random_state=42);Xt=np.hstack([Xt,p.fit_transform(Xt)]);Xv=np.hstack([Xv,p.transform(Xv)])
    m0=LGBMClassifier(n_estimators=100,max_depth=4,num_leaves=15,random_state=42,n_jobs=-1,verbose=-1,force_col_wise=True)
    m0.fit(Xt,y_tr)
    top=np.argsort(-m0.feature_importances_)[:20];Xt=Xt[:,top];Xv=Xv[:,top]
    mt=np.zeros((len(y_tr),1));mv=np.zeros((len(y_te),1))
    ms=[]
    for sd in tqdm([42],desc=name,leave=False):
        Xmt=Xt;Xmv=Xv
        if do_meta:Xmt=np.hstack([Xt,mt]);Xmv=np.hstack([Xv,mv])
        if do_noise:np.random.seed(sd);Xmt+=np.random.randn(*Xmt.shape)*do_noise
        m=LGBMClassifier(class_weight="balanced",random_state=sd,n_estimators=300,
            max_depth=7,num_leaves=127,min_child_samples=50,learning_rate=0.05,
            subsample=0.85,colsample_bytree=0.85,n_jobs=-1,verbose=-1,force_col_wise=True)
        m.fit(Xmt,y_tr,sample_weight=sample_w)
        ms.append(m)
        if do_meta:mt=m.predict_proba(Xmt)[:,1:2];mv=m.predict_proba(Xmv)[:,1:2]
    if do_meta:
        p=np.mean([m.predict_proba(Xmv)[:,1] for m in ms],axis=0)
    else:
        p=np.mean([m.predict_proba(Xv)[:,1] for m in ms],axis=0)
    return ((p>.5)==y_te).mean()

results={}
results["base"]=try_method("baseline")
results["3(curr)"]=try_method("3",sample_w=np.abs(ret_tr)/max(np.abs(ret_tr).mean(),0.001)*0.5+0.5)
results["4(noise)"]=try_method("4",do_noise=0.03)
results["7(pca)"]=try_method("7",do_pca=1)
results["8(temp)"]=try_method("8",sample_w=np.exp(np.linspace(0,1.5,len(y_tr))))
results["9(meta)"]=try_method("9",do_meta=1)

print(f"\n{'Method':>12s}  {'Val acc':>8s}  {'vs base':>8s}")
print("-"*32)
base=results["base"]
for n,a in results.items():
    d=a-base;m="✅" if d>0.005 else"❌" if d<-0.005 else"·" if n!="base" else""
    print(f"{n:>12s}  {a:.4f}  {d:>+7.4f}  {m}")
