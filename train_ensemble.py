#!/usr/bin/env python3
"""LGBM v4: market breadth, feature interactions, threshold tuning."""
import os, sys; sys.path.insert(0, "/tmp/Reinforcement_Trading_Part_2")
import pandas as pd, numpy as np, joblib, pickle, warnings, random, yfinance as yf
warnings.filterwarnings('ignore')
from pathlib import Path; from tqdm import tqdm
from features import add_stationary_features, atr
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

# ── Extra features (same as before) ──
def add_extra(df):
    p = df["Close"]; v = df["Volume"]; h = df["High"]; l = df["Low"]; o = df["Open"]
    a = atr(df,14).replace(0,np.nan)
    up = (p.diff()>0).astype(float)
    df["extra_str_up"] = up*(up.groupby((up!=up.shift()).cumsum()).cumcount()+1)
    dn = (p.diff()<0).astype(float)
    df["extra_str_dn"] = dn*(dn.groupby((dn!=dn.shift()).cumsum()).cumcount()+1)
    h20=h.rolling(20,5).max(); l20=l.rolling(20,5).min()
    df["extra_pos_rng"] = ((p-l20)/(h20-l20).replace(0,np.nan)).shift(1)
    v20mn=v.rolling(20,5).min(); v20mx=v.rolling(20,5).max()
    df["extra_v_pos"] = ((v-v20mn)/(v20mx-v20mn).replace(0,np.nan)).shift(1)
    v5=p.diff().rolling(5,3).std(); v20s=p.diff().rolling(20,10).std()
    df["extra_vr"] = (v5/v20s.replace(0,np.nan)).shift(1)
    df["extra_pv_corr"] = p.rolling(10,5).corr(v).shift(1)
    h50=h.rolling(50,20).max(); l50=l.rolling(50,20).min()
    df["extra_dh"] = ((p/h50-1)*100).shift(1)
    df["extra_dl"] = ((l50/p-1)*100).shift(1)
    df["extra_gap"] = (o/p.shift(1)-1).shift(1)
    df["extra_r_atr"] = ((h-l)/a.replace(0,np.nan)).shift(1)
    df["extra_body"] = ((p-o).abs()/(h-l).replace(0,np.nan)).shift(1)
    # Market return
    spy_aligned = [spy_map.get(d.date(),0.0) for d in df.index]
    df["spy_ret"] = np.roll(spy_aligned,1); df["spy_ret"].iloc[0]=0.0
    vix_aligned = [vix_map.get(d.date(),0.0) for d in df.index]
    df["vix"] = np.roll(vix_aligned,1); df["vix"].iloc[0]=0.0
    tnx_aligned = [tnx_map.get(d.date(),0.0) for d in df.index]
    df["tnx"] = np.roll(tnx_aligned,1); df["tnx"].iloc[0]=0.0
    # Expert prediction
    exp_aligned = [expert_map.get(d.date(),0.5) for d in df.index]
    df["expert_up"] = np.roll(exp_aligned,1); df["expert_up"].iloc[0]=0.5
    return df

ALL_COLS = ["extra_str_up","extra_str_dn","extra_pos_rng","extra_v_pos",
            "extra_vr","extra_pv_corr","extra_dh","extra_dl",
            "extra_gap","extra_r_atr","extra_body","spy_ret","expert_up"]

# ── Expert prediction feature (SPF survey, quarterly GDP nowcast) ──
def get_expert_map():
    """Download SPF quarterly GDP growth nowcast, return {date: 0/1} map for daily use."""
    try:
        from python_calamine import CalamineWorkbook
        url = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
               "survey-of-professional-forecasters/historical-data/meanGrowth.xlsx"
               "?sc_lang=en&hash=0F651C29D8FE2E04AB86BC7DD7EDAD2E")
        import requests, io
        r = requests.get(url, timeout=20)
        with io.BytesIO(r.content) as f:
            wb = CalamineWorkbook.from_filelike(f)
        data = wb.get_sheet_by_name('RGDP').to_python()
        df = pd.DataFrame(data)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df = df.replace('#N/A', np.nan)
        for c in df.columns:
            try: df[c] = pd.to_numeric(df[c], errors='coerce')
            except: pass
        mask = (df['YEAR'] >= 2019) & (df['YEAR'] <= 2027) & (df['QUARTER'].isin([1,2,3,4]))
        spf = df[mask][['YEAR','QUARTER','drgdp2']].copy()
        spf['expert_up'] = (spf['drgdp2'] > 0).astype(int)

        # Create daily map: forward-fill quarterly values
        daily = {}
        for _, row in spf.iterrows():
            yr, q, val = int(row['YEAR']), int(row['QUARTER']), int(row['expert_up'])
            q_start = pd.Timestamp(f'{yr}-{(q-1)*3+1:02d}-01')
            q_end = pd.Timestamp(f'{yr}-{q*3:02d}-01') + pd.offsets.MonthEnd(0)
            for d in pd.date_range(q_start, q_end, freq='D'):
                daily[d.date()] = val

        # Fill gaps (last known value)
        all_dates = sorted(daily.keys())
        if all_dates:
            last_val = daily[all_dates[0]]
            for d in pd.date_range(all_dates[0], pd.Timestamp('2026-12-31'), freq='D'):
                if d.date() in daily:
                    last_val = daily[d.date()]
                else:
                    daily[d.date()] = last_val
        return daily
    except Exception as e:
        print(f"Expert data download failed: {e}")
        return {}

print("Loading data...")
cache_path = Path("cache_data.joblib")
backup_path = Path("cache_data.backup")
try:
    all_raw = joblib.load(cache_path) if cache_path.exists() else {}
except:
    print("Cache corrupted! Loading from backup...")
    all_raw = joblib.load(backup_path) if backup_path.exists() else {}
    if all_raw: pickle.dump(all_raw, open(cache_path,'wb'), protocol=5)  # restore cache
all_raw = {t:d for t,d in all_raw.items() if len(d)>=200}
print(f"Tickers: {len(all_raw)} (min 200 days)")



# Load SPY, VIX, TNX
syms = {"SPY":"close","^VIX":"close","^TNX":"close"}
data_maps = {}
for name, col in syms.items():
    try:
        df = yf.download(name,period="2y",interval="1d",progress=False,auto_adjust=True)
        if df.empty: continue
        df.columns=[c[0].lower() for c in df.columns]
        df.index=pd.to_datetime(df.index).tz_localize(None)
        vals = df[col].pct_change().values if name=="SPY" else df[col].values
        data_maps[name] = {d.date():v for d,v in zip(df.index,vals) if not np.isnan(v)}
    except: pass
spy_map = data_maps.get("SPY",{})
vix_map = data_maps.get("^VIX",{})
tnx_map = data_maps.get("^TNX",{})

# Get sectors for tickers (cache)
sector_file = Path("sector_cache.joblib")
if sector_file.exists():
    sectors = joblib.load(sector_file)
else:
    sectors = {}
sector_ids = {s:i for i,s in enumerate(set(sectors.values()))}
expert_map = get_expert_map()
print(f"Data: SPY={len(spy_map)} VIX={len(vix_map)} TNX={len(tnx_map)} Sectors={len(sector_ids)} Expert={len(expert_map)}")

# Split
all_t = sorted(all_raw.keys()); random.seed(42); random.shuffle(all_t)
n_val = max(1,int(len(all_t)*0.08)); val_ts=set(all_t[:min(n_val,500)]); train_ts=set(all_t[len(val_ts):])
print(f"Train: {len(train_ts)}, Val: {len(val_ts)}")

def build_features(t):
    """Build features without target. Returns (X_df, cols, ret_series)."""
    d = all_raw[t].copy().rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    d.index=pd.to_datetime(d.index).tz_localize("America/New_York"); d.index.name=None
    feat, fcols = add_stationary_features(d)
    feat = add_extra(feat)
    feat["sector_id"] = sector_ids.get(sectors.get(t,"Unknown"),0)
    cols = fcols + [c for c in ALL_COLS if c in feat.columns] + ["vix","tnx","sector_id"]
    feat["gap_x_vr"] = feat["extra_gap"] * feat["extra_vr"]
    feat["ret1_x_spy"] = feat["ret1_atr"] * feat["spy_ret"] if "ret1_atr" in feat.columns else 0
    feat["vr_x_pv"] = feat["extra_vr"] * feat["extra_pv_corr"]
    feat["gap_x_body"] = feat["extra_gap"] * feat["extra_body"]
    feat["ret1_x_macd"] = feat["ret1_atr"] * feat["macd_hist_atr"] if "macd_hist_atr" in feat.columns else 0
    cols = cols + ["gap_x_vr","ret1_x_spy","vr_x_pv","gap_x_body","ret1_x_macd"]
    feat = feat.iloc[60:].dropna(subset=cols)
    if len(feat)<5: return None, None, None, None
    ret = feat["Close"].pct_change().shift(-1).values * 100
    X = feat[cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols, ret, len(feat)

# Build features once for all tickers
print("Building features (once)...")
FEAT_cache = {}
saved_cols = None
for t in tqdm(train_ts, desc="Train"):
    try:
        X_arr, cols, ret_arr, n = build_features(t)
        if X_arr is not None: FEAT_cache[t] = (X_arr, ret_arr); saved_cols = cols
    except: pass
cols = saved_cols
X_tr_raw = np.vstack([FEAT_cache[t][0] for t in FEAT_cache])
FEAT_cache_te = {}
for t in tqdm(val_ts, desc="Val"):
    try:
        X_arr, _, ret_arr, n = build_features(t)
        if X_arr is not None: FEAT_cache_te[t] = (X_arr, ret_arr)
    except: pass
X_te_raw = np.vstack([FEAT_cache_te[t][0] for t in FEAT_cache_te])

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr_raw); X_te_s = sc.transform(X_te_raw)

y_temp = np.concatenate([FEAT_cache[t][1] for t in FEAT_cache])
y_temp_bin = (y_temp > 0).astype(int)

# ── 7: PCA feature interactions ──
from sklearn.decomposition import PCA
pca = PCA(n_components=5, random_state=42)
X_tr_s = np.hstack([X_tr_s, pca.fit_transform(X_tr_s)])
X_te_s = np.hstack([X_te_s, pca.transform(X_te_s)])
print(f"PCA added → {X_tr_s.shape[1]} features")

# Quick importance → top 25
m0 = LGBMClassifier(n_estimators=100,max_depth=4,num_leaves=15,random_state=42,n_jobs=-1,verbose=-1,force_col_wise=True)
m0.fit(X_tr_s, y_temp_bin)
imp = m0.feature_importances_
n_keep = 25; top_idx = np.argsort(-imp)[:n_keep]
X_tr_s = X_tr_s[:, top_idx]; X_te_s = X_te_s[:, top_idx]
print(f"Selected {X_tr_s.shape[1]} features")

# Build binary targets
y_tr = (np.concatenate([FEAT_cache[t][1] for t in FEAT_cache]) > 0).astype(int)
y_te = (np.concatenate([FEAT_cache_te[t][1] for t in FEAT_cache_te]) > 0).astype(int)
ret_val = np.concatenate([FEAT_cache_te[t][1] for t in FEAT_cache_te])
print(f"Train: {X_tr_s.shape}, Val: {X_te_s.shape}, up: {y_tr.mean()*100:.1f}%")

# ── 3+8: Sample weights (recency × abs return) ──
n = len(y_tr)
w = np.exp(np.linspace(0, 1, n) * 2)
w *= (1 + np.nan_to_num(np.abs(np.concatenate([FEAT_cache[t][1] for t in FEAT_cache])), nan=0.0))
w /= w.mean()

# ── Ensemble training (no meta, consistent inference) ──
print(f"\nTraining ensemble (5 seeds)...")
models = []
for i, seed in enumerate(tqdm([42,43,44,45,46], desc="Training")):
    Xm_tr = X_tr_s.copy(); Xm_te = X_te_s.copy()
    np.random.seed(seed)
    Xm_tr += np.random.randn(*Xm_tr.shape) * 0.03
    m = LGBMClassifier(random_state=seed,n_estimators=300,
                       max_depth=7,num_leaves=127,min_child_samples=50,learning_rate=0.05,
                       subsample=0.85,colsample_bytree=0.85,n_jobs=-1,verbose=-1,force_col_wise=True)
    m.fit(Xm_tr, y_tr, sample_weight=w)
    acc = m.score(Xm_te, y_te)
    models.append(m)
    print(f"  Seed {seed}: val={acc:.4f}")

probas = np.mean([m.predict_proba(X_te_s)[:,1] for m in models], axis=0)

# Platt scaling calibration (logistic regression on raw probs)
from sklearn.linear_model import LogisticRegression
cal = LogisticRegression(C=1e3, random_state=42)
cal.fit(probas.reshape(-1,1), y_te)
probas_cal = cal.predict_proba(probas.reshape(-1,1))[:,1]
# Bin-level MAE: compare bin mean pred vs actual up rate
bin_preds = []; bin_actuals = []
for i in range(10):
    m = (probas_cal >= i/10) & (probas_cal < (i+1)/10)
    if m.sum() > 10:
        bin_preds.append(probas_cal[m].mean())
        bin_actuals.append(y_te[m].mean())
cal_mae = np.mean(np.abs(np.array(bin_preds) - np.array(bin_actuals))) if bin_preds else 0.5
print(f"  Calibrated: cal_bin_MAE={cal_mae:.4f}")

best_thr = 0.5; best_wr = 0; best_acc_vote = 0
for thr in np.arange(0.3,0.71,0.02):
    pred = (probas_cal >= thr).astype(int)
    acc_vote = (pred==y_te).mean()
    buy_wr = ((y_te==1)&(pred==1)).sum() / max((pred==1).sum(),1)
    score = acc_vote + 0.2 * (1 - cal_mae)  # reward accuracy + calibration
    if score > best_acc_vote and buy_wr >= 0.55:
        best_acc_vote = score; best_thr = thr; best_wr = buy_wr
best_pred = (probas_cal >= best_thr).astype(int)
final_acc = (best_pred==y_te).mean()
final_wr = ((y_te==1)&(best_pred==1)).sum() / max((best_pred==1).sum(),1)
print(f"  Optimized: thr={best_thr:.2f} acc={final_acc:.4f} buy_wr={final_wr:.4f}")

# Use calibrated probs for bins & plot
probas = probas_cal

# Compute bin averages for display (actual returns)
# Build rets from val data
rets_val = np.concatenate([FEAT_cache_te[t][1] for t in FEAT_cache_te])
bin_avg_ret = {}
for i in range(10):
    lo=i/10; hi=(i+1)/10
    m=(probas>=lo)&(probas<hi)
    if m.sum()>5:
        bin_avg_ret[round(lo,1)] = float(np.nanmean(rets_val[m]))
    else:
        bin_avg_ret[round(lo,1)] = 0.0
print(f"Bin avg returns: {bin_avg_ret}")

for i,m in enumerate(models):
    joblib.dump({"model":m,"scaler":sc,"top_idx":top_idx,"cols":cols,"pca":pca,"calibrator":cal,"all_cols":cols,"threshold":best_thr,"bin_ret":bin_avg_ret}, f"lgbm_ens_{i}.joblib")
joblib.dump({"models":models,"scaler":sc,"pca":pca,"top_idx":top_idx,"cols":cols,"calibrator":cal,"all_cols":cols,"n_models":5,"threshold":best_thr,"bin_ret":bin_avg_ret},"lgbm_ensemble.joblib")
print(f"Saved lgbm_ensemble.joblib (opt_thr={best_thr:.2f})")

# ── Calibration plot ──
print("\nGenerating calibration plot...")
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
probs = np.mean([m.predict_proba(X_te_s)[:,1] for m in models], axis=0)
bc=[];up_rates=[];avg_rets=[];cnts=[]
for i in range(10):
    m=(probs>=i/10)&(probs<(i+1)/10)
    cnts.append(m.sum())
    up_rates.append(y_te[m].mean() if m.sum()>5 else 0)
    avg_rets.append(np.nanmean((y_te[m]*2-1)*1) if m.sum()>5 else 0)
    bc.append(i/10+0.05)

fig,ax=plt.subplots(figsize=(12,7))
valid=[i for i in range(10) if cnts[i]>5]
ax.plot([bc[i] for i in valid],[up_rates[i] for i in valid],'o-',color='#2196F3',lw=2,markersize=8,label='Actual up rate')
ax.plot([0,1],[0,1],'--',color='gray',lw=1,alpha=0.5,label='Perfect')
ax.set_xlabel('Predicted UP probability',fontsize=13);ax.set_ylabel('Actual up rate',fontsize=13,color='#2196F3')
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks(np.arange(0,1.1,0.1));ax.grid(True,alpha=0.3)
for i in range(10):
    if cnts[i]>5: ax.annotate(f'n={cnts[i]}',(bc[i],up_rates[i]),fontsize=8,ha='center',xytext=(0,10),textcoords='offset points')
ax2=ax.twinx()
ax2.plot([bc[i] for i in valid],[avg_rets[i] for i in valid],'s-',color='#4CAF50',lw=2,markersize=8,label='Avg return %')
ax2.set_ylabel('Avg return (%)',fontsize=13,color='#4CAF50');ax2.axhline(y=0,color='gray',ls=':',lw=0.5)
l1,l2=ax.get_legend_handles_labels();l3,l4=ax2.get_legend_handles_labels()
ax.legend(l1+l3,l2+l4,loc='upper left',fontsize=11)
plt.title(f'Calibration — {len(probs)} preds, val acc={final_acc:.1%}',fontsize=14)
plt.tight_layout();plt.savefig('calibration.png',dpi=150,bbox_inches='tight')
plt.show()
