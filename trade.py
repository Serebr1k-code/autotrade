#!/usr/bin/env python3
"""Headless daily trader — downloads data on the fly, no cache needed."""
import os, sys, requests, json, time, numpy as np, pandas as pd, warnings, joblib
warnings.filterwarnings('ignore')
from pathlib import Path; import yfinance as yf
from datetime import datetime, timezone, timedelta
from tqdm import tqdm; import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ.get("STOCK_API_KEY", "")
BASE = "https://stocksimulator.duckdns.org"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
MIN_DAYS = 80  # 60 warmup + safety

def api_call(method, path, json_data=None, max_retries=3):
    delays = [5, 15, 30]
    for attempt in range(max_retries + 1):
        try:
            if method == "GET":
                r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=30, verify=False)
            else:
                r = requests.post(f"{BASE}{path}", headers=HEADERS, json=json_data, timeout=30, verify=False)
            if r.status_code < 500: return r
            print(f"  HTTP {r.status_code}, retry {delays[attempt]}s...", flush=True)
            time.sleep(delays[attempt])
        except Exception as e:
            if attempt < max_retries:
                print(f"  {e}, retry {delays[attempt]}s...", flush=True)
                time.sleep(delays[attempt])
            else: return None
    return None

# ── Load model ──
sys.path.insert(0, "/tmp/Reinforcement_Trading_Part_2")
from features import add_stationary_features, atr

d = joblib.load("lgbm_ensemble.joblib")
lgbm = d["models"]; scaler = d["scaler"]; pca = d.get("pca", None)
top_idx = d.get("top_idx", None); cols = d.get("cols", [])
cal_coef = 0.0; cal_int = 0.0
cal_path = Path("calibrator.json")
if cal_path.exists():
    cd = json.load(open(cal_path))
    cal_coef, cal_int = cd.get("coef", 0.0), cd.get("intercept", 0.0)
print(f"Model: {len(lgbm)} models, {len(cols)} cols, cal={cal_coef:.2f}*x+{cal_int:.2f}", flush=True)

# ── Download macro data (SPY, VIX, TNX) ──
print("Downloading macro data...", flush=True)
spy_map={}; vix_map={}; tnx_map={}
for name,col,dmap in [("SPY","close",spy_map),("^VIX","close",vix_map),("^TNX","close",tnx_map)]:
    try:
        df = yf.download(name, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty: continue
        df.columns=[c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.index=pd.to_datetime(df.index).tz_localize(None)
        vals = df[col].pct_change().values if name=="SPY" else df[col].values
        for d,v in zip(df.index,vals):
            if not np.isnan(v): dmap[d.date()]=v
    except: pass

# ── Expert map ──
expert_map = {}
try:
    from python_calamine import CalamineWorkbook; import io
    url="https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanGrowth.xlsx?sc_lang=en&hash=0F651C29D8FE2E04AB86BC7DD7EDAD2E"
    r=requests.get(url,timeout=20)
    if r.status_code==200:
        with io.BytesIO(r.content) as f: wb=CalamineWorkbook.from_filelike(f)
        data=wb.get_sheet_by_name('RGDP').to_python()
        df=pd.DataFrame(data); df.columns=df.iloc[0]; df=df.iloc[1:].reset_index(drop=True)
        df=df.replace('#N/A',np.nan)
        for c in df.columns:
            try: df[c]=pd.to_numeric(df[c],errors='coerce')
            except: pass
        spf=df[(df['YEAR']>=2019)&(df['YEAR']<=2027)&(df['QUARTER'].isin([1,2,3,4]))][['YEAR','QUARTER','drgdp2']].copy()
        spf['expert_up']=(spf['drgdp2']>0).astype(int)
        for _,r in spf.iterrows():
            yr,q,val=int(r['YEAR']),int(r['QUARTER']),int(r['expert_up'])
            qs=pd.Timestamp(f'{yr}-{(q-1)*3+1:02d}-01')
            qe=pd.Timestamp(f'{yr}-{q*3:02d}-01')+pd.offsets.MonthEnd(0)
            for d in pd.date_range(qs,qe,freq='D'): expert_map[d.date()]=val
        if expert_map:
            lv=expert_map[sorted(expert_map.keys())[0]]
            for d in pd.date_range(sorted(expert_map.keys())[0],pd.Timestamp('2026-12-31'),freq='D'):
                expert_map.setdefault(d.date(), lv)
except: pass

# ── Download tickers and build features ──
TICKERS = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","WMT",
           "JPM","XOM","UNH","V","PG","COST","HD","ORCL","MA","JNJ",
           "BAC","WFC","MRK","ABBV","CVX","KO","PEP","ADBE","CRM","CSCO",
           "ACN","MCD","NFLX","DIS","INTU","TMO","ABT","AMD","QCOM","IBM",
           "LIN","TXN","AMGN","PM","GE","HON","UBER","MS","SBUX","CAT",
           "DE","LRCX","AMAT","MU","GILD","ADI","ISRG","SYK","VRTX","PANW"]

print(f"Downloading {len(TICKERS)} tickers (last {MIN_DAYS+10}d)...", flush=True)

def build_one(t):
    try:
        df = yf.download(t, period=f"{MIN_DAYS+10}d", interval="1d", progress=False, auto_adjust=True)
        if len(df) < MIN_DAYS: return None
        df.columns=[c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df=df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df.index=pd.to_datetime(df.index).tz_localize("America/New_York"); df.index.name=None
        ft,_=add_stationary_features(df)
        p=ft["Close"];v=ft["Volume"];h=ft["High"];l=ft["Low"];o=ft["Open"];a=atr(ft,14).replace(0,np.nan)
        up=(p.diff()>0).astype(float);ft["extra_str_up"]=up*(up.groupby((up!=up.shift()).cumsum()).cumcount()+1)
        dn=(p.diff()<0).astype(float);ft["extra_str_dn"]=dn*(dn.groupby((dn!=dn.shift()).cumsum()).cumcount()+1)
        h20=h.rolling(20,5).max();l20=l.rolling(20,5).min();ft["extra_pos_rng"]=((p-l20)/(h20-l20).replace(0,np.nan)).shift(1)
        v20mn=v.rolling(20,5).min();v20mx=v.rolling(20,5).max();ft["extra_v_pos"]=((v-v20mn)/(v20mx-v20mn).replace(0,np.nan)).shift(1)
        v5=p.diff().rolling(5,3).std();v20s=p.diff().rolling(20,10).std();ft["extra_vr"]=(v5/v20s.replace(0,np.nan)).shift(1)
        ft["extra_pv_corr"]=p.rolling(10,5).corr(v).shift(1)
        h50=h.rolling(50,20).max();l50=l.rolling(50,20).min();ft["extra_dh"]=((p/h50-1)*100).shift(1);ft["extra_dl"]=((l50/p-1)*100).shift(1)
        ft["extra_gap"]=(o/p.shift(1)-1).shift(1);ft["extra_r_atr"]=((h-l)/a.replace(0,np.nan)).shift(1)
        ft["extra_body"]=((p-o).abs()/(h-l).replace(0,np.nan)).shift(1)
        ft["spy_ret"]=[spy_map.get(d.date(),0.0) for d in ft.index]
        ft["vix"]=[vix_map.get(d.date(),0.0) for d in ft.index]
        ft["tnx"]=[tnx_map.get(d.date(),0.0) for d in ft.index]
        ft["expert_up"]=[expert_map.get(d.date(),0.5) for d in ft.index]
        ft["sector_id"]=0
        ft["gap_x_vr"]=ft["extra_gap"]*ft["extra_vr"];ft["ret1_x_spy"]=ft["ret1_atr"]*ft["spy_ret"] if "ret1_atr" in ft.columns else 0
        ft["vr_x_pv"]=ft["extra_vr"]*ft["extra_pv_corr"];ft["gap_x_body"]=ft["extra_gap"]*ft["extra_body"]
        ft["ret1_x_macd"]=ft["ret1_atr"]*ft["macd_hist_atr"] if "macd_hist_atr" in ft.columns else 0
        ft=ft.iloc[60:].fillna(0).dropna(subset=cols)
        if len(ft)<5: return None
        X=np.nan_to_num(ft[cols].values[-1:],nan=0,posinf=0,neginf=0)
        return X
    except: return None

predictions = []
for t in tqdm(TICKERS, desc="Predict"):
    X = build_one(t)
    if X is None: continue
    Xs = scaler.transform(X)
    if pca is not None: Xs = np.hstack([Xs, pca.transform(Xs)])
    if top_idx is not None: Xs = Xs[:, top_idx]
    raw = np.mean([m.predict_proba(Xs)[:,1] for m in lgbm], axis=0)[0]
    cal = np.clip(1/(1+np.exp(-(cal_coef*raw+cal_int))), 0.001, 0.999)
    if cal >= 0.56:
        predictions.append((t, cal, raw))
print(f"Signals: {len(predictions)} bullish", flush=True)

if not predictions: print("No bullish signals."); sys.exit(0)

# ── Get live prices ──
selected = []
for t, cal, raw in sorted(predictions, key=lambda x:-x[1])[:20]:
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty: continue
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        price = float(df['Close'].iloc[-1])
        if price <= 0: continue
        selected.append((t, cal*100, price))
        print(f"  {t:>6s}  {cal*100:5.1f}%  ${price:<8.2f}", flush=True)
    except Exception as e:
        print(f"  {t}: price error {e}", flush=True)
if not selected:
    print("No prices fetched (rate limit?).", flush=True)
    sys.exit(0)

# ── Trade ──
if not API_KEY: print("STOCK_API_KEY not set, skipping trade."); sys.exit(0)
print(f"\n=== Sell existing ===", flush=True)
r = api_call("GET", "/api/v1/external/portfolio")
cash = 0
if r and r.status_code == 200:
    data = r.json(); cash = data.get("cash_balance", 0)
    print(f"  Cash: ${cash:.2f}", flush=True)
    positions = data.get("positions", data.get("holdings", []))
    if isinstance(positions, dict): positions = [{"ticker":k,**v} for k,v in positions.items()]
    sell_orders = [{"symbol":p.get("ticker",p.get("symbol","")), "action":"SELL", "quantity":int(p.get("quantity",p.get("shares",0)))}
                   for p in positions if p.get("quantity",p.get("shares",0)) > 0]
    if sell_orders:
        print(f"  Selling {len(sell_orders)}...", flush=True)
        try:
            r2 = requests.post(f"{BASE}/api/v1/external/v1/trading/batch", headers=HEADERS,
                              json={"orders": sell_orders}, timeout=30, verify=False)
            if r2 and r2.status_code == 200:
                for x in r2.json().get("results", []):
                    s=x.get("symbol",""); st=x.get("status",""); e=x.get("error_code",""); m=x.get("message","")
                    if e or m: print(f"    {s}: {e} — {m}", flush=True)
                    else: print(f"    ✅ {s}: sold {int(x.get('quantity',0))}sh @ ${x.get('price',0):.2f}", flush=True)
        except Exception as e: print(f"    Sell failed: {e}", flush=True)
        r3 = api_call("GET", "/api/v1/external/portfolio")
        if r3 and r3.status_code == 200: cash = r3.json().get("cash_balance", cash)
    print(f"  Cash after sell: ${cash:.2f}", flush=True)
else: print("  Could not fetch portfolio.", flush=True)

print(f"\n=== Buy {len(selected)} signals ===", flush=True)
accs = np.array([p[1] for p in selected]); weights = accs**2; weights /= weights.sum()
buy_orders = []
for (t, acc, price), w in zip(selected, weights):
    s = max(1, int(cash * w / price))
    buy_orders.append({"symbol": t, "action": "BUY", "quantity": s})
    print(f"  BUY {t:>6s}: {s:>4d}sh × ${price:<8.2f} = ${s*price:<8.2f}  ({acc:.1f}%)", flush=True)

print(f"\nPlacing {len(buy_orders)} orders...", flush=True)
try:
    r = requests.post(f"{BASE}/api/v1/external/v1/trading/batch", headers=HEADERS,
                     json={"orders": buy_orders}, timeout=30, verify=False)
    if r and r.status_code == 200:
        for x in r.json().get("results", []):
            s=x.get("symbol",""); st=x.get("status",""); pr=x.get("price") or 0; q=x.get("quantity") or 0; e=x.get("error_code",""); m=x.get("message","")
            if e or m: print(f"  {s}: {e} — {m}", flush=True)
            else: print(f"  ✅ {s}: BUY {int(q)}sh @ ${pr:.2f}", flush=True)
    else: print(f"  API error: {r.status_code if r else 'no response'}", flush=True)
except Exception as e: print(f"  Buy failed: {e}", flush=True)

print("\nDone!", flush=True)
