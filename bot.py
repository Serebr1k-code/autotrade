#!/usr/bin/env python3
"""Bot: 2-candle pattern strategy with interactive TUI selection."""
import os, sys, requests, json, time, numpy as np, pandas as pd, warnings, joblib, curses
from tqdm import tqdm
from pathlib import Path; from collections import defaultdict
import yfinance as yf; import urllib3
from datetime import datetime, timezone, timedelta
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

API_KEY = os.environ.get("STOCK_API_KEY")
if not API_KEY: API_KEY = "st_sim_live_XHjCjmIz_VBfr8WY4wo59N4uG_DMbh9QgRBS0zpmpsw"
BASE = "https://stocksimulator.duckdns.org"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
TIMEOUT = 30

def api_call(method, path, json_data=None, max_retries=3):
    """API call with retry and exponential backoff."""
    delays = [5, 15, 30]
    for attempt in range(max_retries + 1):
        try:
            if method == "GET":
                r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=TIMEOUT, verify=False)
            else:
                r = requests.post(f"{BASE}{path}", headers=HEADERS, json=json_data, timeout=TIMEOUT, verify=False)
            if r.status_code < 500:  # 2xx, 4xx — не ретраим
                return r
            print(f"  HTTP {r.status_code}, retry in {delays[attempt]}s...", flush=True)
        except Exception as e:
            if attempt < max_retries:
                print(f"  timeout, retry in {delays[attempt]}s...", flush=True)
            else:
                print(f"  failed: {type(e).__name__}", flush=True)
                return None
        time.sleep(delays[attempt])
    return None

TICKERS = ["META","JPM","NVDA","AAPL","MSFT","GOOGL","AMZN","TSLA","AVGO","ADBE",
           "CRM","ORCL","CSCO","INTC","AMD","QCOM","TXN","IBM","MU","ANET",
           "PANW","DELL","HPQ","SNOW","PLTR","NOW","BAC","GS","V","MA",
           "WFC","C","MS","AXP","BLK","SPGI","COF","PNC","USB","BRK-B",
           "UNH","JNJ","LLY","PFE","ABBV","MRK","TMO","ABT","WMT","KO",
           "PEP","MCD","NKE","PG","XOM","CVX","DIS","NFLX","SBUX","COST"]

msk_tz = timezone(timedelta(hours=3))
today = datetime.now(msk_tz).strftime("%Y-%m-%d")
print(f"Today (MSK): {today}")

# Patterns not needed — using PPO only

import yfinance as yf; import requests as req
yf_session = req.Session(); yf_session.headers['User-Agent'] = 'Mozilla/5.0'

# Load cached data
import pickle as _pk
cache = Path("cache_data.joblib")
backup = Path("cache_data.backup")
if not cache.exists() and not backup.exists():
    print("Run download_all.py first!"); sys.exit(1)
all_cached = {}
if cache.exists():
    try:
        all_cached = joblib.load(cache)
    except:
        try:
            with open(cache, "rb") as _f: all_cached = _pk.load(_f)
        except:
            pass
if not all_cached and backup.exists():
    print("Cache corrupted! Loading from backup...")
    try:
        all_cached = joblib.load(backup)
    except:
        try:
            with open(backup, "rb") as _f: all_cached = _pk.load(_f)
        except:
            pass
    if all_cached:
        _pk.dump(all_cached, open(cache, "wb"), protocol=5)
print(f"Loaded {len(all_cached)} tickers")



today_msk = datetime.now(msk_tz)
yesterday = (today_msk - timedelta(days=1)).strftime("%Y-%m-%d")

# Refresh last few days from yfinance if cache is stale
import yfinance as yf

# ── Expert prediction feature ──
def get_expert_map():
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
        daily = {}
        for _, row in spf.iterrows():
            yr, q, val = int(row['YEAR']), int(row['QUARTER']), int(row['expert_up'])
            q_start = pd.Timestamp(f'{yr}-{(q-1)*3+1:02d}-01')
            q_end = pd.Timestamp(f'{yr}-{q*3:02d}-01') + pd.offsets.MonthEnd(0)
            for d in pd.date_range(q_start, q_end, freq='D'):
                daily[d.date()] = val
        all_dates = sorted(daily.keys())
        if all_dates:
            last_val = daily[all_dates[0]]
            for d in pd.date_range(all_dates[0], pd.Timestamp('2026-12-31'), freq='D'):
                if d.date() in daily: last_val = daily[d.date()]
                else: daily[d.date()] = last_val
        return daily
    except Exception as e:
        print(f"Expert data failed: {e}")
        return {}

# Load LightGBM model
sys.path.insert(0, "/tmp/Reinforcement_Trading_Part_2")
from features import add_stationary_features, atr
lgbm = None; lgbm_scaler = None; lgbm_pca = None; lgbm_cal_coef = 0.0; lgbm_cal_int = 0.0; lgbm_cols = []; lgbm_bin_ret = {}
lgbm_path = Path("lgbm_ensemble.joblib")
if lgbm_path.exists():
    d = joblib.load(lgbm_path)
    if "models" in d:  # ensemble
        lgbm = d["models"]
        lgbm_scaler = d["scaler"]
        lgbm_pca = d.get("pca", None)
        lgbm_top = d.get("top_idx", None)
        lgbm_n_in = lgbm_top.shape[0] if lgbm_top is not None else 1
        lgbm_cols = d.get("cols", [])
        lgbm_buy_thr = d.get("buy_thr", 0.33)
        lgbm_num_class = d.get("num_class", 2)
        lgbm_bin_ret = d.get("bin_ret", {})
        print(f"Ensemble loaded ({len(lgbm)} models, {lgbm_n_in} features)")
    else:  # single model
        lgbm = [d["model"]]
        lgbm_scaler = d["scaler"]
        lgbm_pca = d.get("pca", None)
        lgbm_top = d.get("top_idx", None)
        lgbm_n_in = lgbm[0].n_features_in_
        lgbm_cols = d.get("all_cols", d.get("cols", []))
        lgbm_buy_thr = d.get("buy_thr", 0.5)
        lgbm_num_class = d.get("num_class", 2)
    # Load calibrator (Platt scaling coefficients)
    cal_path = Path("calibrator.json")
    if cal_path.exists():
        import json
        cal_d = json.load(open(cal_path))
        lgbm_cal_coef = cal_d.get("coef", 0.0)
        lgbm_cal_int = cal_d.get("intercept", 0.0)
        print(f"Calibrator loaded: coef={lgbm_cal_coef:.2f} intercept={lgbm_cal_int:.2f}")
else:
    lgbm_buy_thr = 0.5; lgbm_num_class = 2; lgbm_bin_ret = {}
    print("No model found — run train_ensemble.py first")

# Load SPY, VIX, TNX for market features
spy_ret_map = {}; vix_map = {}; tnx_map = {}
for name, col, dmap in [("SPY","close",spy_ret_map),("^VIX","close",vix_map),("^TNX","close",tnx_map)]:
    try:
        df = yf.download(name, period="2y", interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        vals = df[col].pct_change().values if name=="SPY" else df[col].values
        for d,v in zip(df.index, vals):
            if not np.isnan(v): dmap[d.date()] = v
    except: pass
expert_map = get_expert_map()
print(f"Market data: SPY={len(spy_ret_map)} VIX={len(vix_map)} TNX={len(tnx_map)} Expert={len(expert_map)} days")

# Load sector map
sector_path = Path("sector_cache.joblib")
sectors = joblib.load(sector_path) if sector_path.exists() else {}
sector_ids = {s:i for i,s in enumerate(set(sectors.values()))} if sectors else {}
print(f"Sectors: {len(sector_ids)} ({len(sectors)} tickers mapped)")

cached_end = max(d.index[-1] for d in all_cached.values())
yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
days_missing = (pd.Timestamp.now() - cached_end).days
# Only refresh if cache is at least 2 days old (skip yesterday/weekend)
if days_missing > 1:
        import requests as req; sess = req.Session(); sess.headers['User-Agent'] = 'Mozilla/5.0'
        print(f"Cache ends {cached_end.date()}, checking for new data...")
        new_total = 0
        for t in tqdm(TICKERS, desc="yfinance"):
            if t not in all_cached: continue
            try:
                df = yf.download(t.replace('-','.'), period=f"{days_missing+2}d", interval='1d', progress=False, auto_adjust=True)
                if df.empty: continue
                df.columns = [c[0].lower() for c in df.columns]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[df.index.date < pd.Timestamp.now().date()]
                new_rows = df[~df.index.isin(all_cached[t].index)]
                if len(new_rows):
                    new_rows = new_rows.dropna(subset=['open','high','low','close'])
                    all_cached[t] = pd.concat([all_cached[t], new_rows]).sort_index()
                    new_total += len(new_rows)
            except: pass
        if new_total > 0:
            joblib.dump(all_cached, cache)
            print(f"Added {new_total} new rows, cache updated.")
        else:
            print("No new data.")

print(f"Predicting from cache data ({len(TICKERS)} tickers, PPO model)...")
predictions = []
last_dates = []
for t in tqdm(TICKERS, desc="Predicting"):
    try:
        if t not in all_cached: continue
        d = all_cached[t]
        if len(d) < 100: continue
        
        o = d["open"].values.astype(float); h = d["high"].values.astype(float)
        l = d["low"].values.astype(float); c = d["close"].values.astype(float)
        v = d["volume"].values.astype(float)
        # Remove rows with NaN (partial day at end of cache)
        bad = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
        if bad.any():
            o, h, l, c, v = o[~bad], h[~bad], l[~bad], c[~bad], v[~bad]
            if len(o) < 100: continue
        if len(o) < 100: continue
        last_dates.append(d.index[-1])
        
        # LightGBM prediction
        if lgbm is not None and len(o) >= 80:
            try:
                import sys as _sys; _sys.path.insert(0, "/tmp/Reinforcement_Trading_Part_2")
                from features import add_stationary_features as _sf, atr as _atr
                df = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v})
                df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(o), freq="D")
                ft, _ = _sf(df)
                # Extra features (mirrors train_lgbm.py)
                p = df["Close"]; vv = df["Volume"]; hh = df["High"]; ll = df["Low"]; oo = df["Open"]
                a = _atr(df,14).replace(0,np.nan)
                up = (p.diff()>0).astype(float)
                ft["extra_str_up"] = up * (up.groupby((up!=up.shift()).cumsum()).cumcount()+1)
                dn = (p.diff()<0).astype(float)
                ft["extra_str_dn"] = dn * (dn.groupby((dn!=dn.shift()).cumsum()).cumcount()+1)
                h20=hh.rolling(20,5).max(); l20=ll.rolling(20,5).min()
                ft["extra_pos_rng"] = ((p-l20)/(h20-l20).replace(0,np.nan)).shift(1)
                v20mn=vv.rolling(20,5).min(); v20mx=vv.rolling(20,5).max()
                ft["extra_v_pos"] = ((vv-v20mn)/(v20mx-v20mn).replace(0,np.nan)).shift(1)
                v5=p.diff().rolling(5,3).std(); v20s=p.diff().rolling(20,10).std()
                ft["extra_vr"] = (v5/v20s.replace(0,np.nan)).shift(1)
                ft["extra_pv_corr"] = p.rolling(10,5).corr(vv).shift(1)
                h50=hh.rolling(50,20).max(); l50=ll.rolling(50,20).min()
                ft["extra_dh"] = ((p/h50-1)*100).shift(1)
                ft["extra_dl"] = ((l50/p-1)*100).shift(1)
                ft["extra_gap"] = (oo/p.shift(1)-1).shift(1)
                ft["extra_r_atr"] = ((hh-ll)/a.replace(0,np.nan)).shift(1)
                ft["extra_body"] = ((p-oo).abs()/(hh-ll).replace(0,np.nan)).shift(1)
                ft["spy_ret"] = [spy_ret_map.get(d.date(), 0.0) for d in df.index]
                ft["vix"] = [vix_map.get(d.date(), 0.0) for d in df.index]
                ft["tnx"] = [tnx_map.get(d.date(), 0.0) for d in df.index]
                ft["expert_up"] = [expert_map.get(d.date(), 0.5) for d in df.index]
                ft["sector_id"] = sector_ids.get(sectors.get(t,"Unknown"), 0)
                ft["gap_x_vr"] = ft["extra_gap"] * ft["extra_vr"]
                ft["ret1_x_spy"] = ft["ret1_atr"] * ft["spy_ret"] if "ret1_atr" in ft.columns else 0
                ft["vr_x_pv"] = ft["extra_vr"] * ft["extra_pv_corr"]
                ft["gap_x_body"] = ft["extra_gap"] * ft["extra_body"]
                ft["ret1_x_macd"] = ft["ret1_atr"] * ft["macd_hist_atr"] if "macd_hist_atr" in ft.columns else 0
                
                ft = ft.iloc[60:].fillna(0).dropna(subset=lgbm_cols)
                if len(ft) >= 1:
                    raw_obs = ft[lgbm_cols].values[-1:]
                    obs_s = lgbm_scaler.transform(raw_obs)
                    if lgbm_pca is not None:
                        obs_pca = lgbm_pca.transform(obs_s)
                        obs_s = np.hstack([obs_s, obs_pca])
                    if lgbm_top is not None:
                        obs_s = obs_s[:, lgbm_top[:lgbm_n_in]]
                    probas_all = np.mean([m.predict_proba(obs_s) for m in lgbm], axis=0)
                    # Platt calibration: p = 1/(1 + exp(-(coef * raw + intercept)))
                    raw = probas_all[0,1] if probas_all.ndim==2 and probas_all.shape[1]>=2 else probas_all[0]
                    raw = probas_all[0, 2] if probas_all.ndim==2 and probas_all.shape[1]==3 else (probas_all[0, 1] if probas_all.ndim==2 else probas_all[1])
                    cal_logit = lgbm_cal_coef * raw + lgbm_cal_int
                    proba_up = 1.0 / (1.0 + np.exp(-cal_logit))
                    proba_up = np.clip(proba_up, 0.001, 0.999)
                    proba = proba_up
                    if probas_all.ndim==2 and probas_all.shape[1]==3:
                        pred = 2 if proba_up >= lgbm_buy_thr else 0
                    else:
                        pred = 1 if proba_up >= 0.56 else 0
                    ha_cl = (c[-2:]+h[-2:]+l[-2:]+c[-2:])/4
                    ha_op = np.zeros(2); ha_op[0] = (c[-3]+o[-3])/2 if len(o)>2 else (c[-2]+o[-2])/2
                    ha_op[1] = (ha_op[0]+ha_cl[0])/2; ha_d = "▲" if ha_cl[-1]>ha_op[-1] else "▼"
                    vr = v[-1]/np.mean(v[-20:]) if len(v)>=20 else 1
                    vm = "🔥" if vr>1.5 else "·"; gp = (o[-1]/c[-2]-1)*100 if len(o)>1 else 0
                    gm = "⬆" if gp>0.3 else "⬇" if gp<-0.3 else "·"
                    act_s = "LONG" if pred>=1 else "FLAT"
                    predictions.append((t, proba*100, 1, 0.0, f"LGBM:{act_s}", "ML", f"{ha_d}{vm}{gm}", pred==1, 0.0))
            except Exception as e:
                print(f"  PRED ERR {t}: {e}")
    except Exception as e:
        print(f"  ERROR {t}: {e}")

last_data_date = max(last_dates).strftime("%m-%d") if last_dates else "?"
print(f"Predictions: {len(predictions)} signals (data up to {last_data_date})")
predictions.sort(key=lambda x: -x[1])

# Fetch prices + changes for TUI display (parallel, fast)
from concurrent.futures import ThreadPoolExecutor, as_completed
def fetch_price(i):
    t = predictions[i][0]; tk = t.replace('-','.')
    # Try API first (real-time prices)
    try:
        r = requests.get(f"{BASE}/api/v1/external/prices/{t}", headers=HEADERS, timeout=5, verify=False)
        if r.status_code == 200:
            d = r.json()
            price = float(d.get("current_price", 0))
            if price > 0:
                # Compute change from cached close
                if t in all_cached and price > 0:
                    last_close = all_cached[t]["close"].dropna().iloc[-1]
                    chg = (price / last_close - 1) * 100
                else:
                    chg = float(d.get("percent_change", 0))
                return i, price, chg
    except: pass
    # Fallback: yfinance
    try:
        hs = yf.Ticker(tk).history(period="2d")
        if len(hs) >= 1:
            price = float(hs["Close"].iloc[-1])
            if price > 0:
                if t in all_cached:
                    lc = all_cached[t]["close"].dropna().iloc[-1]
                    return i, price, (price/lc - 1) * 100
                return i, price, 0.0
    except: pass
    return i, 0.0, 0.0

with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(fetch_price, i): i for i in range(len(predictions))}
    for f in tqdm(as_completed(futs), total=len(predictions), desc="Prices"):
        i, price, chg = f.result()
        if price > 0:
            predictions[i] = (predictions[i][0], predictions[i][1], predictions[i][2], price,
                            predictions[i][4], predictions[i][5], predictions[i][6], predictions[i][7], chg)
        elif len(predictions[i]) <= 8:
            predictions[i] = predictions[i] + (0.0,)
predictions = [p if len(p) > 8 else p + (0.0,) for p in predictions]
ok_prices = sum(1 for p in predictions if p[3] > 0)
print(f"Prices: {ok_prices}/{len(predictions)} loaded ({len(predictions)-ok_prices} N/A)")

LOGO = """
  ███████╗████████╗ ██████╗ ██╗  ██╗███████╗██████╗  ██████╗ ████████╗
  ██╔════╝╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝
  ███████╗   ██║   ██║   ██║█████╔╝ ███████╗██████╔╝██║   ██║   ██║   
  ╚════██║   ██║   ██║   ██║██╔═██╗ ╚════██║██╔══██╗██║   ██║   ██║   
  ███████║   ██║   ╚██████╔╝██║  ██╗███████║██████╔╝╚██████╔╝   ██║   
  ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝    ╚═╝   
"""

# ====== CURSES TUI ======
def main_tui(stdscr):
    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)    # selected
    curses.init_pair(2, curses.COLOR_CYAN, -1)      # header
    curses.init_pair(3, curses.COLOR_GREEN, -1)     # buy
    curses.init_pair(4, curses.COLOR_WHITE, -1)     # normal
    curses.init_pair(5, curses.COLOR_YELLOW, -1)    # cursor (bold yellow)
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # cursor on selected (black on yellow bg)
    curses.init_pair(7, curses.COLOR_RED, -1)       # red change
    curses.init_pair(8, curses.COLOR_GREEN, -1)     # green change
    
    selected = [predictions[i][1] >= 80.0 and predictions[i][7] for i in range(len(predictions))]
    cursor = 0
    
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        
        # Header
        n_sel = sum(selected)
        top_n = min(10, len(predictions))
        stdscr.addstr(0, 0, f" {today} MSK  |  Data: ~{last_data_date}  |  Top{top_n}+{len(predictions)-top_n}  |  [{n_sel}]sel  |  Space=tog  Enter=buy  →=stats  q=quit", curses.color_pair(2))
        stdscr.addstr(1, 0, "─" * min(w-1, 100))
        
        # List: show top 10 by default, scroll for more
        list_start = 0
        max_visible = h - 8
        if cursor < 10:
            list_start = 0
        else:
            list_start = max(0, cursor - max_visible + 1)
        
        visible_end = min(len(predictions), list_start + max_visible)
        for i in range(list_start, visible_end):
            y = i - list_start + 2
            if y >= h - 2: break
            data = predictions[i]
            t, acc, cnt, price, pat, c1, c2, ha_up = data[:8]
            chg = data[8] if len(data) > 8 else 0
            ha_mark = " ▲" if ha_up else " ▼"
            sel_char = "●" if selected[i] else " "
            # Expected return from calibration
            bin_idx = min(int(acc // 10), 9)
            exp_ret = lgbm_bin_ret.get(bin_idx * 0.1, 0.0)
            price_str = f"${price:>7.2f}" if price > 0 else "  N/A   "
            chg_str = f" {chg:>+5.2f}%" if price > 0 else "   N/A  "
            line = f"[{sel_char}] {t:>6s}{ha_mark}  {acc:>5.1f}%  {exp_ret:>+5.2f}%  {price_str}{chg_str}"
            
            # Determine base color for the line
            if i == cursor and selected[i]:
                base_color = curses.color_pair(6) | curses.A_BOLD
            elif i == cursor:
                base_color = curses.color_pair(5) | curses.A_BOLD
            elif selected[i]:
                base_color = curses.color_pair(1)
            else:
                base_color = curses.color_pair(4)
            
            # Write line without change part, then change part in color
            base_len = len(line) - len(chg_str)
            stdscr.addstr(y, 0, line[:base_len], base_color)
            if chg > 0:
                stdscr.addstr(y, base_len, chg_str, curses.color_pair(8) | curses.A_BOLD)  # green
            elif chg < 0:
                stdscr.addstr(y, base_len, chg_str, curses.color_pair(7) | curses.A_BOLD)  # red
            else:
                stdscr.addstr(y, base_len, chg_str, base_color)
        
        # Bottom — ASCII logo + cursor info
        logo_lines = LOGO.strip().split("\n")
        logo_y = h - len(logo_lines) - 1
        if logo_y > 5:
            for li, lline in enumerate(logo_lines):
                if li >= h - logo_y: break
                stdscr.addstr(logo_y + li, 0, lline, curses.color_pair(2))
        
        if len(predictions) > 0:
            cur = predictions[cursor]; t, acc, cnt, price = cur[0], cur[1], cur[2], cur[3]
            pat, c1, c2, ha_up, chg = cur[4], cur[5], cur[6], cur[7], cur[8] if len(cur) > 8 else 0
            ha_s = "▲ uptrend" if ha_up else "▼ downtrend"
            bot_txt = f" {t}: HA={ha_s}  acc={acc:.1f}%  ${price:.2f}  {chg:+.2f}%"
            if len(bot_txt) >= w: bot_txt = bot_txt[:w-1]
            stdscr.addstr(h-1, 0, bot_txt, curses.color_pair(2))
        
        stdscr.refresh()
        
        key = stdscr.getch()
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == curses.KEY_UP and cursor > 0:
            cursor -= 1
        elif key == curses.KEY_DOWN and cursor < len(predictions) - 1:
            cursor += 1
        elif key == ord(' '):
            # Check if shift is pressed
            selected[cursor] = not selected[cursor]
        elif key == ord('\n') or key == ord('\r'):  # Enter
            break
        elif key == curses.KEY_RIGHT:
            # Show stats
            cur = predictions[cursor]; t, acc, cnt, price = cur[0], cur[1], cur[2], cur[3]
            pat, c1, c2, ha_up, chg = cur[4], cur[5], cur[6], cur[7], cur[8] if len(cur) > 8 else 0
            stdscr.clear()
            stdscr.addstr(0, 0, f" {t} — detailed stats (press any key to return)", curses.color_pair(2))
            stdscr.addstr(2, 0, f" Pattern:    {c1} | {c2}")
            stdscr.addstr(3, 0, f" Accuracy:   {acc:.1f}%")
            stdscr.addstr(4, 0, f" History:    {cnt} occurrences in training data")
            stdscr.addstr(5, 0, f" Price:      ${price:.2f}  ({chg:+.2f}% from cached close)")
            stdscr.addstr(6, 0, f" Current:    {'BUY signal' if acc >= 82 else 'NO signal'}  |  HA: {'▲ uptrend' if ha_up else '▼ downtrend'}")
            
            # Decode candles
            for idx, code in enumerate([c1, c2]):
                if len(code) >= 5:
                    color = "GREEN" if code[0] == 'G' else "RED" if code[0] == 'R' else "DOJI"
                    size = {"S":"Small","M":"Medium","L":"Large"}.get(code[1], code[1])
                    day = {"M":"Mon","T":"Tue","W":"Wed","H":"Thu","F":"Fri"}.get(code[2] if len(code)==5 else code[2], code[2] if len(code)>=5 else '?')
                    gap = {"U":"Up","F":"Flat","D":"Down"}.get(code[3] if len(code)>=5 else '?', '?')
                    vol = {"N":"Normal","H":"High"}.get(code[4] if len(code)>=5 else '?', '?')
                    stdscr.addstr(8 + idx, 0, f" Candle {idx+1}: {color}  Body={size}  Day={day}  Gap={gap}  Vol={vol}")
            
            stdscr.addstr(11, 0, f" This pattern was correct {int(acc*cnt/100)}/{int(cnt)} times in history.")
            stdscr.addstr(12, 0, f" In {int(cnt) - int(acc*cnt/100)} cases it was wrong (predicted BUY but market went DOWN).")
            stdscr.refresh()
            stdscr.getch()  # Wait for key
    
    # Return selected tickers (without ha_up for simplicity)
    return [(predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3]) 
            for i in range(len(predictions)) if selected[i]]

selected_preds = curses.wrapper(main_tui)

# Fetch prices only for selected tickers
if selected_preds:
    print("Fetching prices for selected tickers...")
    for i in tqdm(range(len(selected_preds)), desc="Prices"):
        t = selected_preds[i][0]
        r = api_call("GET", f"/api/v1/external/prices/{t}")
        if r and r.status_code == 200:
            price = float(r.json()["current_price"])
            selected_preds[i] = (t, selected_preds[i][1], selected_preds[i][2], price)
        else:
            print(f"  ⚠️ {t}: price unavailable")
    # Remove tickers with no price
    selected_preds = [p for p in selected_preds if p[3] > 0]
    if not selected_preds:
        print("No tickers with prices available. Exiting.")
        sys.exit(0)

# Sell all first (batch)
print("\n=== Sell all existing positions ===")
r = api_call("GET", "/api/v1/external/portfolio")
CASH = 0
if r and r.status_code == 200:
    data = r.json()
    CASH = data.get("cash_balance", 0)
    print(f"Cash: ${CASH:.2f}")
    positions = data.get("positions", data.get("holdings", []))
    if isinstance(positions, dict): positions = [{"ticker":k,**v} for k,v in positions.items()]
    sell_orders = [{"symbol":p.get("ticker",p.get("symbol","")), "action":"SELL", "quantity":int(p.get("quantity",p.get("shares",0)))}
                   for p in positions if p.get("quantity",p.get("shares",0)) > 0]
    if sell_orders:
        print(f"Selling {len(sell_orders)} positions in batch...", flush=True)
        try:
            r2 = requests.post(f"{BASE}/api/v1/external/v1/trading/batch", headers=HEADERS,
                              json={"orders": sell_orders}, timeout=30, verify=False)
            if r2 and r2.status_code == 200:
                res = r2.json()
                if "results" in res:
                    for x in res["results"]:
                        sym = x.get("symbol","")
                        err = x.get("error_code","")
                        msg = x.get("message","")
                        status = x.get("status","")
                        if err or msg:
                            print(f"  {sym}: {err} — {msg}")
                        elif status == "success":
                            print(f"  ✅ {sym}: sold {int(x.get('quantity',0))}sh @ ${x.get('price',0):.2f}")
                        else:
                            print(f"  {sym}: {status} {int(x.get('quantity',0))}sh @ ${x.get('price',0):.2f}")
                else:
                    print(f"  Response: {res}")
            else:
                print(f"  ❌ Sell API error {r2.status_code if r2 else 'no response'}: {r2.text[:200] if r2 else 'connection failed'}")
        except Exception as e:
            print(f"  ❌ Sell failed: {e}")
        # Re-fetch cash after selling (proceeds now available)
        r3 = api_call("GET", "/api/v1/external/portfolio")
        if r3 and r3.status_code == 200:
            CASH = r3.json().get("cash_balance", CASH)
            print(f"Cash after sell: ${CASH:.2f}")
    else:
        print("  No positions to sell.")
if selected_preds:
    print(f"\n=== BUY {len(selected_preds)} / {len(predictions)} ===")
    # Confidence-weighted allocation (improvement #1)
    accs = np.array([p[1] for p in selected_preds])
    weights = accs ** 2
    weights /= weights.sum()
    for (t, acc, cnt, price), w in zip(selected_preds, weights):
        s = max(1, int(CASH * w / price))
        print(f"  BUY {t:>6s}: {s:>4d} sh × ${price:<8.2f} = ${s*price:<8.2f}  (acc={acc:.1f}%, weight={w*100:.0f}%)")
    if input(f"\nPlace {len(selected_preds)} orders? (y/N): ").strip().lower() == 'y':
        buy_orders = []
        for (t, acc, cnt, price), w in zip(selected_preds, weights):
            s = max(1, int(CASH * w / price))
            buy_orders.append({"symbol":t, "action":"BUY", "quantity":s})
        print(f"Sending {len(buy_orders)} buy orders in batch...", flush=True)
        try:
            r = requests.post(f"{BASE}/api/v1/external/v1/trading/batch", headers=HEADERS,
                             json={"orders": buy_orders}, timeout=30, verify=False)
            if r and r.status_code == 200:
                res = r.json()
                if "results" in res:
                    for x in res["results"]:
                        sym = x.get("symbol","")
                        status = x.get("status","")
                        price = x.get("price") or 0
                        qty = x.get("quantity") or 0
                        msg = x.get("message","")
                        err = x.get("error_code","")
                        if err or msg:
                            print(f"  {sym}: {err} — {msg}")
                        else:
                            print(f"  ✅ {sym}: {status} {int(qty)}sh @ ${price:.2f}")
                else:
                    print(f"  Response: {res}")
            else:
                print(f"  ❌ API error {r.status_code if r else 'no response'}: {r.text[:200] if r else 'connection failed'}")
        except Exception as e:
            print(f"  ❌ Buy failed: {e}")
        print("Done!")
else:
    print("Nothing selected to buy.")
