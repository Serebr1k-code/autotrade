#!/usr/bin/env python3
"""Download 4500+ NASDAQ stocks (1 year daily data), build patterns, save cache."""
import yfinance as yf, warnings, pandas as pd, numpy as np, joblib, pickle
from pathlib import Path; from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import requests
warnings.filterwarnings('ignore')

# Get all US tickers (NASDAQ + NYSE + AMEX)
all_t = set()
for url in ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"]:
    try:
        df = pd.read_csv(url, sep="|")
        all_t.update(s for s in df["Symbol"].dropna() if "TEST" not in s and len(s) <= 4
                     and not any(s.endswith(x) for x in ["W","U","R","P","V","Z","Y"]))
    except: pass
all_t = sorted(all_t)
print(f"Total US tickers: {len(all_t)}")

cache_file = Path("cache_data.joblib")
existing = set(pickle.load(open(cache_file,'rb'))) if cache_file.exists() else set()
new_t = [t for t in all_t if t not in existing and t.replace('-','.') not in existing]
print(f"Already have: {len(existing)}, New: {len(new_t)}")

def dl(t):
    try:
        df = yf.download(t.replace('-','.'), period="5y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 200: return None
        df.columns = [c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return (t, df)
    except: return None

# Download in chunks with incremental save
chunk_size = 200
all_data = pickle.load(open(cache_file,'rb')) if cache_file.exists() else {}

for ci in range(0, len(new_t), chunk_size):
    chunk = new_t[ci:ci+chunk_size]
    print(f"\nChunk {ci//chunk_size+1}/{(len(new_t)-1)//chunk_size+1} ({len(chunk)} tickers)")
    new_data = {}
    with ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(dl, t): t for t in chunk}
        for f in tqdm(as_completed(futs), total=len(chunk), desc="DL"):
            r = f.result()
            if r: new_data[r[0]] = r[1]
    all_data.update(new_data)
    pickle.dump(all_data, open(cache_file,'wb'), protocol=5)
    print(f"  Saved. Total: {len(all_data)}")

print(f"\nDone! Total: {len(all_data)} tickers cached")

# Auto-backup
pickle.dump(all_data, open(Path("cache_data.backup"),'wb'), protocol=5)
print(f"Backup saved to cache_data.backup")
