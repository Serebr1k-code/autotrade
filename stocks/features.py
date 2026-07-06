import pandas as pd, numpy as np

def atr(df, n=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def add_stationary_features(df):
    p = df['Close']
    a = atr(df).replace(0, np.nan)

    df['ret1_atr'] = (p.diff() / a.shift()).shift(1)
    df['ret5_atr'] = (p.diff(5) / a.shift(5)).shift(1)

    ema12 = p.ewm(span=12).mean()
    ema26 = p.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_hist_atr'] = ((macd - signal) / a).shift(1)

    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi14'] = (100 - 100 / (1 + rs)) / 100
    df['rsi14'] = df['rsi14'].shift(1)

    ma20 = p.rolling(20).mean()
    df['ma20_dist'] = ((p - ma20) / a).shift(1)
    ma50 = p.rolling(50).mean()
    df['ma50_dist'] = ((p - ma50) / a).shift(1)

    bb_mid = p.rolling(20).mean()
    bb_std = p.rolling(20).std()
    df['bb_pos'] = ((p - bb_mid) / (2 * bb_std)).shift(1)

    v = df['Volume']
    v_ma20 = v.rolling(20).mean()
    df['vol_ratio'] = (v / v_ma20.replace(0, np.nan)).shift(1)

    df['volatility'] = p.diff().rolling(20).std().shift(1)
    df['mom1'] = (p.diff() / p.shift()).shift(1)
    df['mom5'] = (p.diff(5) / p.shift(5)).shift(1)
    df['mom20'] = (p.diff(20) / p.shift(20)).shift(1)

    fcols = ['ret1_atr', 'ret5_atr', 'macd_hist_atr', 'rsi14',
             'ma20_dist', 'ma50_dist', 'bb_pos', 'vol_ratio',
             'volatility', 'mom1', 'mom5', 'mom20']
    return df, fcols
