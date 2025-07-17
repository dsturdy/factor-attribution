import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START = '1990-01-01'

factor_tickers = [
    'SPY','TLT','HYG','DBC','EEM','UUP','TIP',
    'SVXY','SHY','CWY','USMV','MTUM','QUAL','IVE','IWM','ACWI',
    'GLD','USO','VIXY'
]

rename_map = {
    'SPY': 'Equity',
    'TLT': 'Interest Rates',
    'HYG': 'Credit',
    'DBC': 'Commodities',
    'EEM': 'Emerging Markets',
    'UUP': 'FX',
    'TIP': 'Real Yields',
    'SVXY': 'Equity Short Vol',
    'CWY': 'FX Carry',
    'USMV': 'Low Risk',
    'MTUM': 'Momentum',
    'QUAL': 'Quality',
    'IVE': 'Value',
    'IWM': 'Small Cap',
    'GLD': 'Gold',
    'USO': 'Oil',
    'VIXY': 'Volatility'
}

factor_cols = [
    'Equity','Interest Rates','Credit','Commodities',
    'Emerging Markets','FX','Real Yields','Local Inflation','Local Equity',
    'Equity Short Vol','FI Carry','FX Carry','Trend',
    'Low Risk','Momentum','Quality','Value','Small Cap',
    'Gold','Oil','Volatility'
]

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def download_prices(tickers):
    dfs = []
    for t in tickers:
        try:
            df = yf.download(t, start=START, auto_adjust=False, progress=False)
            if df.empty:
                print(f'Warning: No data for {t}')
                continue
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            frame = df[[col]].rename(columns={col: t})
            dfs.append(frame)
        except Exception as e:
            print(f'Error downloading {t}: {e}')
    if not dfs:
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns.name = None
    return prices


def prepare_factors():
    price_df = download_prices(factor_tickers).resample('MS').last()
    today = pd.Timestamp.today().normalize()
    price_df = price_df[price_df.index < today.replace(day=1)]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)
    if 'ACWI' in raw_rets.columns:
        f['Local Equity'] = f.get('Equity', 0) - raw_rets['ACWI']
    f['Local Inflation'] = raw_rets.get('TIP', 0) - raw_rets.get('TLT', 0)
    f['FI Carry'] = raw_rets.get('TLT', 0) - raw_rets.get('SHY', 0)
    f['Trend'] = price_df.get('SPY', pd.Series()).pct_change(12)
    available = [c for c in factor_cols if c in f.columns]
    return f[available]


def get_rf(index):
    try:
        rf_raw = yf.download('^IRX', start=START, progress=False)['Close']
        rf = rf_raw / 1200
        rf = rf.reindex(index, method='ffill').fillna(0)
    except:
        rf = pd.Series(0.0, index=index, name='RF')
    return rf


def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    rf = get_rf(factors.index)
    fund_prices = download_prices(fund_tickers).resample('MS').last()
    today = pd.Timestamp.today().normalize()
    fund_prices = fund_prices[fund_prices.index < today.replace(day=1)]
    fund_rets = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None
    df = fund_rets.join(factors, how='outer').ffill().dropna()
    rf_aligned = rf.reindex(df.index, method='ffill').astype(float)
    for fund in fund_rets.columns:
        df[f'{fund}_Excess'] = df[fund] - rf_aligned
    return df


def compute_static(df, fund):
    X = sm.add_constant(df[factor_cols])
    y = df[f'{fund}_Excess']
    m = sm.OLS(y, X).fit()
    return m.params.round(3)


def compute_rolling(df, fund, window=36):
    cols = [f'{fund}_Excess'] + factor_cols
    df_fund = df[cols].dropna()
    betas, dates = [], []
    for i in range(window - 1, len(df_fund)):
        y_win = df_fund[f'{fund}_Excess'].iloc[i-window+1:i+1]
        X_win = sm.add_constant(df_fund[factor_cols].iloc[i-window+1:i+1])
        m = sm.OLS(y_win, X_win).fit()
        betas.append(m.params.values)
        dates.append(df_fund.index[i])
    cols_out = ['const'] + factor_cols
    roll = pd.DataFrame(betas, index=dates, columns=cols_out)
    return roll.drop(columns=['const'])

# ─── STREAMLIT UI ───────────────────────────────────────────────────────────────
st.title('Multi‑Factor Exposures Dashboard')

fund_ticker = st.text_input('Fund ticker (e.g. SGIIX)', value='SGIIX')
window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)

if st.button('Run Analysis'):
    if not fund_ticker:
        st.error('Please enter a fund ticker.')
    else:
        df = load_and_merge_all_data([fund_ticker])
        if df is None or df.empty:
            st.error(f'No data found for ticker {fund_ticker}.')
        else:
            static = compute_static(df, fund_ticker)
            st.subheader('Static Exposures')
            st.table(static.to_frame(name='β'))

            rolling = compute_rolling(df, fund_ticker, window=window)
            st.subheader(f'{window}-Month Rolling Betas')
            st.line_chart(rolling)

            latest = rolling.iloc[-1].round(3)
            st.subheader('Current (Last-Month) Betas')
            st.write(latest)
