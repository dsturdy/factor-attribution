# To launch this Streamlit app:
#    streamlit run factor_attribution_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import matplotlib.pyplot as plt

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

# ─── CACHING HELPERS ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=24*3600)
def download_prices(tickers):
    """
    Download price series for given tickers. Cached for 24h by Streamlit to avoid yfinance rate limits.
    """
    dfs = []
    for t in tickers:
        try:
            df = yf.download(t, start=START, auto_adjust=False, progress=False)
            if df.empty:
                df = yf.download(t, period='max', auto_adjust=False, progress=False)
            if df.empty:
                continue
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            dfs.append(df[[col]].rename(columns={col: t}))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1).loc[:, ~pd.concat(dfs, axis=1).columns.duplicated()]
    return prices

@st.cache_data(ttl=24*3600)
def get_rf(index):
    """
    Fetch and align 3M T-bill rate series, cached for 24h to reduce API hits.
    """
    try:
        rf_raw = yf.download('^IRX', start=START, progress=False)['Close']
        rf = rf_raw / 1200
        rf = rf.reindex(index, method='ffill').fillna(0)
    except Exception:
        rf = pd.Series(0.0, index=index, name='RF')
    return rf

# ─── FACTOR PREP & MERGE ─────────────────────────────────────────────────────────
def prepare_factors():
    price_df = download_prices(factor_tickers).resample('MS').last()
    today = pd.Timestamp.today().normalize()
    price_df = price_df[price_df.index < today.replace(day=1)]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)

    if 'Equity' in f.columns and 'ACWI' in raw_rets.columns:
        eq, acwi = f['Equity'].align(raw_rets['ACWI'], join='inner')
        f.loc[eq.index, 'Local Equity'] = eq - acwi

    if 'TIP' in raw_rets.columns and 'TLT' in raw_rets.columns:
        tip, tlt = raw_rets['TIP'].align(raw_rets['TLT'], join='inner')
        f.loc[tip.index, 'Local Inflation'] = tip - tlt

    if 'TLT' in raw_rets.columns and 'SHY' in raw_rets.columns:
        tlt, shy = raw_rets['TLT'].align(raw_rets['SHY'], join='inner')
        f.loc[tlt.index, 'FI Carry'] = tlt - shy

    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)

    available = [c for c in factor_cols if c in f.columns]
    return f[available]


def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = download_prices(fund_tickers).resample('MS').last()
    today = pd.Timestamp.today().normalize()
    fund_prices = fund_prices[fund_prices.index < today.replace(day=1)]
    fund_rets = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None
    df = fund_rets.join(factors, how='outer').ffill().dropna()
    for fund in fund_rets.columns:
        df[f'{fund}_Excess'] = df[fund] - rf.reindex(df.index, method='ffill')
    return df

# --- OLS with numpy (avoiding scipy/statsmodels) ---
def compute_static(df, fund):
    cols = [c for c in factor_cols if c in df.columns]
    X = df[cols].values
    y = df[f'{fund}_Excess'].values
    X_ = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(X_, y, rcond=None)
    return pd.Series(coef[1:], index=cols).round(3)


def compute_rolling(df, fund, window=36):
    cols = [c for c in factor_cols if c in df.columns]
    df_fund = df[[f'{fund}_Excess'] + cols].dropna()
    if len(df_fund) < window:
        return pd.DataFrame()
    betas, dates = [], []
    for i in range(window - 1, len(df_fund)):
        y_win = df_fund[f'{fund}_Excess'].iloc[i-window+1:i+1].values
        X_win = df_fund[cols].iloc[i-window+1:i+1].values
        X_win_ = np.column_stack([np.ones(X_win.shape[0]), X_win])
        coef, *_ = np.linalg.lstsq(X_win_, y_win, rcond=None)
        betas.append(coef[1:])
        dates.append(df_fund.index[i])
    return pd.DataFrame(betas, index=dates, columns=cols)


def plot_rolling_betas(rolling, top_n=5):
    if rolling.empty:
        return None
    stddevs = rolling.std().sort_values(ascending=False)
    top = stddevs.head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in top:
        ax.plot(rolling.index, rolling[col], label=col)
    ax.set_title(f'Rolling Betas: Top {top_n} Most Variable Factors')
    ax.set_ylabel('Beta')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ─── STREAMLIT UI ───────────────────────────────────────────────────────────────
st.title('Multi‑Factor Exposures Dashboard')

st.markdown("""
Analyze rolling and static multi-factor exposures for any ticker.
""")

fund_ticker = st.text_input('Fund ticker (e.g. SGIIX)', value='SGIIX')
window = st.slider('Rolling window (months)', 12, 60, 36, 6)
top_n = st.slider('Max betas to plot', 2, 10, 5)

if st.button('Run Analysis'):
    df = load_and_merge_all_data([fund_ticker])
    if df is None or df.empty:
        st.error(f'No data could be downloaded for ticker {fund_ticker}.')
    else:
        static = compute_static(df, fund_ticker)
        st.subheader('Static Exposures (full-sample)')
        st.table(static.to_frame(name='β'))

        rolling = compute_rolling(df, fund_ticker, window=window)
        if not rolling.empty:
            st.subheader(f'{window}-Month Rolling Betas (Streamlit)')
            st.line_chart(rolling)
            latest = rolling.iloc[-1].round(3)
            st.subheader('Current (Last-Month) Betas')
            st.write(latest)
            fig = plot_rolling_betas(rolling, top_n=top_n)
            if fig:
                st.subheader('Historical Rolling Betas (Matplotlib)')
                st.pyplot(fig)
        else:
            st.warning(f'Not enough data for a {window}-month window.')
