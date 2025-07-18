import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests  # << THE ANTI-RATELIMIT HACK
import warnings
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)

# === Impersonate Chrome for all Yahoo calls ===
session = requests.Session(impersonate="chrome")

# ─── CONFIG ─────────────────────
START = '1990-01-01'

factor_tickers = [
    'SPY', 'TLT', 'HYG', 'DBC', 'EEM', 'UUP', 'TIP',
    'SVXY', 'SHY', 'CWY', 'USMV', 'MTUM', 'QUAL', 'IVE', 'IWM', 'ACWI',
    'GLD', 'USO', 'VIXY'
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
    'Equity', 'Interest Rates', 'Credit', 'Commodities',
    'Emerging Markets', 'FX', 'Real Yields', 'Local Inflation', 'Local Equity',
    'Equity Short Vol', 'FI Carry', 'FX Carry', 'Trend',
    'Low Risk', 'Momentum', 'Quality', 'Value', 'Small Cap',
    'Gold', 'Oil', 'Volatility'
]

# ─── HELPERS ─────────────────────

def download_prices(tickers):
    dfs = []
    for t in tickers:
        try:
            df = yf.download(
                t, start=START, auto_adjust=False, progress=False, session=session
            )
            if df.empty:
                print(f'Warning: No data for {t}')
                st.warning(f"[yfinance] No data for {t}. You may still be rate limited.", icon="⚠️")
                continue
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            frame = df[[col]].rename(columns={col: t})
            dfs.append(frame)
            print(f'[OK] Downloaded {t}, {len(df)} rows.')
        except Exception as e:
            print(f'Error downloading {t}: {e}')
            st.error(f"Error downloading {t}: {e}")
    if not dfs:
        err = "All ticker downloads failed! Are you being rate limited by Yahoo? Try again later."
        print(err)
        st.error(err, icon="🚫")
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns.name = None
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except Exception as e:
            st.error(f"Error converting price DataFrame index to DatetimeIndex: {e}")
            return pd.DataFrame()
    return prices

def prepare_factors():
    st.info("Downloading factor history from yfinance...", icon="🕑")
    price_df = download_prices(factor_tickers)
    if not isinstance(price_df.index, pd.DatetimeIndex):
        try:
            price_df.index = pd.to_datetime(price_df.index)
        except Exception:
            price_df = pd.DataFrame()
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.resample('MS').last()
    today = pd.Timestamp.today().normalize()
    price_df = price_df[price_df.index < today.replace(day=1)]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)
    # Local Equity, etc.
    if ('Equity' in f.columns) and ('ACWI' in raw_rets.columns):
        eq, acwi = f['Equity'].align(raw_rets['ACWI'], join='inner')
        f.loc[eq.index, 'Local Equity'] = eq - acwi
    else:
        f['Local Equity'] = pd.NA
    if ('TIP' in raw_rets.columns) and ('TLT' in raw_rets.columns):
        tip, tlt = raw_rets['TIP'].align(raw_rets['TLT'], join='inner')
        f.loc[tip.index, 'Local Inflation'] = tip - tlt
    else:
        f['Local Inflation'] = pd.NA
    if ('TLT' in raw_rets.columns) and ('SHY' in raw_rets.columns):
        tlt, shy = raw_rets['TLT'].align(raw_rets['SHY'], join='inner')
        f.loc[tlt.index, 'FI Carry'] = tlt - shy
    else:
        f['FI Carry'] = pd.NA
    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA
    available = [c for c in factor_cols if c in f.columns]
    return f[available]

def get_rf(index):
    try:
        rf_raw = yf.download('^IRX', start=START, progress=False, session=session)['Close']
        rf = rf_raw / 1200
        rf = rf.reindex(index, method='ffill').fillna(0)
    except Exception as e:
        st.warning(f"Failed to download or align risk-free rate: {e}")
        rf = pd.Series(0.0, index=index, name='RF')
    return rf

def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = download_prices(fund_tickers)
    if not isinstance(fund_prices.index, pd.DatetimeIndex):
        try:
            fund_prices.index = pd.to_datetime(fund_prices.index)
        except Exception:
            return None
    fund_prices = fund_prices.resample('MS').last()
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
    cols = [c for c in factor_cols if c in df.columns]
    X = df[cols].values
    y = df[f'{fund}_Excess'].values
    X_ = np.column_stack([np.ones(X.shape[0]), X])
    coef, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
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
        coef, _, _, _ = np.linalg.lstsq(X_win_, y_win, rcond=None)
        betas.append(coef[1:])  # exclude intercept
        dates.append(df_fund.index[i])
    return pd.DataFrame(betas, index=dates, columns=cols)

def plot_rolling_betas_plotly(rolling, top_n=5):
    if rolling.empty:
        return None
    stddevs = rolling.std().sort_values(ascending=False)
    plot_factors = stddevs.head(top_n).index.tolist()
    df_to_plot = rolling[plot_factors].copy()
    df_to_plot['Date'] = df_to_plot.index
    df_melted = df_to_plot.melt(id_vars="Date", var_name="Factor", value_name="Beta")
    fig = px.line(
        df_melted,
        x="Date",
        y="Beta",
        color="Factor",
        title=f"Rolling Betas: Top {top_n} Most Variable Factors",
        labels={"Beta": "Beta"},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig

# ─── STREAMLIT UI ──────────────────────────
st.title('Multi‑Factor Exposures Dashboard')

st.markdown("""
Analyze rolling and static multi-factor exposures for any mutual fund, ETF, or index with a ticker.
*Now uses Chrome-impersonation to minimize Yahoo Finance rate limiting.*
""")

fund_ticker = st.text_input('Fund ticker (e.g. SGIIX, SPY)', value='SGIIX')
window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)
top_n = st.slider('Max betas to plot (plotly)', 2, 10, 5)

if st.button('Run Analysis'):
    if not fund_ticker:
        st.error('Please enter a fund ticker.')
    else:
        with st.spinner('Downloading and analyzing data...'):
            df = load_and_merge_all_data([fund_ticker])
        if df is None or df.empty or not any(f in df.columns for f in factor_cols):
            st.error(f'No usable return or factor data for ticker {fund_ticker}.')
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
                # Plotly version:
                fig = plot_rolling_betas_plotly(rolling, top_n=top_n)
                if fig:
                    st.subheader('Historical Rolling Betas (Plotly)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Not enough data for rolling beta calculation with a {window}-month window.")

        st.caption('Note: If a factor data download fails, it will be omitted. Not all funds will have long history.')

