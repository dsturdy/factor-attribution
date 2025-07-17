import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── CONFIG ─────────────────────
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

# ─── HELPERS ─────────────────────

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
    print(prices)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns.name = None
    return prices

def prepare_factors():
    price_df = download_prices(factor_tickers).resample('MS').last()
    today = pd.Timestamp.today().normalize()
    price_df = price_df[price_df.index < today.replace(day=1)]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)
    
    # Local Equity
    if ('Equity' in f.columns) and ('ACWI' in raw_rets.columns):
        eq, acwi = f['Equity'].align(raw_rets['ACWI'], join='inner')
        f.loc[eq.index, 'Local Equity'] = eq - acwi
    else:
        f['Local Equity'] = pd.NA

    # Local Inflation
    if ('TIP' in raw_rets.columns) and ('TLT' in raw_rets.columns):
        tip, tlt = raw_rets['TIP'].align(raw_rets['TLT'], join='inner')
        f.loc[tip.index, 'Local Inflation'] = tip - tlt
    else:
        f['Local Inflation'] = pd.NA

    # FI Carry
    if ('TLT' in raw_rets.columns) and ('SHY' in raw_rets.columns):
        tlt, shy = raw_rets['TLT'].align(raw_rets['SHY'], join='inner')
        f.loc[tlt.index, 'FI Carry'] = tlt - shy
    else:
        f['FI Carry'] = pd.NA

    # Trend (12M change in SPY)
    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA

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
    rf_aligned = rf.reindex(df.index, method='ffill').astype(float)
    for fund in fund_rets.columns:
        df[f'{fund}_Excess'] = df[fund] - rf_aligned
    return df

def compute_static(df, fund):
    X = sm.add_constant(df[[col for col in factor_cols if col in df.columns]])
    y = df[f'{fund}_Excess']
    m = sm.OLS(y, X).fit()
    return m.params.round(3)

def compute_rolling(df, fund, window=36):
    cols = [f'{fund}_Excess'] + [col for col in factor_cols if col in df.columns]
    df_fund = df[cols].dropna()
    betas, dates = [], []
    these_factors = [col for col in factor_cols if col in df_fund.columns]
    for i in range(window - 1, len(df_fund)):
        y_win = df_fund[f'{fund}_Excess'].iloc[i-window+1:i+1]
        X_win = sm.add_constant(df_fund[these_factors].iloc[i-window+1:i+1])
        m = sm.OLS(y_win, X_win).fit()
        betas.append(m.params.values)
        dates.append(df_fund.index[i])
    cols_out = ['const'] + these_factors
    roll = pd.DataFrame(betas, index=dates, columns=cols_out)
    return roll.drop(columns=['const']) if not roll.empty else roll

def plot_rolling_betas(rolling, top_n=5):
    if rolling.empty:
        return None
    # Choose the top N most variable betas (using stddev for significance)
    stddevs = rolling.std().sort_values(ascending=False)
    plot_factors = stddevs.head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    rolling[plot_factors].plot(ax=ax)
    ax.set_title(f"Rolling Betas: Top {top_n} Most Variable Factors")
    ax.set_ylabel("Beta")
    ax.set_xlabel("Date")
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

# ─── STREAMLIT UI ──────────────────────────
st.title('Multi‑Factor Exposures Dashboard')

st.markdown("""
Analyze rolling and static multi-factor exposures for any mutual fund, ETF, or index with a ticker. Enter the fund ticker below, select your rolling window, and click "Run Analysis".
""")

fund_ticker = st.text_input('Fund ticker (e.g. SGIIX)', value='SGIIX')
window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)
top_n = st.slider('Max betas to plot (matplotlib)', 2, 10, 5)

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
                # Matplotlib version:
                fig = plot_rolling_betas(rolling, top_n=top_n)
                if fig:
                    st.subheader('Historical Rolling Betas (Matplotlib)')
                    st.pyplot(fig)
            else:
                st.warning(f"Not enough data for rolling beta calculation with a {window}-month window.")

        st.caption('Note: If a factor data download fails, it will be omitted. Not all funds will have long history.')
