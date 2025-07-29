import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ─── CONFIG ─────────────────────
BASE_DIR = os.path.dirname(__file__)
CSV_DIR  = os.path.join(BASE_DIR, "Factor_Attribution_csvs")

factor_tickers = [
    'SPY','TLT','HYG','DBC','EEM','UUP','TIP',
    'SVXY','SHY','CWY','USMV','MTUM','QUAL','IVE','IWM','ACWI',
    'GLD','USO','VIXY'
]

rename_map = {
    'SPY':'Equity','TLT':'Interest Rates','HYG':'Credit','DBC':'Commodities',
    'EEM':'Emerging Markets','UUP':'FX','TIP':'Real Yields','SVXY':'Equity Short Vol',
    'CWY':'FX Carry','USMV':'Low Risk','MTUM':'Momentum','QUAL':'Quality',
    'IVE':'Value','IWM':'Small Cap','ACWI':'Local Equity Proxy','GLD':'Gold',
    'USO':'Oil','VIXY':'Volatility'
}

# Final list of factor columns, including FI Carry, FX Carry, etc.
factor_cols = [
    'Equity','Interest Rates','Credit','Commodities','Emerging Markets','FX',
    'Real Yields','Local Inflation','Global Equity','Equity Short Vol','FI Carry',
    'FX Carry','Trend','Low Risk','Momentum','Quality','Value','Small Cap',
    'Gold','Oil','Volatility'
]

# ─── HELPERS ─────────────────────

def load_prices_from_csv(ticker):
    path = os.path.join(CSV_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        st.error(f"No price CSV for {ticker} at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df

def load_yield_from_csv(ticker):
    """
    Load yields (e.g. TLT_Yield.csv, SHY_Yield.csv) where the CSV has
    columns ['Date','Close'] or ['Date','Adj Close'] or ['Date','Yield'].
    Converts 4.123 → 0.04123.
    """
    path = os.path.join(CSV_DIR, f"{ticker}_Yield.csv")
    if not os.path.exists(path):
        st.error(f"No yield CSV for {ticker} at {path}")
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    for col in ("Yield","Adj Close","Close"):
        if col in df.columns:
            return (df[col] / 100.0).rename(ticker)
    st.error(f"No valid yield column in {path}; found: {df.columns.tolist()}")
    return pd.Series(dtype=float)

def download_prices(tickers):
    dfs = []
    for t in tickers:
        df = load_prices_from_csv(t)
        if df.empty:
            continue
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        frame = df[[col]].rename(columns={col: t})
        dfs.append(frame)
    if not dfs:
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    return prices

def prepare_factors():
    # 1) Load price returns
    price_df = download_prices(factor_tickers)
    if price_df.empty:
        return pd.DataFrame()

    # 2) Resample to month start, take last price, compute returns
    price_df = price_df.resample('MS').last()
    cutoff   = pd.Timestamp.today().normalize().replace(day=1)
    price_df = price_df[price_df.index < cutoff]
    raw_rets = price_df.pct_change().dropna()

    f = raw_rets.rename(columns=rename_map)

    # 3) Local Equity (Equity minus ACWI)
    if 'Equity' in f.columns and 'ACWI' in raw_rets.columns:
        eq, acwi = f['Equity'].align(raw_rets['ACWI'], join='inner')
        f.loc[eq.index, 'Global Equity'] = acwi
    else:
        f['Local Equity'] = pd.NA

    # 4) Local Inflation (TIP minus TLT returns)
    if 'TIP' in raw_rets.columns and 'TLT' in raw_rets.columns:
        tip, tlt = raw_rets['TIP'].align(raw_rets['TLT'], join='inner')
        f.loc[tip.index, 'Local Inflation'] = tip - tlt
    else:
        f['Local Inflation'] = pd.NA

    # 5) FI Carry = monthly yield spread TLT_Yield - SHY_Yield
    tlt_month = load_yield_from_csv("TLT").resample('MS').last()
    shy_month = load_yield_from_csv("SHY").resample('MS').last()
    fi_carry  = (tlt_month - shy_month).reindex(f.index)
    f['FI Carry'] = fi_carry

    # 6) Trend = 12‑month return of SPY
    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA

    # 7) Keep only the factors in factor_cols (and in f.columns)
    return f[[c for c in factor_cols if c in f.columns]]

def get_rf(index):
    # Load ^IRX CSV (risk‑free rate), monthly align to index
    df = load_prices_from_csv("^IRX")
    if df.empty:
        return pd.Series(0.0, index=index, name='RF')
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    rf = (df[col] / 1200.0).resample('MS').last()
    return rf.reindex(index, method='ffill').fillna(0.0)

def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    if factors.empty:
        return None

    rf = get_rf(factors.index)

    fund_prices = download_prices(fund_tickers)
    if fund_prices.empty:
        return None
    fund_prices = fund_prices.resample('MS').last()
    cutoff      = pd.Timestamp.today().normalize().replace(day=1)
    fund_prices = fund_prices[fund_prices.index < cutoff]
    fund_rets   = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None

    df = fund_rets.join(factors, how='outer').ffill().dropna()
    rf_aligned = rf.reindex(df.index, method='ffill').astype(float)
    for fund in fund_rets.columns:
        df[f'{fund}_Excess'] = df[fund] - rf_aligned

    return df

def compute_static(df, fund):
    cols = [c for c in factor_cols if c in df.columns]
    X    = df[cols].values
    y    = df[f'{fund}_Excess'].values
    X_   = np.column_stack([np.ones(len(X)), X])
    coef,*_ = np.linalg.lstsq(X_, y, rcond=None)
    return pd.Series(coef[1:], index=cols).round(3)

def compute_rolling(df, fund, window=36):
    cols    = [c for c in factor_cols if c in df.columns]
    df_fund = df[[f'{fund}_Excess'] + cols].dropna()
    if len(df_fund) < window:
        return pd.DataFrame()
    betas, dates = [], []
    for i in range(window-1, len(df_fund)):
        y_win = df_fund[f'{fund}_Excess'].iloc[i-window+1:i+1].values
        X_win = df_fund[cols].iloc[i-window+1:i+1].values
        X_    = np.column_stack([np.ones(len(X_win)), X_win])
        coef,*_ = np.linalg.lstsq(X_, y_win, rcond=None)
        betas.append(coef[1:])
        dates.append(df_fund.index[i])
    return pd.DataFrame(betas, index=dates, columns=cols)

def plot_rolling_betas_plotly(rolling, top_n=5):
    if rolling.empty:
        return None
    top = rolling.std().nlargest(top_n).index
    dfm = rolling[top].reset_index().melt(
        id_vars='index', var_name='Factor', value_name='Beta'
    )
    fig = px.line(dfm, x='index', y='Beta', color='Factor',
                  title=f"Rolling Betas: Top {top_n} Most Variable Factors")
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig

# ─── STREAMLIT UI ──────────────────────────

st.title('Factor Attribution Dashboard')
st.markdown("""
Analyze static and rolling multi‑factor betas for various Funds/ETFs/Indexes<br>
App Created by Dylan Sturdevant
""")

fund_ticker = st.text_input('Fund ticker (e.g. SPY)', value='')
window      = st.slider('Rolling window (months)', 12, 60, 36, 6)
top_n       = st.slider('Max betas to plot (plotly)', 2, 10, 5)

if st.button('Run Analysis'):
    if not fund_ticker:
        st.error('Please enter a fund ticker.')
    else:
        with st.spinner('Loading and analyzing data...'):
            df = load_and_merge_all_data([fund_ticker])

        if df is None or df.empty:
            st.error(f'No usable data for ticker {fund_ticker}.')
        else:
            static = compute_static(df, fund_ticker)
            st.subheader('Static Exposures (Full Sample)')
            st.table(static.to_frame(name='β'))

            rolling = compute_rolling(df, fund_ticker, window=window)
            if not rolling.empty:
                st.subheader(f'{window}-Month Rolling Betas')
                st.line_chart(rolling)
                st.subheader('Current (Last-Month) Betas')
                st.write(rolling.iloc[-1].round(3))

                fig = plot_rolling_betas_plotly(rolling, top_n=top_n)
                if fig:
                    st.subheader('Historical Rolling Betas (Plotly)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Not enough history for a {window}-month window.")

      
