import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

# CONFIGURATION
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"    # <--- PUT YOUR KEY HERE
POLYGON_BASE = "https://api.polygon.io"

START = "1990-01-01"  # Polygon starts ~1992-2000 for most assets

factor_tickers = [
    'SPY', 'TLT', 'HYG', 'DBC', 'EEM', 'UUP', 'TIP',
    'SVXY', 'SHY', 'USMV', 'MTUM', 'QUAL', 'IVE', 'IWM', 'ACWI',
    'GLD', 'USO', 'VIXY'
]
rename_map = {
    'SPY': 'Equity',        'TLT': 'Interest Rates',    'HYG': 'Credit',
    'DBC': 'Commodities',   'EEM': 'Emerging Markets',  'UUP': 'FX',
    'TIP': 'Real Yields',   'SVXY': 'Equity Short Vol',
    'USMV': 'Low Risk',     'MTUM': 'Momentum',         'QUAL': 'Quality',
    'IVE': 'Value',         'IWM': 'Small Cap',         'GLD': 'Gold',
    'USO': 'Oil',           'VIXY': 'Volatility'
}
factor_cols = [
    'Equity', 'Interest Rates', 'Credit', 'Commodities',
    'Emerging Markets', 'FX', 'Real Yields', 'Local Inflation', 'Local Equity',
    'Equity Short Vol', 'FI Carry', 'FX Carry', 'Trend',
    'Low Risk', 'Momentum', 'Quality', 'Value', 'Small Cap',
    'Gold', 'Oil', 'Volatility'
]

# -- POLYGON DATA DOWNLOADER --
def polygon_fetch_daily(ticker, apikey, from_date=START, to_date=None):
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date or pd.Timestamp.today().strftime('%Y-%m-%d')}"
    params = dict(adjusted='true', sort='asc', apiKey=apikey, limit=50000)
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if "results" not in data or not data["results"]:
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.set_index('date')
    df = df[["c"]].rename(columns={'c': ticker})  # c = close price
    return df

def download_prices_polygon(tickers):
    dfs = []
    for t in tickers:
        # For mutual funds, Polygon uses A:SGIIX, etc.
        actual_ticker = t
        if len(t) > 5 and not t.startswith("A:"):
            actual_ticker = "A:"+t.upper()
        try:
            df = polygon_fetch_daily(actual_ticker, POLYGON_API_KEY, from_date=START)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"Polygon: No data for {actual_ticker}")
        except Exception as e:
            print(f"Polygon error {actual_ticker}: {e}")
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, axis=1)
    result.index = pd.to_datetime(result.index)
    result = result[~result.index.duplicated()]
    return result.sort_index()

def prepare_factors():
    price_df = download_prices_polygon(factor_tickers)
    if price_df.empty or not isinstance(price_df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    price_df = price_df.resample('MS').last()
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
    # Trend
    if 'Equity' in f.columns and 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA
    available = [c for c in factor_cols if c in f.columns]
    return f[available]

def get_rf(index):
    # Polygon does not serve T-bill rates; set rf = 0 for simplicity
    return pd.Series(0.0, index=index, name='RF')

def download_fund_prices_polygon(fund_tickers):
    return download_prices_polygon(fund_tickers)

def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = download_fund_prices_polygon(fund_tickers)
    if fund_prices.empty:
        return None
    fund_prices = fund_prices.resample('MS').last()
    today = pd.Timestamp.today().normalize()
    fund_prices = fund_prices[fund_prices.index < today.replace(day=1)]
    fund_rets = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None
    df = fund_rets.join(factors, how='outer').ffill().dropna()
    if df.empty:
        return None
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
        betas.append(coef[1:])
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
        df_melted, x="Date", y="Beta", color="Factor",
        title=f"Rolling Betas: Top {top_n} Most Variable Factors",
        labels={"Beta": "Beta"},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return fig

# STREAMLIT UI
st.title('Multi‑Factor Exposures Dashboard (Polygon API)')

st.markdown("""
Analyze rolling and static multi-factor exposures for **any US stock, ETF, or supported mutual fund (A:SGIIX, etc.)** with a ticker. 
Enter e.g. "SPY" for S&P 500 ETF or "A:SGIIX" for funds. For funds not on Polygon, use the CSV upload tab in the future.
""")

fund_ticker = st.text_input('Fund/ETF/stock ticker (use "A:SGIIX" for some funds)', value='SPY')
window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)
top_n = st.slider('Max betas to plot (plotly)', 2, 10, 5)

if st.button('Run Analysis'):
    if not fund_ticker:
        st.error('Please enter a fund ticker.')
    else:
        with st.spinner('Downloading Polygon.io price data and analyzing...'):
            df = load_and_merge_all_data([fund_ticker])
        if df is None or df.empty or not any(f in df.columns for f in factor_cols):
            st.error(
                f'No usable return or factor data for ticker {fund_ticker} (Polygon.io). \n\n'
                'Try a valid US stock/ETF or Polygon-supported fund (A:SGIIX etc.).'
            )
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
                fig = plot_rolling_betas_plotly(rolling, top_n=top_n)
                if fig:
                    st.subheader('Historical Rolling Betas (Plotly)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Not enough data for rolling beta calculation with a {window}-month window.")
        st.caption('Powered by Polygon.io. If a factor download fails, it is skipped. Mutual funds must use "A:<ticker>" format.')

# END OF APP
