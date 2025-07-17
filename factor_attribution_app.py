import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import plotly.express as px
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── CONFIG ─────────────────────
START = '1990-01-01'
ALPHA_VANTAGE_API_KEY = '6FQ98LZHTWIN2CP6'  

factor_tickers = [
    'SPY', 'TLT', 'HYG', 'DBC', 'EEM', 'UUP', 'TIP',
    'SVXY', 'SHY', 'CWY', 'USMV', 'MTUM', 'QUAL', 'IVE', 'IWM', 'ACWI',
    'GLD', 'USO', 'VIXY'
]
rename_map = {
    'SPY': 'Equity',        'TLT': 'Interest Rates',    'HYG': 'Credit',
    'DBC': 'Commodities',   'EEM': 'Emerging Markets',  'UUP': 'FX',
    'TIP': 'Real Yields',   'SVXY': 'Equity Short Vol', 'CWY': 'FX Carry',
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

# ─── HELPERS ─────────────────────

def download_prices_alpha_vantage(tickers):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas', indexing_type='date')
    dfs = []
    for t in tickers:
        try:
            # 'outputsize' can be 'compact' or 'full': full is slower
            data, meta = ts.get_daily_adjusted(symbol=t, outputsize='full')
            if not data.empty:
                price = data[['5. adjusted close']].rename(columns={'5. adjusted close': t})
                dfs.append(price)
            else:
                print(f'No data for {t}')
        except Exception as e:
            print(f'Alpha Vantage error for {t}: {e}')
        time.sleep(2)  # AV limit: 5 requests/minute. Don't decrease!
    if not dfs:
        return pd.DataFrame(index=pd.to_datetime([]))
    df = pd.concat(dfs, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def download_prices_csv(uploaded_files):
    # For user-uploaded price CSV files
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        # Try to parse date column
        for date_col in ('date', 'Date', 'DATE'):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break
        # Guess ticker name from filename or column
        if 'Adj Close' in df.columns:
            col = 'Adj Close'
        elif 'adjusted_close' in df.columns:
            col = 'adjusted_close'
        elif df.columns[-1] != 'Volume':
            col = df.columns[-1]
        else:
            col = df.columns[1]
        ticker = uploaded_file.name.split('.')[0].upper()
        dfs.append(df[[col]].rename(columns={col: ticker}))
    if not dfs:
        return pd.DataFrame(index=pd.to_datetime([]))
    df = pd.concat(dfs, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def prepare_factors(prices_func, csv_files=None):
    price_df = prices_func(factor_tickers) if csv_files is None else download_prices_csv(csv_files)
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
    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA
    available = [c for c in factor_cols if c in f.columns]
    return f[available]

def get_rf(index):
    # For simplicity, use 1-month T-Bill from Alpha Vantage or set RF=0 for demo
    # Alpha Vantage doesn't have treasury data—use 0 as fallback
    return pd.Series(0.0, index=index, name='RF')

def download_fund_prices_alpha_vantage(fund_tickers):
    return download_prices_alpha_vantage(fund_tickers)

def load_and_merge_all_data(fund_tickers, prices_func, fund_csv=None, factors_csv=None):
    factors = prepare_factors(prices_func, factors_csv)
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = prices_func(fund_tickers) if fund_csv is None else download_prices_csv(fund_csv)
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

# ─── STREAMLIT UI ──────────────────────────

st.title('Multi‑Factor Exposures Dashboard (Alpha Vantage Version)')

st.markdown("""
Analyze rolling and static multi-factor exposures for any stock or ETF ticker (Alpha Vantage powered).
For mutual funds (e.g. 'SGIIX'), upload a CSV file downloaded from Yahoo or the fund manager's site.<br>
**Note:** Alpha Vantage does **not** support most mutual fund tickers. For funds, use the CSV uploader below.
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Live Data (stocks & ETFs)", "CSV Upload (funds or custom)"])

with tab1:
    fund_ticker = st.text_input('Stock or ETF ticker (e.g. SPY, AAPL, VOO)', value='SPY')
    window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)
    top_n = st.slider('Max betas to plot (plotly)', 2, 10, 5)

    if st.button('Run Analysis (Live)', key='run_live'):
        with st.spinner('Downloading and analyzing live Alpha Vantage data...'):
            df = load_and_merge_all_data(
                [fund_ticker],
                download_prices_alpha_vantage
            )
        if df is None or df.empty or not any(f in df.columns for f in factor_cols):
            st.error(
                f'No usable return or factor data for ticker {fund_ticker}. \n\n'
                'If a mutual fund, try the "CSV Upload" tab below. For stocks/ETFs, verify symbol or try again in a few moments. '
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
        st.caption('Factor data comes from Alpha Vantage API. Some tickers may not be supported. Try the CSV tab for custom datasets.')

with tab2:
    st.write("**Upload CSV(s) for the fund/asset and optionally for factors.**")
    fund_files = st.file_uploader(
        "Upload CSV(s) for one or more fund price histories (NAV/price, must have a date column).", 
        accept_multiple_files=True,
        key='fund_csv'
    )
    factor_files = st.file_uploader(
        "Upload optional custom CSV(s) for factors (leave blank to use Alpha Vantage factor data).", 
        accept_multiple_files=True,
        key='factor_csv'
    )
    fund_name = st.text_input('Fund ticker or custom name (matches your fund CSV column)', value='SGIIX', key='csv_fund_name')
    window2 = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6, key='csv_window')
    top_n2 = st.slider('Max betas to plot (plotly)', 2, 10, 5, key='csv_topn')

    if st.button('Run Analysis (CSV)', key='run_csv'):
        if not fund_files:
            st.error('Upload at least one CSV file for the fund/asset.')
        else:
            with st.spinner('Analyzing uploaded CSV data...'):
                def csv_factors_func(tickers):
                    return prepare_factors(download_prices_csv, factor_files) if factor_files else prepare_factors(download_prices_alpha_vantage)
                df = load_and_merge_all_data(
                    [fund_name],
                    download_prices_csv,
                    fund_csv=fund_files,
                    factors_csv=factor_files if factor_files else None,
                )
            if df is None or df.empty or not any(f in df.columns for f in factor_cols):
                st.error(f'No usable return or factor data for asset name {fund_name} in uploaded CSV(s).')
            else:
                static = compute_static(df, fund_name)
                st.subheader('Static Exposures (full-sample)')
                st.table(static.to_frame(name='β'))

                rolling = compute_rolling(df, fund_name, window=window2)
                if not rolling.empty:
                    st.subheader(f'{window2}-Month Rolling Betas (Streamlit)')
                    st.line_chart(rolling)
                    latest = rolling.iloc[-1].round(3)
                    st.subheader('Current (Last-Month) Betas')
                    st.write(latest)
                    fig = plot_rolling_betas_plotly(rolling, top_n=top_n2)
                    if fig:
                        st.subheader('Historical Rolling Betas (Plotly)')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data in upload for rolling beta calculation with a {window2}-month window.")
        st.caption('Upload one or more CSVs for custom funds or factors. Format: date,NAV/price/ad close... with a date column.')
