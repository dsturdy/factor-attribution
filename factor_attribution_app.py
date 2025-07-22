import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)

# ─── CONFIG ─────────────────────
START = '1990-01-01'

CSV_DIR = '/Users/dylansturdevant/Desktop/untitled folder 2/Factor_Attribution_csvs'

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

def available_tickers():
    # List all CSV tickers available in folder for user info/validation
    files = os.listdir(CSV_DIR)
    return sorted([f[:-4] for f in files if f.endswith('.csv')])


def download_prices(tickers):
    dfs = []
    for t in tickers:
        csv_path = os.path.join(CSV_DIR, f'{t}.csv')
        if not os.path.exists(csv_path):
            st.warning(f"No data for ticker '{t}' (file not found).", icon="⚠️")
            continue
        try:
            # Load CSV, assuming date is index or first column
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            if col not in df.columns:
                st.warning(f"File {t}.csv has no '{col}' column.", icon="⚠️")
                continue
            frame = df[[col]].rename(columns={col: t})
            dfs.append(frame)
            print(f'[OK] Loaded {t} from CSV, {len(df)} rows.')
        except Exception as e:
            st.error(f"Error loading CSV for {t}: {e}")
    if not dfs:
        st.error('All ticker loads failed or files not found!', icon="🚫")
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.columns.name = None
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except Exception as e:
            st.error(f"Error converting index to datetime: {e}")
            return pd.DataFrame()
    return prices


def prepare_factors():
    st.info("Loading factor history from local CSVs...", icon="🕑")
    price_df = download_prices(factor_tickers)
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.resample('MS').last()
    today = pd.Timestamp.today().normalize()
    price_df = price_df[price_df.index < today.replace(day=1)]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)

    # Calculate Local Equity = Equity - ACWI
    if ('Equity' in f.columns) and ('ACWI' in raw_rets.columns):
        eq, acwi = f['Equity'].align(raw_rets['ACWI'], join='inner')
        f.loc[eq.index, 'Local Equity'] = eq - acwi
    else:
        f['Local Equity'] = pd.NA

    # Local Inflation = TIP - TLT
    if ('TIP' in raw_rets.columns) and ('TLT' in raw_rets.columns):
        tip, tlt = raw_rets['TIP'].align(raw_rets['TLT'], join='inner')
        f.loc[tip.index, 'Local Inflation'] = tip - tlt
    else:
        f['Local Inflation'] = pd.NA

    # FI Carry = TLT - SHY
    if ('TLT' in raw_rets.columns) and ('SHY' in raw_rets.columns):
        tlt, shy = raw_rets['TLT'].align(raw_rets['SHY'], join='inner')
        f.loc[tlt.index, 'FI Carry'] = tlt - shy
    else:
        f['FI Carry'] = pd.NA

    # Trend = 12-month pct_change on SPY prices
    if 'SPY' in price_df.columns:
        f['Trend'] = price_df['SPY'].pct_change(12)
    else:
        f['Trend'] = pd.NA

    available = [c for c in factor_cols if c in f.columns]
    return f[available]


def get_rf(index):
    # Load the risk-free rate from local CSV if exists else fallback to zero
    # You can add a CSV named 'IRX.csv' with daily close prices divided by 1200 in CSV_DIR
    rf_path = os.path.join(CSV_DIR, 'IRX.csv')
    if os.path.exists(rf_path):
        try:
            rf_raw = pd.read_csv(rf_path, index_col=0, parse_dates=True)
            rf_raw = rf_raw['Close']
            rf = rf_raw / 1200  # Approximate monthly rf
            rf = rf.reindex(index, method='ffill').fillna(0)
            return rf
        except Exception as e:
            st.warning(f"Failed to load or process risk-free rate CSV: {e}")
    # If no file found or error, return zeros
    return pd.Series(0.0, index=index, name='RF')


def load_and_merge_all_data(fund_tickers):
    factors = prepare_factors()
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = download_prices(fund_tickers)
    if fund_prices.empty:
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
        y_win = df_fund[f'{fund}_Excess'].iloc[i - window + 1:i + 1].values
        X_win = df_fund[cols].iloc[i - window + 1:i + 1].values
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
*Now works fully from local CSV files in your designated folder.*
""")

all_available = available_tickers()
fund_ticker = st.text_input('Fund ticker (e.g. AAPL, SPY)', value='SPY').upper()
window = st.slider('Rolling window (months)', min_value=12, max_value=60, value=36, step=6)
top_n = st.slider('Max betas to plot (plotly)', 2, 10, 5)

if st.button('Run Analysis'):
    if not fund_ticker:
        st.error('Please enter a fund ticker.')
    elif fund_ticker not in all_available:
        st.error(f"Ticker '{fund_ticker}' not found in local data. Available tickers:\n\n{', '.join(all_available)}")
    else:
        with st.spinner('Loading and analyzing data...'):
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

        st.caption('Note: Ensure all required factor CSV files exist in the CSV folder. Tickers are case-sensitive.')
