import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "Factor_Attribution_csvs")

# Underlying tickers you have CSVs for
factor_tickers = [
    "SPY", "TLT", "HYG", "DBC", "EEM", "UUP", "TIP",
    "SVXY", "SHY", "CWY", "USMV", "MTUM", "QUAL", "IVE", "IWM", "ACWI",
    "GLD", "USO", "VIXY"
]

# Map tickers to factor names
rename_map = {
    "SPY":  "Equity",
    "TLT":  "Interest Rates",
    "HYG":  "Credit",
    "DBC":  "Commodities",
    "EEM":  "Emerging Markets",
    "UUP":  "FX",
    "TIP":  "Real Yields",
    "SVXY": "Equity Short Vol",
    "CWY":  "FX Carry",
    "USMV": "Low Risk",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "IVE":  "Value",
    "IWM":  "Small Cap",
    "ACWI": "Global Equity",
    "GLD":  "Gold",
    "USO":  "Oil",
    "VIXY": "Volatility",
}

# Final list of factors to use in regressions
factor_cols = [
    "Equity",
    "Interest Rates",
    "Credit",
    "Commodities",
    "Emerging Markets",
    "FX",
    "Real Yields",
    "Local Inflation",
    "Global Equity",
    "Local Equity",
    "Equity Short Vol",
    "FI Carry",
    "FX Carry",
    "Trend",
    "Low Risk",
    "Momentum",
    "Quality",
    "Value",
    "Small Cap",
    "Gold",
    "Oil",
    "Volatility",
]


# =========================
# DATA HELPERS
# =========================

def load_prices_from_csv(ticker: str) -> pd.DataFrame:
    """
    Load a price CSV with columns that include Date and Close or Adj Close.
    Index is set to Date.
    """
    path = os.path.join(CSV_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        st.error(f"No price CSV for {ticker} at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def load_yield_from_csv(ticker: str) -> pd.Series:
    """
    Load yields (for example TLT_Yield.csv, SHY_Yield.csv) where the CSV has
    one of the columns ['Yield', 'Adj Close', 'Close'].
    Values are assumed to be in percent (for example 4.12) and converted to decimals.
    """
    path = os.path.join(CSV_DIR, f"{ticker}_Yield.csv")
    if not os.path.exists(path):
        st.error(f"No yield CSV for {ticker} at {path}")
        return pd.Series(dtype=float)

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    for col in ("Yield", "Adj Close", "Close"):
        if col in df.columns:
            return (df[col] / 100.0).rename(ticker)

    st.error(f"No valid yield column in {path}; found: {df.columns.tolist()}")
    return pd.Series(dtype=float)


def download_prices(tickers) -> pd.DataFrame:
    """
    Load close or adjusted close for each ticker and combine into one prices DataFrame.
    """
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


def prepare_factors() -> pd.DataFrame:
    """
    Build the factor returns dataframe:
    - Monthly returns of underlying ETFs
    - Local inflation: TIP - TLT
    - Global equity: ACWI
    - Local equity: SPY minus ACWI
    - FI carry: (TLT yield - SHY yield) converted to a monthly series
    - Trend: 12 month SPY return
    """
    price_df = download_prices(factor_tickers)
    if price_df.empty:
        return pd.DataFrame()

    # Resample to month start, using last price of each month
    price_df = price_df.resample("MS").last()

    # Cut off current incomplete month
    cutoff = pd.Timestamp.today().normalize().replace(day=1)
    price_df = price_df[price_df.index < cutoff]

    raw_rets = price_df.pct_change().dropna()

    # Base factor set from simple ETF returns
    f = raw_rets.rename(columns=rename_map)

    # Global equity from ACWI
    if "ACWI" in raw_rets.columns:
        f["Global Equity"] = raw_rets["ACWI"]

    # Local equity as SPY minus ACWI
    if "Equity" in f.columns and "ACWI" in raw_rets.columns:
        f["Local Equity"] = f["Equity"] - raw_rets["ACWI"]

    # Local inflation: TIP - TLT returns
    if "TIP" in raw_rets.columns and "TLT" in raw_rets.columns:
        tip, tlt = raw_rets["TIP"].align(raw_rets["TLT"], join="inner")
        f.loc[tip.index, "Local Inflation"] = tip - tlt
    else:
        f["Local Inflation"] = np.nan

    # FI carry: yield spread TLT - SHY, converted to monthly carry (divide by 12)
    tlt_y = load_yield_from_csv("TLT").resample("MS").last()
    shy_y = load_yield_from_csv("SHY").resample("MS").last()
    fi_carry = (tlt_y - shy_y) / 12.0
    f["FI Carry"] = fi_carry.reindex(f.index)

    # Trend: 12 month price change of SPY
    if "SPY" in price_df.columns:
        f["Trend"] = price_df["SPY"].pct_change(12)

    # Keep only the factors we care about and that exist
    keep = [c for c in factor_cols if c in f.columns]
    return f[keep]


def get_rf(index: pd.Index) -> pd.Series:
    """
    Load ^IRX (13 week T bill) as annualized percent and convert to monthly decimal rate.
    IRX ~ annualized percent yield, for example 5.20 means 5.2 percent.
    Monthly rf = (IRX / 100) / 12
    """
    df = load_prices_from_csv("^IRX")
    if df.empty:
        # Just zero if you do not have IRX
        return pd.Series(0.0, index=index, name="RF")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    rf = (df[col] / 100.0 / 12.0).resample("MS").last()
    return rf.reindex(index, method="ffill").fillna(0.0).rename("RF")


def load_and_merge_all_data(fund_tickers):
    """
    Build a combined dataframe with:
    - fund returns
    - factor returns
    - excess returns for each fund
    """
    factors = prepare_factors()
    if factors.empty:
        return None

    rf = get_rf(factors.index)

    fund_prices = download_prices(fund_tickers)
    if fund_prices.empty:
        return None

    fund_prices = fund_prices.resample("MS").last()
    cutoff = pd.Timestamp.today().normalize().replace(day=1)
    fund_prices = fund_prices[fund_prices.index < cutoff]
    fund_rets = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None

    # Outer join then ffill to avoid throwing away early history
    df = fund_rets.join(factors, how="outer").ffill().dropna()

    # Align rf to the combined index
    rf_aligned = rf.reindex(df.index, method="ffill").astype(float)

    # Excess returns for each fund
    for fund in fund_rets.columns:
        df[f"{fund}_Excess"] = df[fund] - rf_aligned

    return df


# =========================
# FACTOR REGRESSIONS
# =========================

def compute_static(df: pd.DataFrame, fund: str):
    """
    Full sample OLS of fund excess returns on the factor set.
    Returns betas, t stats, and R squared.
    """
    cols = [c for c in factor_cols if c in df.columns]
    if not cols:
        return None, None, None

    X = df[cols]
    y = df[f"{fund}_Excess"]

    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()

    betas = model.params[1:]    # drop intercept
    tvals = model.tvalues[1:]
    r2 = model.rsquared

    return betas.round(3), tvals.round(2), r2


def compute_rolling(df: pd.DataFrame, fund: str, window: int = 36) -> pd.DataFrame:
    """
    Rolling OLS using least squares on window-length slices.
    Returns a DataFrame of betas over time.
    """
    cols = [c for c in factor_cols if c in df.columns]
    df_fund = df[[f"{fund}_Excess"] + cols].dropna()

    if len(df_fund) < window:
        return pd.DataFrame()

    y = df_fund[f"{fund}_Excess"].values
    X = df_fund[cols].values

    intercept = np.ones((len(X), 1))
    X_full = np.hstack([intercept, X])

    betas = []

    for i in range(window - 1, len(X_full)):
        X_win = X_full[i - window + 1:i + 1]
        y_win = y[i - window + 1:i + 1]

        coef, *_ = np.linalg.lstsq(X_win, y_win, rcond=None)
        betas.append(coef[1:])   # drop intercept

    idx = df_fund.index[window - 1:]
    return pd.DataFrame(betas, index=idx, columns=cols)

def plot_rolling_betas_plotly(rolling: pd.DataFrame, top_n: int = 5):
    """
    Safely pick the top_n most time-varying factors and plot them with Plotly,
    with correct hover formatting and guaranteed no KeyError from melt.
    """
    if rolling.empty:
        return None

    # --- Clean columns (remove NaN std or empty series) ---
    valid_cols = rolling.columns[rolling.std().notna()]
    if valid_cols.empty:
        return None

    # Number of factors to use
    n = min(top_n, len(valid_cols))

    # Most variable factors
    top_factors = rolling[valid_cols].std().nlargest(n).index.tolist()

    # --- Build tidy dataframe ---
    df_reset = rolling[top_factors].reset_index()

    # Force the index column to be called "index" no matter what
    index_col_name = rolling.index.name or "index"
    df_reset = df_reset.rename(columns={index_col_name: "index"})

    # Melt into long form
    dfm = df_reset.melt(
        id_vars="index",
        var_name="Factor",
        value_name="Beta"
    )

    # --- Plotly figure ---
    fig = px.line(
        dfm,
        x="index",
        y="Beta",
        color="Factor",
        title=f"Rolling Betas: Top {n} Most Variable Factors"
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Date: %{x|%b %Y}<br>"
            "Beta: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")

    return fig

def plot_rolling_level_view(rolling: pd.DataFrame):
    """
    Clean Plotly version of the full rolling betas (level view),
    with rounded hover.
    """
    if rolling.empty:
        return None

    # Long-form tidy dataframe
    dfm = (
        rolling.reset_index()
               .rename(columns={rolling.index.name or "index": "index"})
               .melt(id_vars="index", var_name="Factor", value_name="Beta")
    )

    fig = px.line(
        dfm,
        x="index",
        y="Beta",
        color="Factor",
        title="36 month rolling betas - level view"
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Date: %{x|%b %Y}<br>"
            "Beta: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_dark",  # match your theme
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")

    return fig



# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Factor Attribution Dashboard", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center;">Factor Attribution Dashboard</h1>
    <p style="text-align:center; font-size:16px;">
        Static and rolling multi factor regressions on ETF and index returns<br>
        App created by Dylan Sturdevant
    </p>
    """,
    unsafe_allow_html=True,
)

fund_ticker = st.text_input(
    "Fund ticker",
    value="",
    placeholder="SPY, EFA, AGG, etc."
)

window = st.slider(
    "Rolling window (months)",
    min_value=12,
    max_value=60,
    value=36,
    step=6
)

top_n = st.slider(
    "Top N betas to plot (Plotly)",
    min_value=2,
    max_value=10,
    value=5
)


run = st.button("Run analysis")

if run:
    if not fund_ticker:
        st.error("Please enter a fund ticker.")
    else:
        with st.spinner("Loading and analyzing data..."):
            df = load_and_merge_all_data([fund_ticker])

        if df is None or df.empty:
            st.error(f"No usable data for ticker {fund_ticker}. Check CSVs and paths.")
        else:
            st.success(
                f"Data loaded for {fund_ticker}. Sample from {df.index.min().date()} "
                f"to {df.index.max().date()}."
            )

            # ---------- Static exposures ----------
            betas, tvals, r2 = compute_static(df, fund_ticker)
            if betas is None:
                st.error("No overlapping factors found for regression.")
            else:
                st.subheader(f"Static factor exposures (full sample, R² = {r2:.2f})")

                static_table = pd.DataFrame(
                    {
                        "beta": betas,
                        "t stat": tvals,
                    }
                )
                static_table = static_table.sort_values("beta", key=np.abs, ascending=False)

                st.dataframe(
                    static_table.style.format("{:,.3f}"),
                    use_container_width=True,
                )

            # ---------- Rolling exposures ----------
            rolling = compute_rolling(df, fund_ticker, window=window)

            if rolling.empty:
                st.warning(
                    f"Not enough history for a {window}-month rolling window for {fund_ticker}."
                )
            else:
                st.subheader(f"{window} month rolling betas - level view")
                fig_level = plot_rolling_level_view(rolling)
                st.plotly_chart(fig_level, use_container_width=True)


                st.subheader("Current (last month) betas")
                last_row = rolling.iloc[-1].sort_values(key=np.abs, ascending=False)
                st.dataframe(
                    last_row.to_frame("beta").style.format("{:,.3f}"),
                    use_container_width=True,
                )

                fig = plot_rolling_betas_plotly(rolling, top_n=top_n)
                if fig is not None:
                    st.subheader("Historical rolling betas (Plotly)")
                    st.plotly_chart(fig, use_container_width=True)
