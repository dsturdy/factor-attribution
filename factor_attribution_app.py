import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(__file__)
CSV_DIR  = os.path.join(BASE_DIR, "Factor_Attribution_4_csvs")

factor_tickers = [
    "SPY", "ACWI", "TLT", "HYG", "DBC", "EEM", "UUP", "TIP",
    "SVXY", "SHY", "USMV", "MTUM", "QUAL", "IVE", "IWM",
    "GLD", "USO", "VIXY", "^TNX", "^IRX"
]

rename_map = {
    "TLT":  "Interest Rates",
    "HYG":  "Credit",
    "DBC":  "Commodities",
    "EEM":  "Emerging Markets",
    "UUP":  "FX",
    "TIP":  "Real Yields",
    "SVXY": "Equity Short Vol",
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

factor_cols = [
    "Global Equity",
    "Interest Rates",
    "Credit",
    "Commodities",
    "Emerging Markets",
    "FX",
    "Real Yields",
    "Local Inflation",
    "Equity Short Vol",
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
    "FI Carry",
]

PLOTLY_THEME = "plotly_dark"

# =========================
# DATA HELPERS (cached)
# =========================

def load_prices_from_csv(ticker: str) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def download_prices(tickers) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        t = str(t).strip().upper()
        df = load_prices_from_csv(t)
        if df.empty:
            continue
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        dfs.append(df[[col]].rename(columns={col: t}))
    if not dfs:
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    return prices


@st.cache_data
def prepare_factors() -> pd.DataFrame:
    price_df = download_prices(factor_tickers)
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.resample("MS").last()
    cutoff = pd.Timestamp.today().normalize().replace(day=1)
    price_df = price_df[price_df.index < cutoff]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)
    if "EEM" in raw_rets.columns and "UUP" in raw_rets.columns:
        f["FX Carry"] = raw_rets["EEM"] - raw_rets["UUP"]
    else:
        f["FX Carry"] = np.nan

    if "TIP" in raw_rets.columns and "TLT" in raw_rets.columns:
        tip, tlt = raw_rets["TIP"].align(raw_rets["TLT"], join="inner")
        f.loc[tip.index, "Local Inflation"] = tip - tlt
    else:
        f["Local Inflation"] = np.nan

    if "SPY" in price_df.columns:
        f["Trend"] = price_df["SPY"].pct_change(12)

        # FI Carry: 10Y yield minus 3M yield, monthly carry
    tnx = load_prices_from_csv("^TNX")
    irx = load_prices_from_csv("^IRX")
    if not tnx.empty and not irx.empty:
        col_tnx = "Adj Close" if "Adj Close" in tnx.columns else "Close"
        col_irx = "Adj Close" if "Adj Close" in irx.columns else "Close"
        tnx_y = (tnx[col_tnx] / 100.0).resample("MS").last()
        irx_y = (irx[col_irx] / 100.0).resample("MS").last()
        fi_carry = (tnx_y - irx_y).diff()  # change in spread each month
        f["FI Carry"] = fi_carry.reindex(f.index)
    else:
        f["FI Carry"] = np.nan

    keep = [c for c in factor_cols if c in f.columns]
    return f[keep]


def get_rf(index: pd.Index) -> pd.Series:
    df = load_prices_from_csv("^IRX")
    if df.empty:
        return pd.Series(0.0, index=index, name="RF")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    rf = (df[col] / 100.0 / 12.0).resample("MS").last()
    return rf.reindex(index, method="ffill").fillna(0.0).rename("RF")


@st.cache_data
def load_and_merge_all_data(fund_tickers: tuple):
    factors = prepare_factors()
    if factors.empty:
        return None
    rf = get_rf(factors.index)
    fund_prices = download_prices(list(fund_tickers))
    if fund_prices.empty:
        return None
    fund_prices = fund_prices.resample("MS").last()
    cutoff = pd.Timestamp.today().normalize().replace(day=1)
    fund_prices = fund_prices[fund_prices.index < cutoff]
    fund_rets = fund_prices.pct_change().dropna()
    if fund_rets.empty:
        return None
    df = fund_rets.join(factors, how="outer").ffill().dropna()
    rf_aligned = rf.reindex(df.index, method="ffill").astype(float)
    for fund in fund_rets.columns:
        df[f"{fund}_Excess"] = df[fund] - rf_aligned
    return df


# =========================
# FACTOR REGRESSIONS
# =========================

def compute_static(df: pd.DataFrame, fund: str):
    cols = [c for c in factor_cols if c in df.columns]
    if not cols:
        return None, None, None
    X = df[cols]
    y = df[f"{fund}_Excess"]
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()
    betas = model.params[1:]
    tvals = model.tvalues[1:]
    r2 = model.rsquared
    return betas.round(3), tvals.round(2), r2


def compute_rolling(df: pd.DataFrame, fund: str, window: int = 36) -> pd.DataFrame:
    cols = [c for c in factor_cols if c in df.columns]
    df_fund = df[[f"{fund}_Excess"] + cols].dropna()
    if len(df_fund) < window:
        return pd.DataFrame()
    y = df_fund[f"{fund}_Excess"].values
    X = df_fund[cols].values
    X_full = np.hstack([np.ones((len(X), 1)), X])
    betas = []
    for i in range(window - 1, len(X_full)):
        X_win = X_full[i - window + 1:i + 1]
        y_win = y[i - window + 1:i + 1]
        coef, *_ = np.linalg.lstsq(X_win, y_win, rcond=None)
        betas.append(coef[1:])
    idx = df_fund.index[window - 1:]
    return pd.DataFrame(betas, index=idx, columns=cols)


# =========================
# PERFORMANCE METRICS
# =========================

def compute_performance_metrics(df: pd.DataFrame, fund: str, benchmark_ticker: str | None = None) -> pd.DataFrame:
    ret = df[fund]
    rows = []
    horizons = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36}
    for label, months in horizons.items():
        if len(ret) < months:
            continue
        r = ret.iloc[-months:]
        cumret  = (1 + r).prod() - 1
        ann_ret = (1 + cumret) ** (12 / months) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        roll_max = (1 + r).cumprod().cummax()
        drawdown = ((1 + r).cumprod() / roll_max - 1).min()
        row = {
            "Period": label, "Cum Return": cumret, "Ann Return": ann_ret,
            "Ann Vol": ann_vol, "Sharpe": sharpe, "Max Drawdown": drawdown,
        }
        if benchmark_ticker and benchmark_ticker in df.columns:
            b_cum = (1 + df[benchmark_ticker].iloc[-months:]).prod() - 1
            row["Active Return"] = cumret - b_cum
        rows.append(row)
    return pd.DataFrame(rows).set_index("Period")


def compute_return_attribution(df: pd.DataFrame, fund: str, betas: pd.Series) -> pd.DataFrame:
    cols = [c for c in betas.index if c in df.columns]
    factor_ann_ret = df[cols].mean() * 12
    contributions  = (betas[cols] * factor_ann_ret).rename("Contribution")
    pct = (contributions / contributions.abs().sum() * 100).rename("% of Total")
    result = pd.concat([betas[cols].rename("Beta"), contributions, pct], axis=1)
    return result.reindex(result["Contribution"].abs().sort_values(ascending=False).index)


# =========================
# SCORECARD
# =========================

def flag_manager(metrics: pd.DataFrame, active_col: bool) -> tuple[str, str]:
    if "1Y" not in metrics.index:
        return "⚪ Insufficient Data", "Need ≥12 months of history"
    r1y  = metrics.loc["1Y", "Ann Return"]
    sh1y = metrics.loc["1Y", "Sharpe"]
    dd1y = metrics.loc["1Y", "Max Drawdown"]
    if r1y < -0.05 or sh1y < 0:
        return "Under Review", f"1Y Return: {r1y:.1%} | Sharpe: {sh1y:.2f}"
    if r1y < 0.0 or dd1y < -0.15:
        return "On Watch", f"1Y Return: {r1y:.1%} | Max DD: {dd1y:.1%}"
    return "Approved", f"1Y Return: {r1y:.1%} | Sharpe: {sh1y:.2f}"


# =========================
# CHART HELPERS
# =========================

def plot_rolling_heatmap(rolling: pd.DataFrame):
    if rolling.empty:
        return None
    data  = rolling.clip(-3, 3)
    order = data.abs().mean().sort_values(ascending=False).index.tolist()
    data  = data[order]
    x_labels = [d.strftime("%b %Y") for d in data.index]
    fig = go.Figure(go.Heatmap(
        z=data.T.values,
        x=x_labels,
        y=order,
        colorscale=[
            [0.0,  "#b91c1c"],
            [0.35, "#fca5a5"],
            [0.5,  "#1e293b"],
            [0.65, "#86efac"],
            [1.0,  "#15803d"],
        ],
        zmid=0, zmin=-2, zmax=2,
        colorbar=dict(
            title="Beta", thickness=12, len=0.8,
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["≤-2", "-1", "0", "1", "≥2"],
        ),
        hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Beta: %{z:.3f}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        template=PLOTLY_THEME,
        title="Rolling betas — heatmap (winsorised ±3)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickangle=-45, nticks=12, title=""),
        yaxis=dict(title="", autorange="reversed"),
        height=max(350, len(order) * 28 + 80),
    )
    return fig


def plot_rolling_filtered(rolling: pd.DataFrame, selected_factors: list):
    if rolling.empty or not selected_factors:
        return None
    data = rolling[selected_factors].clip(-3, 3)
    dfm = (
        data.reset_index()
            .rename(columns={data.index.name or "index": "index"})
            .melt(id_vars="index", var_name="Factor", value_name="Beta")
    )
    fig = px.line(dfm, x="index", y="Beta", color="Factor",
                  title=f"Rolling betas — {len(selected_factors)} selected factors")
    fig.update_traces(hovertemplate=(
        "<b>%{fullData.name}</b><br>Date: %{x|%b %Y}<br>Beta: %{y:.3f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(template=PLOTLY_THEME,
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")
    return fig


def plot_attribution_bar(attr: pd.DataFrame, fund: str):
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in attr["Contribution"]]
    fig = go.Figure(go.Bar(
        x=attr.index, y=attr["Contribution"], marker_color=colors,
        text=[f"{v:.2%}" for v in attr["Contribution"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Contribution: %{y:.3%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Return Attribution — {fund} (annualised factor contributions)",
        template=PLOTLY_THEME, yaxis_tickformat=".1%",
        margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    return fig


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Factor Attribution Dashboard", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    div[data-testid="metric-container"] {
        background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center; margin-bottom:4px;">FACTOR ATTRIBUTION DASHBOARD</h1>
<p style="text-align:center; color:#888; font-family:'IBM Plex Mono',monospace; font-size:13px; letter-spacing:1px;">
    MULTI-FACTOR REGRESSION  ·  MANAGER OVERSIGHT  ·  RETURN ATTRIBUTION
</p>
<hr style="border-color:#333; margin:16px 0;">
""", unsafe_allow_html=True)

tab_factor, tab_scorecard, tab_attribution = st.tabs([
    "Factor Analysis", "Manager Scorecard", "Return Attribution",
])

# ─────────────────────────────────────────────
# TAB 1: FACTOR ANALYSIS
# ─────────────────────────────────────────────
with tab_factor:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        fund_ticker = st.text_input("Fund ticker", value="", placeholder="SPY, AGG, EFA …").strip().upper()
    with c2:
        window = st.slider("Rolling window (months)", 12, 60, 36, 6)
    with c3:
        top_n = st.slider("Top N variable factors", 2, 10, 5)

    run = st.button("Run Analysis", type="primary")

    if run and fund_ticker:
        with st.spinner("Loading data and running regressions…"):
            df = load_and_merge_all_data((fund_ticker,))

        if df is None or df.empty:
            st.error(f"No usable data for {fund_ticker}. Check your CSVs.")
        else:
            st.success(
                f"Data loaded for **{fund_ticker}** · "
                f"{df.index.min().date()} → {df.index.max().date()} "
                f"({len(df)} months)"
            )

            betas, tvals, r2 = compute_static(df, fund_ticker)

            if betas is None:
                st.error("No overlapping factors found for regression.")
            else:
                st.subheader(f"Static factor exposures — full sample  (R² = {r2:.2f})")
                static_table = pd.DataFrame({"beta": betas, "t stat": tvals})
                static_table = static_table.sort_values("beta", key=np.abs, ascending=False)

                def color_beta(val):
                    if val > 0.1:  return "color:#52b788"
                    if val < -0.1: return "color:#e63946"
                    return "color:#aaa"

                st.dataframe(
                    static_table.style.format("{:,.3f}").map(color_beta, subset=["beta"]),
                    use_container_width=True,
                )

                st.session_state["betas"]       = betas
                st.session_state["fund_ticker"] = fund_ticker
                st.session_state["df"]          = df

            rolling = compute_rolling(df, fund_ticker, window=window)

            if rolling.empty:
                st.warning(f"Not enough history for a {window}-month rolling window.")
            else:
                st.session_state["rolling"] = rolling

                st.subheader(f"{window}-month rolling betas — heatmap")
                fig_heat = plot_rolling_heatmap(rolling)
                if fig_heat:
                    st.plotly_chart(fig_heat, use_container_width=True)

                st.divider()

                st.subheader("Current (last month) betas")
                last = rolling.iloc[-1].sort_values(key=np.abs, ascending=False)
                st.dataframe(last.to_frame("beta").style.format("{:,.3f}"),
                             use_container_width=True)

                st.session_state["all_factors"]     = rolling.columns.tolist()
                st.session_state["default_factors"]  = rolling.std().nlargest(top_n).index.tolist()
                st.session_state["selected_factors"] = st.session_state["default_factors"]

    elif run and not fund_ticker:
        st.error("Please enter a fund ticker.")

    # Multiselect outside if run — persists on rerun
    if "all_factors" in st.session_state:
        st.divider()
        st.subheader("Factor detail — line view")
        selected = st.multiselect(
            "Select factors to display",
            options=st.session_state["all_factors"],
            key="selected_factors",
            help="Winsorised at ±2 to suppress artifacts.",
        )
        if selected and "rolling" in st.session_state:
            fig_line = plot_rolling_filtered(st.session_state["rolling"], selected)
            if fig_line:
                st.plotly_chart(fig_line, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2: MANAGER SCORECARD
# ─────────────────────────────────────────────
with tab_scorecard:
    st.markdown("### Manager Oversight Scorecard")
    st.caption("Flags are rule-based: Under Review · On Watch · Approved")

    col_inp, col_bench = st.columns([3, 1])
    with col_inp:
        raw_input = st.text_input("Tickers (comma-separated)", placeholder="SPY, AGG, EFA, HYG, GLD")
    with col_bench:
        benchmark = st.text_input("Benchmark ticker", value="SPY", placeholder="SPY")

    run_sc = st.button("▶  Generate Scorecard", type="primary")

    if run_sc:
        tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
        if not tickers:
            st.error("Enter at least one ticker.")
        else:
            scorecard_rows = []
            fund_metrics_store = {}
            progress = st.progress(0, text="Loading fund data…")
            all_tickers = list(set(tickers + ([benchmark] if benchmark else [])))

            with st.spinner("Running performance analysis…"):
                df_all = load_and_merge_all_data(tuple(all_tickers))

            if df_all is None:
                st.error("Could not load data. Check your CSVs.")
            else:
                for i, ticker in enumerate(tickers):
                    progress.progress((i + 1) / len(tickers), text=f"Analysing {ticker}…")
                    if ticker not in df_all.columns:
                        scorecard_rows.append({
                            "Ticker": ticker, "Status": "⚪ No Data", "Reason": "CSV not found"
                        })
                        continue
                    metrics = compute_performance_metrics(
                        df_all, ticker,
                        benchmark_ticker=benchmark if benchmark != ticker else None
                    )
                    fund_metrics_store[ticker] = metrics
                    active_col = "Active Return" in metrics.columns
                    status, reason = flag_manager(metrics, active_col)
                    row = {"Ticker": ticker, "Status": status, "Reason": reason}
                    for period in ["1M", "3M", "1Y", "3Y"]:
                        if period in metrics.index:
                            row[f"{period} Return"] = metrics.loc[period, "Ann Return"]
                            row[f"{period} Sharpe"] = metrics.loc[period, "Sharpe"]
                    if "1Y" in metrics.index:
                        row["Max DD (1Y)"] = metrics.loc["1Y", "Max Drawdown"]
                        if active_col:
                            row["Active Ret (1Y)"] = metrics.loc["1Y", "Active Return"]
                    scorecard_rows.append(row)

                progress.empty()
                sc_df = pd.DataFrame(scorecard_rows)

                k1, k2, k3 = st.columns(3)
                k1.metric("Approved",     sc_df["Status"].str.startswith("").sum())
                k2.metric("⚠On Watch",     sc_df["Status"].str.startswith("").sum())
                k3.metric("Under Review", sc_df["Status"].str.startswith("").sum())

                st.divider()

                fmt_cols = {c: "{:.2%}" for c in sc_df.columns if "Return" in c or "DD" in c}
                fmt_cols.update({c: "{:.2f}" for c in sc_df.columns if "Sharpe" in c})
                st.dataframe(sc_df.style.format(fmt_cols, na_rep="—"),
                             use_container_width=True,
                             height=min(400, 40 + 35 * len(sc_df)))

                # Store in session state so drill-down persists
                st.session_state["fund_metrics_store"] = fund_metrics_store

    # Drill-down OUTSIDE if run_sc — persists on rerun
    if "fund_metrics_store" in st.session_state and st.session_state["fund_metrics_store"]:
        st.divider()
        st.markdown("#### Fund Detail")
        drill = st.selectbox(
            "Select fund to drill into",
            options=list(st.session_state["fund_metrics_store"].keys()),
            key="drill_fund",
        )
        if drill:
            st.dataframe(
                st.session_state["fund_metrics_store"][drill].style.format({
                    "Cum Return": "{:.2%}", "Ann Return": "{:.2%}",
                    "Ann Vol": "{:.2%}", "Sharpe": "{:.2f}",
                    "Max Drawdown": "{:.2%}", "Active Return": "{:.2%}",
                }, na_rep="—"),
                use_container_width=True,
            )

# ─────────────────────────────────────────────
# TAB 3: RETURN ATTRIBUTION
# ─────────────────────────────────────────────
with tab_attribution:
    st.markdown("### Return Attribution")
    st.caption("Run Factor Analysis first (Tab 1) to unlock this tab.")

    if "betas" not in st.session_state or "df" not in st.session_state:
        st.info("Run a Factor Analysis first (Tab 1) to unlock attribution.")
    else:
        betas  = st.session_state["betas"]
        df_ra  = st.session_state["df"]
        ticker = st.session_state["fund_ticker"]

        attr = compute_return_attribution(df_ra, ticker, betas)

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(f"**{ticker} — factor contributions (annualised)**")
            st.dataframe(
                attr.style.format({
                    "Beta": "{:.3f}", "Contribution": "{:.2%}", "% of Total": "{:.1f}%",
                }),
                use_container_width=True,
            )
        with col_right:
            st.plotly_chart(plot_attribution_bar(attr, ticker), use_container_width=True)

        total_factor_contribution = attr["Contribution"].sum()
        actual_ann_ret = df_ra[ticker].mean() * 12
        alpha = actual_ann_ret - total_factor_contribution
        st.metric(
            "Unexplained Alpha (annualised)", f"{alpha:.2%}",
            help="Actual annualised return minus sum of factor contributions. Positive = manager added value beyond factor exposures."
        )
