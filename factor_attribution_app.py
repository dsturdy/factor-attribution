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
CSV_DIR = os.path.join(BASE_DIR, "Factor_Attribution_csvs")

factor_tickers = [
    "SPY", "TLT", "HYG", "DBC", "EEM", "UUP", "TIP",
    "SVXY", "SHY", "CWY", "USMV", "MTUM", "QUAL", "IVE", "IWM", "ACWI",
    "GLD", "USO", "VIXY"
]

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

factor_cols = [
    "Equity", "Interest Rates", "Credit", "Commodities", "Emerging Markets",
    "FX", "Real Yields", "Local Inflation", "Global Equity", "Local Equity",
    "Equity Short Vol", "FI Carry", "FX Carry", "Trend", "Low Risk",
    "Momentum", "Quality", "Value", "Small Cap", "Gold", "Oil", "Volatility",
]

# Benchmark map for scorecard
BENCHMARK_MAP = {
    "Equity":        "SPY",
    "Fixed Income":  "AGG",
    "Alternatives":  "ABRYX",
    "Balanced":      "AOR",
}

PLOTLY_THEME = "plotly_dark"

# =========================
# DATA HELPERS  (unchanged from your original)
# =========================

def load_prices_from_csv(ticker: str) -> pd.DataFrame:
    path = os.path.join(CSV_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def load_yield_from_csv(ticker: str) -> pd.Series:
    path = os.path.join(CSV_DIR, f"{ticker}_Yield.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    for col in ("Yield", "Adj Close", "Close"):
        if col in df.columns:
            s = df[col] / 100.0
            # ── Artifact guard: cap yield spread at ±5% annualised ──
            s = s.clip(-0.05, 0.05)
            return s.rename(ticker)
    return pd.Series(dtype=float)


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


def prepare_factors() -> pd.DataFrame:
    price_df = download_prices(factor_tickers)
    if price_df.empty:
        return pd.DataFrame()
    price_df = price_df.resample("MS").last()
    cutoff = pd.Timestamp.today().normalize().replace(day=1)
    price_df = price_df[price_df.index < cutoff]
    raw_rets = price_df.pct_change().dropna()
    f = raw_rets.rename(columns=rename_map)
    if "ACWI" in raw_rets.columns:
        f["Global Equity"] = raw_rets["ACWI"]
    if "Equity" in f.columns and "ACWI" in raw_rets.columns:
        f["Local Equity"] = f["Equity"] - raw_rets["ACWI"]
    if "TIP" in raw_rets.columns and "TLT" in raw_rets.columns:
        tip, tlt = raw_rets["TIP"].align(raw_rets["TLT"], join="inner")
        f.loc[tip.index, "Local Inflation"] = tip - tlt
    else:
        f["Local Inflation"] = np.nan

    # FI carry with artifact guard already applied in load_yield_from_csv
    tlt_y = load_yield_from_csv("TLT").resample("MS").last()
    shy_y = load_yield_from_csv("SHY").resample("MS").last()
    fi_carry = (tlt_y - shy_y) / 12.0
    f["FI Carry"] = fi_carry.reindex(f.index)

    if "SPY" in price_df.columns:
        f["Trend"] = price_df["SPY"].pct_change(12)

    keep = [c for c in factor_cols if c in f.columns]
    return f[keep]


def get_rf(index: pd.Index) -> pd.Series:
    df = load_prices_from_csv("^IRX")
    if df.empty:
        return pd.Series(0.0, index=index, name="RF")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    rf = (df[col] / 100.0 / 12.0).resample("MS").last()
    return rf.reindex(index, method="ffill").fillna(0.0).rename("RF")


def load_and_merge_all_data(fund_tickers):
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
    """
    Returns a table of standard performance metrics across multiple horizons.
    Optionally computes active return vs a benchmark.
    """
    ret = df[fund]
    rf  = df[f"{fund}_Excess"] - ret   # reconstruct rf as excess - raw (sign flip)

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
            "Period":       label,
            "Cum Return":   cumret,
            "Ann Return":   ann_ret,
            "Ann Vol":      ann_vol,
            "Sharpe":       sharpe,
            "Max Drawdown": drawdown,
        }

        if benchmark_ticker and benchmark_ticker in df.columns:
            bret = df[benchmark_ticker].iloc[-months:]
            b_cum = (1 + bret).prod() - 1
            row["Active Return"] = cumret - b_cum

        rows.append(row)

    return pd.DataFrame(rows).set_index("Period")


def compute_return_attribution(df: pd.DataFrame, fund: str, betas: pd.Series) -> pd.DataFrame:
    """
    Decompose the fund's total return into factor contributions.
    Contribution of factor f ≈ beta_f × mean_factor_return × 12  (annualised).
    """
    cols = [c for c in betas.index if c in df.columns]
    factor_ann_ret = df[cols].mean() * 12          # annualised mean return per factor
    contributions  = (betas[cols] * factor_ann_ret).rename("Contribution")
    pct            = (contributions / contributions.abs().sum() * 100).rename("% of Total")
    result = pd.concat([betas[cols].rename("Beta"), contributions, pct], axis=1)
    result = result.reindex(result["Contribution"].abs().sort_values(ascending=False).index)
    return result


# =========================
# SCORECARD
# =========================

def flag_manager(metrics: pd.DataFrame, active_col: bool) -> tuple[str, str]:
    """
    Simple rule-based flag:
      🔴 Under Review  – 1Y Ann Return < -5%  OR  Sharpe(1Y) < 0
      🟡 On Watch      – 1Y Ann Return < 0    OR  Max Drawdown < -15%
      🟢 Approved
    Returns (status_emoji_label, reason_string)
    """
    if "1Y" not in metrics.index:
        return "⚪ Insufficient Data", "Need ≥12 months of history"

    r1y   = metrics.loc["1Y", "Ann Return"]
    sh1y  = metrics.loc["1Y", "Sharpe"]
    dd1y  = metrics.loc["1Y", "Max Drawdown"]

    if r1y < -0.05 or sh1y < 0:
        return "🔴 Under Review", f"1Y Return: {r1y:.1%} | Sharpe: {sh1y:.2f}"
    if r1y < 0.0 or dd1y < -0.15:
        return "🟡 On Watch", f"1Y Return: {r1y:.1%} | Max DD: {dd1y:.1%}"
    return "🟢 Approved", f"1Y Return: {r1y:.1%} | Sharpe: {sh1y:.2f}"


# =========================
# PLOTLY HELPERS
# =========================

def plot_rolling_level_view(rolling: pd.DataFrame):
    dfm = (
        rolling.reset_index()
               .rename(columns={rolling.index.name or "index": "index"})
               .melt(id_vars="index", var_name="Factor", value_name="Beta")
    )
    fig = px.line(dfm, x="index", y="Beta", color="Factor",
                  title="Rolling betas — all factors (level view)")
    fig.update_traces(hovertemplate=(
        "<b>%{fullData.name}</b><br>Date: %{x|%b %Y}<br>Beta: %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(template=PLOTLY_THEME,
                      legend=dict(orientation="h", y=1.1),
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")
    return fig


def plot_rolling_betas_plotly(rolling: pd.DataFrame, top_n: int = 5):
    valid_cols = rolling.columns[rolling.std().notna()]
    if valid_cols.empty:
        return None
    n = min(top_n, len(valid_cols))
    top_factors = rolling[valid_cols].std().nlargest(n).index.tolist()
    df_reset = rolling[top_factors].reset_index()
    idx_name = rolling.index.name or "index"
    df_reset = df_reset.rename(columns={idx_name: "index"})
    dfm = df_reset.melt(id_vars="index", var_name="Factor", value_name="Beta")
    fig = px.line(dfm, x="index", y="Beta", color="Factor",
                  title=f"Rolling Betas: Top {n} Most Variable Factors")
    fig.update_traces(hovertemplate=(
        "<b>%{fullData.name}</b><br>Date: %{x|%b %Y}<br>Beta: %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(template=PLOTLY_THEME,
                      legend=dict(orientation="h", y=1.1),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title="Beta")
    fig.update_xaxes(title="")
    return fig


def plot_attribution_bar(attr: pd.DataFrame, fund: str):
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in attr["Contribution"]]
    fig = go.Figure(go.Bar(
        x=attr.index,
        y=attr["Contribution"],
        marker_color=colors,
        text=[f"{v:.2%}" for v in attr["Contribution"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Contribution: %{y:.3%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Return Attribution — {fund} (annualised factor contributions)",
        template=PLOTLY_THEME,
        yaxis_tickformat=".1%",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Factor Attribution Dashboard", layout="wide")

# ── Inject custom CSS for a more polished, institutional look ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }

    /* Status badge colours */
    .badge-approved   { background:#1b4332; color:#52b788; padding:4px 12px; border-radius:4px; font-weight:600; font-size:1.1rem; }
    .badge-watch      { background:#433a1b; color:#f4a261; padding:4px 12px; border-radius:4px; font-weight:600; font-size:1.1rem; }
    .badge-review     { background:#3b1a1a; color:#e63946; padding:4px 12px; border-radius:4px; font-weight:600; font-size:1.1rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px;
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

# ── Tabs ──
tab_factor, tab_scorecard, tab_attribution = st.tabs([
    "📊 Factor Analysis",
    "🏷️ Manager Scorecard",
    "📈 Return Attribution",
])

# ─────────────────────────────────────────────
# TAB 1: FACTOR ANALYSIS  (your original flow)
# ─────────────────────────────────────────────
with tab_factor:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        fund_ticker = st.text_input("Fund ticker", value="", placeholder="SPY, AGG, EFA …").strip().upper()
    with c2:
        window = st.slider("Rolling window (months)", 12, 60, 36, 6)
    with c3:
        top_n = st.slider("Top N variable factors", 2, 10, 5)

    run = st.button("▶  Run Analysis", type="primary")

    if run:
        if not fund_ticker:
            st.error("Please enter a fund ticker.")
        else:
            with st.spinner("Loading data and running regressions…"):
                df = load_and_merge_all_data([fund_ticker])

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
                    # ── Static exposures ──
                    st.subheader(f"Static factor exposures — full sample  (R² = {r2:.2f})")
                    static_table = pd.DataFrame({"beta": betas, "t stat": tvals})
                    static_table = static_table.sort_values("beta", key=np.abs, ascending=False)

                    def color_beta(val):
                        if val > 0.1:   return "color:#52b788"
                        if val < -0.1:  return "color:#e63946"
                        return "color:#aaa"

                    st.dataframe(
                        static_table.style
                            .format("{:,.3f}")
                            .map(color_beta, subset=["beta"]),
                        use_container_width=True,
                    )

                    # Store in session for other tabs
                    st.session_state["betas"]       = betas
                    st.session_state["fund_ticker"] = fund_ticker
                    st.session_state["df"]          = df

                # ── Rolling ──
                rolling = compute_rolling(df, fund_ticker, window=window)

                if rolling.empty:
                    st.warning(f"Not enough history for a {window}-month rolling window.")
                else:
                    st.session_state["rolling"] = rolling

                    st.subheader(f"{window}-month rolling betas — all factors")
                    st.plotly_chart(plot_rolling_level_view(rolling), use_container_width=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Current (last month) betas")
                        last = rolling.iloc[-1].sort_values(key=np.abs, ascending=False)
                        st.dataframe(
                            last.to_frame("beta").style.format("{:,.3f}"),
                            use_container_width=True,
                        )
                    with col_b:
                        st.subheader(f"Top {top_n} most variable factors")
                        fig = plot_rolling_betas_plotly(rolling, top_n=top_n)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2: MANAGER SCORECARD
# ─────────────────────────────────────────────
with tab_scorecard:
    st.markdown("### Manager Oversight Scorecard")
    st.caption(
        "Enter one or more fund tickers below to generate a watch-list scorecard. "
        "Flags are rule-based: 🔴 Under Review · 🟡 On Watch · 🟢 Approved"
    )

    col_inp, col_bench = st.columns([3, 1])
    with col_inp:
        raw_input = st.text_input(
            "Tickers (comma-separated)",
            placeholder="SPY, AGG, EFA, HYG, GLD",
        )
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
                df_all = load_and_merge_all_data(all_tickers)

            if df_all is None:
                st.error("Could not load data. Check your CSVs.")
            else:
                for i, ticker in enumerate(tickers):
                    progress.progress((i + 1) / len(tickers), text=f"Analysing {ticker}…")

                    if ticker not in df_all.columns:
                        scorecard_rows.append({
                            "Ticker": ticker, "Status": "⚪ No Data",
                            "Reason": "CSV not found", "1Y Return": None,
                            "Sharpe (1Y)": None, "Max DD (1Y)": None,
                            "Active Return (1Y)": None,
                        })
                        continue

                    metrics = compute_performance_metrics(
                        df_all, ticker,
                        benchmark_ticker=benchmark if benchmark != ticker else None
                    )
                    fund_metrics_store[ticker] = metrics

                    active_col = "Active Return" in metrics.columns
                    status, reason = flag_manager(metrics, active_col)

                    row = {
                        "Ticker":  ticker,
                        "Status":  status,
                        "Reason":  reason,
                    }
                    for period in ["1M", "3M", "1Y", "3Y"]:
                        if period in metrics.index:
                            row[f"{period} Return"]  = metrics.loc[period, "Ann Return"]
                            row[f"{period} Sharpe"]  = metrics.loc[period, "Sharpe"]
                    if "1Y" in metrics.index:
                        row["Max DD (1Y)"] = metrics.loc["1Y", "Max Drawdown"]
                        if active_col and "1Y" in metrics.index:
                            row["Active Ret (1Y)"] = metrics.loc["1Y", "Active Return"]

                    scorecard_rows.append(row)

                progress.empty()

                sc_df = pd.DataFrame(scorecard_rows)

                # Summary KPIs
                k1, k2, k3 = st.columns(3)
                approved = sc_df["Status"].str.startswith("🟢").sum()
                watch    = sc_df["Status"].str.startswith("🟡").sum()
                review   = sc_df["Status"].str.startswith("🔴").sum()
                k1.metric("✅ Approved",    approved)
                k2.metric("⚠️ On Watch",    watch)
                k3.metric("🚨 Under Review", review)

                st.divider()

                # Full scorecard table
                fmt_cols = {c: "{:.2%}" for c in sc_df.columns if "Return" in c or "DD" in c}
                fmt_cols.update({c: "{:.2f}" for c in sc_df.columns if "Sharpe" in c})

                st.dataframe(
                    sc_df.style.format(fmt_cols, na_rep="—"),
                    use_container_width=True,
                    height=min(400, 40 + 35 * len(sc_df)),
                )

                # Drill-down
                st.divider()
                st.markdown("#### Fund Detail")
                drill = st.selectbox("Select fund to drill into", options=list(fund_metrics_store.keys()))
                if drill:
                    st.dataframe(
                        fund_metrics_store[drill].style.format({
                            "Cum Return":    "{:.2%}",
                            "Ann Return":    "{:.2%}",
                            "Ann Vol":       "{:.2%}",
                            "Sharpe":        "{:.2f}",
                            "Max Drawdown":  "{:.2%}",
                            "Active Return": "{:.2%}",
                        }, na_rep="—"),
                        use_container_width=True,
                    )

# ─────────────────────────────────────────────
# TAB 3: RETURN ATTRIBUTION
# ─────────────────────────────────────────────
with tab_attribution:
    st.markdown("### Return Attribution")
    st.caption(
        "Run the Factor Analysis tab first, then come back here to see how each factor "
        "contributed to the fund's annualised return."
    )

    if "betas" not in st.session_state or "df" not in st.session_state:
        st.info("👈 Run a Factor Analysis first (Tab 1) to unlock attribution.")
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
                    "Beta":         "{:.3f}",
                    "Contribution": "{:.2%}",
                    "% of Total":   "{:.1f}%",
                }),
                use_container_width=True,
            )
        with col_right:
            fig_attr = plot_attribution_bar(attr, ticker)
            st.plotly_chart(fig_attr, use_container_width=True)

        # Unexplained alpha
        total_factor_contribution = attr["Contribution"].sum()
        actual_ann_ret = (df_ra[ticker].mean() * 12)
        alpha = actual_ann_ret - total_factor_contribution
        st.metric(
            "Unexplained Alpha (annualised)",
            f"{alpha:.2%}",
            help="Actual annualised return minus sum of factor contributions. Positive = manager added value beyond factor exposures."
        )
