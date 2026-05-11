"""
Drop-in replacements for the two chart functions in factor_attribution_app.py
Also includes a Streamlit UI snippet for the factor filter.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# ── 1. HEATMAP — replaces plot_rolling_level_view ──────────────────────────

def plot_rolling_heatmap(rolling):
    """
    Factor × time heatmap of rolling betas.
    Green = positive exposure, Red = negative.
    Much cleaner than 22 overlapping lines.
    """
    if rolling.empty:
        return None

    # Winsorize to ±2 so one bad data point doesn't wash out the colour scale
    data = rolling.clip(-2, 2)

    # Sort factors by absolute mean beta (most important at top)
    order = data.abs().mean().sort_values(ascending=False).index.tolist()
    data  = data[order]

    x_labels = [d.strftime("%b %Y") for d in data.index]
    y_labels  = order

    fig = go.Figure(go.Heatmap(
        z=data.T.values,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0,  "#b91c1c"],   # strong negative → deep red
            [0.35, "#fca5a5"],   # mild negative   → light red
            [0.5,  "#1e293b"],   # zero            → near-black (dark bg)
            [0.65, "#86efac"],   # mild positive   → light green
            [1.0,  "#15803d"],   # strong positive → deep green
        ],
        zmid=0,
        zmin=-2,
        zmax=2,
        colorbar=dict(
            title="Beta",
            thickness=12,
            len=0.8,
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["-2 (or less)", "-1", "0", "1", "2 (or more)"],
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Date: %{x}<br>"
            "Beta: %{z:.3f}"
            "<extra></extra>"
        ),
        xgap=1,
        ygap=1,
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Rolling betas — factor heatmap (winsorised ±2)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            tickangle=-45,
            tickmode="auto",
            nticks=12,
            title="",
        ),
        yaxis=dict(title="", autorange="reversed"),
        height=max(350, len(y_labels) * 28 + 80),
    )

    return fig


# ── 2. FILTERED LINE CHART — replaces plot_rolling_betas_plotly ────────────

def plot_rolling_filtered(rolling, selected_factors):
    """
    Line chart of rolling betas for a user-selected subset of factors.
    Winsorised to ±2 to suppress data artifacts.
    """
    if rolling.empty or not selected_factors:
        return None

    data = rolling[selected_factors].clip(-2, 2)

    dfm = (
        data.reset_index()
            .rename(columns={data.index.name or "index": "index"})
            .melt(id_vars="index", var_name="Factor", value_name="Beta")
    )

    fig = px.line(
        dfm,
        x="index",
        y="Beta",
        color="Factor",
        title=f"Rolling betas — {len(selected_factors)} selected factors",
    )

    fig.update_traces(hovertemplate=(
        "<b>%{fullData.name}</b><br>"
        "Date: %{x|%b %Y}<br>"
        "Beta: %{y:.3f}"
        "<extra></extra>"
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")

    fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Beta",
        xaxis_title="",
    )

    return fig


# ── 3. STREAMLIT SNIPPET — paste this into Tab 1 where the rolling charts go ──
"""
Replace this block in Tab 1:

    st.subheader(f"{window}-month rolling betas — all factors")
    st.plotly_chart(plot_rolling_level_view(rolling), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        ...
    with col_b:
        ...

With this:

    # ── Heatmap ──
    st.subheader(f"{window}-month rolling betas — heatmap")
    fig_heat = plot_rolling_heatmap(rolling)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Filtered line chart ──
    st.subheader("Factor detail — line view")
    all_factors = rolling.columns.tolist()

    # Default: top 5 by variance
    default_factors = rolling.std().nlargest(5).index.tolist()

    selected = st.multiselect(
        "Select factors to display",
        options=all_factors,
        default=default_factors,
        help="Pick any subset. Chart is winsorised at ±2 to suppress data artifacts.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Current (last month) betas")
        last = rolling.iloc[-1].sort_values(key=np.abs, ascending=False)
        st.dataframe(
            last.to_frame("beta").style.format("{:,.3f}"),
            use_container_width=True,
        )
    with col_b:
        if selected:
            fig_line = plot_rolling_filtered(rolling, selected)
            if fig_line:
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Select at least one factor above.")
"""
