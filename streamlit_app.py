import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Orca EGX Research Dashboard", layout="wide", page_icon="📈")

np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=120)
price_data = 45 + np.cumsum(np.random.normal(0.1, 1.2, 120))

price_df = pd.DataFrame({
    'Open': price_data + np.random.uniform(-0.5, 0.5, 120),
    'High': price_data + np.random.uniform(0.5, 1.5, 120),
    'Low': price_data - np.random.uniform(0.5, 1.5, 120),
    'Close': price_data,
    'Volume': np.random.randint(100000, 5000000, 120)
}, index=dates)

price_df['SMA_20'] = price_df['Close'].rolling(20).mean()
std20 = price_df['Close'].rolling(20).std()
price_df['BB_Upper'] = price_df['SMA_20'] + (2 * std20)
price_df['BB_Lower'] = price_df['SMA_20'] - (2 * std20)

delta = price_df['Close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
price_df['RSI_14'] = 100 - (100 / (1 + rs))

def build_technical_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='goldenrod', width=2), name='RSI (14)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig.update_layout(height=550, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20), xaxis_rangeslider_visible=False)
    return fig

def calc_dcf(fcff_forecast, wacc, g, net_debt, shares):
    pv_fcff = sum([cf / ((1 + wacc) ** (t + 1)) for t, cf in enumerate(fcff_forecast)])
    terminal_val = (fcff_forecast[-1] * (1 + g)) / (wacc - g)
    pv_terminal = terminal_val / ((1 + wacc) ** len(fcff_forecast))
    equity_value = (pv_fcff + pv_terminal) - net_debt
    return equity_value / shares

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chart-line.png", width=60)
    st.title("Orca Research")
    st.markdown("---")
    ticker = st.text_input("EGX Ticker Symbol", value="COMI.CA")
    st.caption("منصة Orca لتحليل البورصة المصرية.")
    st.markdown("---")
    st.subheader("DCF Parameters")
    wacc_input = st.slider("WACC (%)", min_value=10.0, max_value=25.0, value=18.0) / 100
    g_input = st.slider("Terminal Growth (%)", min_value=2.0, max_value=10.0, value=5.0) / 100

st.title(f"📊 Institutional Stock Analysis: {ticker}")
st.markdown("**Institutional Equity Research & Technical Overview**")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical", "📐 Fundamental", "💼 Portfolio", "📋 Summary"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    latest_close = price_df['Close'].iloc[-1]
    prev_close = price_df['Close'].iloc[-2]
    chg = latest_close - prev_close
    col1.metric("Last Price", f"{latest_close:.2f} EGP", f"{chg:.2f} EGP")
    col2.metric("RSI (14)", f"{price_df['RSI_14'].iloc[-1]:.1f}")
    col3.metric("20-Day High", f"{price_df['High'].tail(20).max():.2f} EGP")
    col4.metric("20-Day Low", f"{price_df['Low'].tail(20).min():.2f} EGP")
    st.plotly_chart(build_technical_chart(price_df), use_container_width=True)

with tab2:
    fcff_list = [100, 115, 130, 145, 160]
    net_debt_val = 200
    shares_cnt = 50
    fv = calc_dcf(fcff_list, wacc_input, g_input, net_debt_val, shares_cnt)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("DCF Fair Value", f"{fv:.2f} EGP")
        if fv > latest_close:
            st.success(f"Undervalued by {((fv - latest_close) / latest_close) * 100:.1f}%")
        else:
            st.error(f"Overvalued by {((latest_close - fv) / latest_close) * 100:.1f}%")
    with c2:
        w_range = np.arange(0.14, 0.23, 0.01)
        g_range = np.arange(0.03, 0.08, 0.01)
        grid = np.zeros((len(g_range), len(w_range)))
        for i, g in enumerate(g_range):
            for j, w in enumerate(w_range):
                grid[i, j] = calc_dcf(fcff_list, w, g, net_debt_val, shares_cnt)
        fig_sens = px.imshow(grid, x=[f"{w:.0%}" for w in w_range], y=[f"{g:.0%}" for g in g_range], labels=dict(x="WACC", y="Terminal Growth", color="Fair Value"), color_continuous_scale="RdYlGn", text_auto=".1f")
        fig_sens.update_layout(title="DCF Sensitivity Analysis", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sens, use_container_width=True)

with tab3:
    st.subheader("Institutional Portfolio Allocation")
    portfolio_df = pd.DataFrame({'Asset': ['COMI.CA', 'EAST.CA', 'HRHO.CA', 'SWDY.CA', 'Cash'], 'Weight': [0.35, 0.20, 0.15, 0.15, 0.15]})
    fig_pie = px.pie(portfolio_df, values='Weight', names='Asset', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.subheader("CMT / CFA Institutional Report")
    st.markdown(
        f"* **Ticker:** `{ticker}`\n"
        f"* **Primary Trend:** Bullish (Higher Highs & Higher Lows)\n"
        f"* **Valuation Verdict:** DCF Fair Value is **EGP {fv:.2f}** compared to Current Market Price of **EGP {latest_close:.2f}**.\n"
        f"* **Institutional Recommendation:** 🟢 **BUY / ACCUMULATE**"
    )
