import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ────────────────────────────────
# 페이지 기본 설정
# ────────────────────────────────
st.set_page_config(
    page_title="📈 주식 데이터 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 인터랙티브 주식 데이터 분석 대시보드")
st.caption("Yahoo Finance 데이터 기반 · Plotly 시각화")

# ────────────────────────────────
# 사이드바 - 사용자 입력
# ────────────────────────────────
st.sidebar.header("⚙️ 설정")

ticker_input = st.sidebar.text_input(
    "티커 입력 (쉼표로 구분, 최대 5개)",
    value="AAPL, MSFT, GOOGL"
).upper()

tickers = [t.strip() for t in ticker_input.split(",") if t.strip()][:5]

period_options = {
    "1개월": "1mo", "3개월": "3mo", "6개월": "6mo",
    "1년": "1y", "2년": "2y", "5년": "5y", "전체": "max"
}
period_label = st.sidebar.selectbox("기간 선택", list(period_options.keys()), index=3)
period = period_options[period_label]

interval_options = {
    "1일": "1d", "1주": "1wk", "1개월": "1mo"
}
interval_label = st.sidebar.selectbox("간격(interval)", list(interval_options.keys()), index=0)
interval = interval_options[interval_label]

st.sidebar.subheader("📊 보조지표")
show_ma = st.sidebar.checkbox("이동평균선 (MA)", value=True)
ma_periods = st.sidebar.multiselect("MA 기간", [5, 20, 60, 120, 200], default=[20, 60])
show_bollinger = st.sidebar.checkbox("볼린저 밴드", value=False)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)
show_volume = st.sidebar.checkbox("거래량", value=True)

chart_type = st.sidebar.radio("차트 유형", ["캔들스틱", "라인"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance)")

if not tickers:
    st.warning("사이드바에 티커를 최소 1개 이상 입력해주세요. (예: AAPL, TSLA, 005930.KS)")
    st.stop()

# ────────────────────────────────
# 데이터 로드 함수
# ────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

@st.cache_data(ttl=3600)
def load_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}

def compute_indicators(df, ma_periods):
    df = df.copy()
    for p in ma_periods:
        df[f"MA{p}"] = df["Close"].rolling(window=p).mean()

    # 볼린저 밴드 (20일 기준)
    df["BB_MID"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std

    # RSI (14일 기준)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    return df

# ────────────────────────────────
# 메인 티커 (첫 번째)로 상세 차트 표시
# ────────────────────────────────
main_ticker = tickers[0]

with st.spinner(f"{main_ticker} 데이터를 불러오는 중..."):
    df = load_data(main_ticker, period, interval)
    info = load_info(main_ticker)

if df.empty:
    st.error(f"'{main_ticker}' 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
    st.stop()

df = compute_indicators(df, ma_periods)

# ────────────────────────────────
# 상단 요약 지표 (KPI 카드)
# ────────────────────────────────
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
pct_change = (change / prev_close * 100) if prev_close else 0
period_high = float(df["High"].max())
period_low = float(df["Low"].min())
avg_volume = float(df["Volume"].mean())

company_name = info.get("longName", main_ticker)
currency = info.get("currency", "USD")

st.subheader(f"{company_name} ({main_ticker})")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("현재가", f"{last_close:,.2f} {currency}", f"{change:+.2f} ({pct_change:+.2f}%)")
col2.metric("기간 최고가", f"{period_high:,.2f}")
col3.metric("기간 최저가", f"{period_low:,.2f}")
col4.metric("평균 거래량", f"{avg_volume:,.0f}")
if info.get("marketCap"):
    col5.metric("시가총액", f"{info['marketCap']/1e8:,.0f}억")
else:
    col5.metric("시가총액", "N/A")

st.markdown("---")

# ────────────────────────────────
# 메인 차트 (가격 + 거래량 + RSI + MACD)
# ────────────────────────────────
rows = 1
row_heights = [0.6]
specs_titles = ["가격"]

if show_volume:
    rows += 1
    row_heights.append(0.15)
    specs_titles.append("거래량")
if show_rsi:
    rows += 1
    row_heights.append(0.15)
    specs_titles.append("RSI")
if show_macd:
    rows += 1
    row_heights.append(0.2)
    specs_titles.append("MACD")

total = sum(row_heights)
row_heights = [h / total for h in row_heights]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=row_heights,
    subplot_titles=specs_titles
)

current_row = 1

# 가격 차트
if chart_type == "캔들스틱":
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=main_ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=current_row, col=1)
else:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="종가",
        line=dict(color="#2962ff", width=2)
    ), row=current_row, col=1)

if show_ma:
    colors = ["#ff9800", "#9c27b0", "#4caf50", "#795548", "#607d8b"]
    for i, p in enumerate(ma_periods):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA{p}"], mode="lines",
            name=f"MA{p}", line=dict(width=1.3, color=colors[i % len(colors)])
        ), row=current_row, col=1)

if show_bollinger:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], line=dict(width=1, color="rgba(150,150,150,0.5)"), name="BB 상단"), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], line=dict(width=1, color="rgba(150,150,150,0.5)"), name="BB 하단", fill="tonexty", fillcolor="rgba(150,150,150,0.1)"), row=current_row, col=1)

# 거래량
if show_volume:
    current_row += 1
    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=vol_colors), row=current_row, col=1)

# RSI
if show_rsi:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#7b1fa2", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)

# MACD
if show_macd:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2962ff", width=1.3)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#ff6d00", width=1.3)), row=current_row, col=1)
    hist_colors = np.where(df["MACD_HIST"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram", marker_color=hist_colors), row=current_row, col=1)

fig.update_layout(
    height=250 * rows + 200,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────
# 다중 티커 비교 차트 (수익률 기준)
# ────────────────────────────────
if len(tickers) > 1:
    st.markdown("---")
    st.subheader("📊 종목 간 수익률 비교 (기간 시작 대비 %)")

    compare_fig = go.Figure()
    comparison_data = {}

    for t in tickers:
        try:
            d = load_data(t, period, interval)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d.empty:
                continue
            norm = (d["Close"] / d["Close"].iloc[0] - 1) * 100
            comparison_data[t] = norm
            compare_fig.add_trace(go.Scatter(
                x=d.index, y=norm, mode="lines", name=t
            ))
        except Exception:
            st.warning(f"'{t}' 데이터를 불러오지 못했습니다.")

    compare_fig.update_layout(
        height=450,
        yaxis_title="수익률 (%)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    if comparison_data:
        summary_df = pd.DataFrame({
            t: [f"{s.iloc[-1]:+.2f}%"] for t, s in comparison_data.items()
        }, index=["기간 수익률"]).T
        st.dataframe(summary_df, use_container_width=True)

# ────────────────────────────────
# 원본 데이터 테이블 + 다운로드
# ────────────────────────────────
st.markdown("---")
with st.expander("📄 원본 데이터 보기"):
    st.dataframe(df.tail(200).sort_index(ascending=False), use_container_width=True)
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드", data=csv,
        file_name=f"{main_ticker}_{period}_{interval}.csv",
        mime="text/csv"
    )

st.caption("⚠️ 본 대시보드는 정보 제공 목적이며 투자 조언이 아닙니다.")
