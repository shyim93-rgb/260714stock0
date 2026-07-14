import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ────────────────────────────────
# 페이지 기본 설정
# ────────────────────────────────
st.set_page_config(
    page_title="🇰🇷 한국 AI·반도체 주식 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🇰🇷 한국 AI·반도체 대표주 분석 대시보드")
st.caption("Yahoo Finance 데이터 기반 · Plotly 인터랙티브 시각화")

# ────────────────────────────────
# 종목 카탈로그 (카테고리별)
# ────────────────────────────────
STOCK_CATALOG = {
    "반도체 대형주": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "삼성전자우": "005935.KS",
    },
    "반도체 소부장": {
        "DB하이텍": "000990.KS",
        "한미반도체": "042700.KS",
        "리노공업": "058470.KQ",
        "원익IPS": "240810.KQ",
        "이오테크닉스": "039030.KQ",
        "티씨케이": "064760.KQ",
        "동진쎄미켐": "005290.KQ",
        "solbrain(솔브레인)": "357780.KQ",
    },
    "AI / 플랫폼": {
        "네이버": "035420.KS",
        "카카오": "035720.KS",
        "크래프톤": "259960.KS",
        "더존비즈온": "012510.KS",
        "삼성SDS": "018260.KS",
    },
    "AI 중소형주": {
        "셀바스AI": "108860.KQ",
        "솔트룩스": "304100.KQ",
        "코난테크놀로지": "402030.KQ",
        "알체라": "347860.KQ",
        "마인즈랩": "377480.KQ",
        "루닛": "328130.KQ",
    },
    "IT/전자 대형주": {
        "LG전자": "066570.KS",
        "삼성전기": "009150.KS",
        "LG이노텍": "011070.KS",
    },
}

# 카테고리 → {이름: 티커} 를 하나로 펼친 딕셔너리
ALL_STOCKS = {}
for cat, stocks in STOCK_CATALOG.items():
    ALL_STOCKS.update(stocks)

# ────────────────────────────────
# 사이드바 - 종목 선택
# ────────────────────────────────
st.sidebar.header("⚙️ 설정")
st.sidebar.subheader("📌 종목 선택 (최대 5개)")

selected_names = []
for cat, stocks in STOCK_CATALOG.items():
    with st.sidebar.expander(cat, expanded=(cat in ["반도체 대형주", "AI / 플랫폼"])):
        for name in stocks:
            default = name in ["삼성전자", "SK하이닉스", "네이버"]
            if st.checkbox(name, value=default, key=f"chk_{name}"):
                selected_names.append(name)

if len(selected_names) > 5:
    st.sidebar.warning("최대 5개까지만 표시됩니다. 상위 5개만 사용합니다.")
    selected_names = selected_names[:5]

tickers = [ALL_STOCKS[n] for n in selected_names]
name_map = {ALL_STOCKS[n]: n for n in selected_names}

st.sidebar.markdown("---")

period_options = {
    "1개월": "1mo", "3개월": "3mo", "6개월": "6mo",
    "1년": "1y", "2년": "2y", "5년": "5y", "전체": "max"
}
period_label = st.sidebar.selectbox("기간 선택", list(period_options.keys()), index=3)
period = period_options[period_label]

interval_options = {"1일": "1d", "1주": "1wk", "1개월": "1mo"}
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
st.sidebar.caption("데이터 출처: Yahoo Finance (yfinance) · 실시간 대비 지연 가능")

if not tickers:
    st.warning("사이드바에서 종목을 최소 1개 이상 선택해주세요.")
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
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def compute_indicators(df, ma_periods):
    df = df.copy()
    for p in ma_periods:
        df[f"MA{p}"] = df["Close"].rolling(window=p).mean()

    df["BB_MID"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    return df

def format_krw(value):
    """원화 금액을 조/억 단위로 보기 좋게 표시"""
    if value is None or pd.isna(value):
        return "N/A"
    if value >= 1e12:
        return f"{value/1e12:,.1f}조원"
    elif value >= 1e8:
        return f"{value/1e8:,.0f}억원"
    else:
        return f"{value:,.0f}원"

# ────────────────────────────────
# 메인 종목 (첫 번째 선택) 상세 차트
# ────────────────────────────────
main_ticker = tickers[0]
main_name = name_map[main_ticker]

with st.spinner(f"{main_name} 데이터를 불러오는 중..."):
    df = load_data(main_ticker, period, interval)
    info = load_info(main_ticker)

if df.empty:
    st.error(f"'{main_name} ({main_ticker})' 데이터를 불러올 수 없습니다.")
    st.stop()

df = compute_indicators(df, ma_periods)

# ────────────────────────────────
# 상단 요약 지표
# ────────────────────────────────
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
pct_change = (change / prev_close * 100) if prev_close else 0
period_high = float(df["High"].max())
period_low = float(df["Low"].min())
avg_volume = float(df["Volume"].mean())
market_cap = info.get("marketCap")

st.subheader(f"{main_name} ({main_ticker})")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("현재가", f"{last_close:,.0f}원", f"{change:+,.0f}원 ({pct_change:+.2f}%)")
col2.metric("기간 최고가", f"{period_high:,.0f}원")
col3.metric("기간 최저가", f"{period_low:,.0f}원")
col4.metric("평균 거래량", f"{avg_volume:,.0f}주")
col5.metric("시가총액", format_krw(market_cap))

st.markdown("---")

# ────────────────────────────────
# 메인 차트
# ────────────────────────────────
rows = 1
row_heights = [0.6]
specs_titles = ["가격"]

if show_volume:
    rows += 1; row_heights.append(0.15); specs_titles.append("거래량")
if show_rsi:
    rows += 1; row_heights.append(0.15); specs_titles.append("RSI")
if show_macd:
    rows += 1; row_heights.append(0.2); specs_titles.append("MACD")

total = sum(row_heights)
row_heights = [h / total for h in row_heights]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=row_heights,
    subplot_titles=specs_titles
)

current_row = 1

if chart_type == "캔들스틱":
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=main_name,
        increasing_line_color="#d32f2f", decreasing_line_color="#1565c0"
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

if show_volume:
    current_row += 1
    vol_colors = np.where(df["Close"] >= df["Open"], "#d32f2f", "#1565c0")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=vol_colors), row=current_row, col=1)

if show_rsi:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#7b1fa2", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)

if show_macd:
    current_row += 1
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2962ff", width=1.3)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#ff6d00", width=1.3)), row=current_row, col=1)
    hist_colors = np.where(df["MACD_HIST"] >= 0, "#d32f2f", "#1565c0")
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
# 다중 종목 비교 (수익률 기준)
# ────────────────────────────────
if len(tickers) > 1:
    st.markdown("---")
    st.subheader("📊 선택 종목 간 수익률 비교 (기간 시작 대비 %)")

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
            label = name_map.get(t, t)
            comparison_data[label] = norm
            compare_fig.add_trace(go.Scatter(x=d.index, y=norm, mode="lines", name=label))
        except Exception:
            st.warning(f"'{name_map.get(t, t)}' 데이터를 불러오지 못했습니다.")

    compare_fig.update_layout(
        height=450, yaxis_title="수익률 (%)", hovermode="x unified",
        template="plotly_white", margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    if comparison_data:
        summary_df = pd.DataFrame({
            name: [f"{s.iloc[-1]:+.2f}%"] for name, s in comparison_data.items()
        }, index=["기간 수익률"]).T
        st.dataframe(summary_df, use_container_width=True)

# ────────────────────────────────
# 원본 데이터 + 다운로드
# ────────────────────────────────
st.markdown("---")
with st.expander("📄 원본 데이터 보기"):
    st.dataframe(df.tail(200).sort_index(ascending=False), use_container_width=True)
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드", data=csv,
        file_name=f"{main_name}_{period}_{interval}.csv",
        mime="text/csv"
    )

st.caption("⚠️ 본 대시보드는 정보 제공 목적이며 투자 조언이 아닙니다. 데이터는 Yahoo Finance 기준으로 실제 거래가와 차이가 있을 수 있습니다.")
