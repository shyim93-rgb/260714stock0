import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")

st.sidebar.header("🔍 설정 옵션")
ticker = st.sidebar.text_input("종목 코드 입력 (예: AAPL, TSLA, 005930.KS)", value="AAPL")
start_date = st.sidebar.date_input("시작 날짜", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("종료 날짜", value=date.today())

st.title(f"📈 주식 데이터 분석 대시보드: {ticker.upper()}")

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

try:
    df = load_data(ticker, start_date, end_date)
    
    if not df.empty:
        latest_price = df['Close'].iloc[-1]
        st.metric(label=f"{ticker.upper()} 현재 종가", value=f"${latest_price:,.2f}")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Market Data"
        )])
        
        fig.update_layout(
            title=f"{ticker.upper()} 주가 차트",
            yaxis_title="가격 (USD)",
            xaxis_title="날짜",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("데이터 상세 보기"):
            st.write(df)
            
    else:
        st.error("데이터를 찾을 수 없습니다. 종목 코드를 다시 확인해주세요.")
        
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
