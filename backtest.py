import streamlit as st
import yfinance as yf
import pandas as pd

st.title("日経レバ 持ち越し戦略 バックテスト（カスタム調整版）")
st.write("見つけた黄金バランスを初期値にセットしつつ、いつでもスライダーでパラメータを微調整できるようにしました。")

# --- データの取得と整形（過去5年） ---
@st.cache_data
def get_historical_data():
    t_1570 = yf.Ticker("1570.T").history(period="5y")
    t_nq = yf.Ticker("NQ=F").history(period="5y")
    t_sox = yf.Ticker("^SOX").history(period="5y")
    t_usd = yf.Ticker("USDJPY=X").history(period="5y")
    t_vix = yf.Ticker("^VIX").history(period="5y")

    t_1570.index = t_1570.index.tz_localize(None)
    t_nq.index = t_nq.index.tz_localize(None)
    t_sox.index = t_sox.index.tz_localize(None)
    t_usd.index = t_usd.index.tz_localize(None)
    t_vix.index = t_vix.index.tz_localize(None)

    df = pd.DataFrame({
        '1570_Close': t_1570['Close'],
        '1570_Next_Open': t_1570['Open'].shift(-1), 
        'NQ_Close': t_nq['Close'],
        'SOX_Close': t_sox['Close'],
        'USD_Close': t_usd['Close'],
        'VIX_Close': t_vix['Close']
    }).dropna()

    df['NQ_pct'] = df['NQ_Close'].pct_change() * 100
    df['SOX_pct'] = df['SOX_Close'].pct_change() * 100
    df['USD_pct'] = df['USD_Close'].pct_change() * 100

    df['Trade_Return_pct'] = ((df['1570_Next_Open'] - df['1570_Close']) / df['1570_Close']) * 100

    return df.dropna()

with st.spinner("過去5年分の市場データを読み込み中..."):
    df = get_historical_data()

# --- 左側のメニュー（見つけたベストバランスを初期値に設定） ---
st.sidebar.header("🔧 ルールの調整（カイゼン）")
st.sidebar.write("見つけた黄金バランス（勝率81%超）を初期値にしています。ここからさらに微調整可能です！")

# value= に先ほどのベストな数値を指定しています
p_nq = st.sidebar.slider("ナスダックの基準値(%)", min_value=-1.0, max_value=2.0, value=0.1, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)
p_sox = st.sidebar.slider("SOX指数の基準値(%)", min_value=-2.0, max_value=3.0, value=0.1, step=0.1)

# --- バックテストの実行 ---
conditions = (
    (df['NQ_pct'] >= p_nq) &
    (df['VIX_Close'] < p_vix) &
    (df['USD_pct'] > p_usd) &
    (df['SOX_pct'] >= p_sox)
)
df_trade = df[conditions].copy()

if len(df_trade) > 0:
    df_trade['Cumulative_Return'] = df_trade['Trade_Return_pct'].cumsum()
    
    total_trades = len(df_trade)
    win_trades = len(df_trade[df_trade['Trade_Return_pct'] > 0])
    win_rate = (win_trades / total_trades) * 100
    total_profit = df_trade['Trade_Return_pct'].sum()
    avg_return_per_trade = df_trade['Trade_Return_pct'].mean()

    st.header("📊 過去5年の検証結果（カスタム調整中）")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総エントリー回数", f"{total_trades} 回")
    col2.metric("勝率", f"{win_rate:.1f} %")
    col3.metric("1回あたり平均リターン", f"{avg_return_per_trade:+.2f} %")
    col4.metric("合計リターン(単利)", f"{total_profit:.1f} %")

    st.subheader("5年間の資産推移グラフ")
    st.line_chart(df_trade['Cumulative_Return'])
    
else:
    st.warning("条件が厳しすぎます。過去5年間でエントリーできる日が1日もありませんでした。左のメニューから条件を緩めてください。")