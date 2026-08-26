import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.title("日中デイトレ戦略（安全フィルター ＋ 最初の30分ブレイク）検証ダッシュボード")
st.write("VIX・ドル円のフィルターと、朝9:00〜9:30の値動き（ORB）を組み合わせた日中戦略をテストします。")

# --- 1. サイドバー：パラメータ調整 ---
st.sidebar.header("⚙️ デイトレ戦略の条件設定")
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

# 期間設定
years = list(range(2015, datetime.now().year + 1))
months = list(range(1, 13))

col_y1, col_m1 = st.sidebar.columns(2)
start_year = col_y1.selectbox("開始 年", years, index=0)
start_month = col_m1.selectbox("開始 月", months, index=0)

col_y2, col_m2 = st.sidebar.columns(2)
end_year = col_y2.selectbox("終了 年", years, index=len(years)-1)
end_month = col_m2.selectbox("終了 月", months, index=datetime.now().month-1)

start_date = f"{start_year}-{start_month:02d}-01"
if end_month == 12:
    end_date = f"{end_year+1}-01-01"
else:
    end_date = f"{end_year}-{end_month+1:02d}-01"

# --- 2. データの取得 ---
@st.cache_data(ttl=3600)
def load_day_data():
    t_vix = yf.Ticker("^VIX").history(period="max")
    t_usd = yf.Ticker("USDJPY=X").history(period="max")
    t_n225 = yf.Ticker("^N225").history(period="max") # 日経平均

    for df in [t_vix, t_usd, t_n225]:
        df.index = df.index.tz_localize(None)

    df = pd.DataFrame({
        'VIX_Close': t_vix['Close'],
        'USD_Close': t_usd['Close'],
        'N225_Open': t_n225['Open'],
        'N225_High': t_n225['High'],
        'N225_Low': t_n225['Low'],
        'N225_Close': t_n225['Close']
    }).dropna()

    df['USD_pct'] = df['USD_Close'].pct_change() * 100
    
    # 日中のリターン（寄り付きから引けまで）: (Close - Open) / Open
    df['Intraday_Return'] = (df['N225_Close'] - df['N225_Open']) / df['N225_Open'] * 100

    # 最初の30分の擬似ブレイク判定（日足のHigh/Openの勢いなどを代理指標として活用）
    # ※日足ベースで「前日終値より高く寄り付き、かつ日中も高値に向かって伸びたか」をシミュレーション
    df['Gap_Open'] = (df['N225_Open'] - df['N225_Close'].shift(1)) / df['N225_Close'].shift(1) * 100
    
    return df.dropna()

with st.spinner("データを読み込んでデイトレ戦略をシミュレーション中..."):
    df_all = load_day_data()

mask = (df_all.index >= start_date) & (df_all.index < end_date)
df = df_all.loc[mask].copy()

if len(df) == 0:
    st.error("指定された期間のデータがありません。")
    st.stop()

# --- 3. 判定ロジックの適用 ---
df['Cond_VIX'] = df['VIX_Close'] < p_vix
df['Cond_USD'] = df['USD_pct'] > p_usd
# 窓開けしてスタートし、かつ日中もプラスに伸びる傾向がある日をORB成功と仮定
df['Cond_ORB'] = df['Gap_Open'] > 0 

df['Signal'] = df['Cond_VIX'] & df['Cond_USD'] & df['Cond_ORB']

trades = df[df['Signal']].copy()
win_trades = trades[trades['Intraday_Return'] > 0]

total_trades = len(trades)
win_count = len(win_trades)
win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
total_return = trades['Intraday_Return'].sum()

# --- 4. 結果表示 ---
st.header(f"📊 デイトレ検証結果 （期間: {start_year}年{start_month}月 〜 {end_year}年{end_month}月）")

col1, col2, col3, col4 = st.columns(4)
col1.metric("総トレード回数", f"{total_trades} 回")
col2.metric("勝率", f"{win_rate:.1f}%" if total_trades > 0 else "0%")
col3.metric("勝ち / 負け", f"{win_count}勝 / {total_trades - win_count}敗")
col4.metric("累積リターン(合算)", f"{total_return:+.2f}%")

st.markdown("---")

if total_trades > 0:
    trades['Cumulative_Return'] = trades['Intraday_Return'].cumsum()
    st.subheader("📈 日中戦略の資産推移グラフ")
    st.line_chart(trades['Cumulative_Return'])
else:
    st.warning("条件に一致するトレードがありませんでした。")

st.subheader("📋 トレード履歴（日中）")
if total_trades > 0:
    display_df = trades[['Gap_Open', 'USD_pct', 'VIX_Close', 'Intraday_Return']].copy()
    display_df.columns = ['寄り付き窓開け(%)', 'ドル円前日比(%)', 'VIX値', '日中リターン(9時→15時)(%)']
    st.dataframe(display_df.sort_index(ascending=False))