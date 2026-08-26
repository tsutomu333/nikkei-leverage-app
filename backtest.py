import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("夕方買い・翌朝売り戦略（ナイトセッション）検証ダッシュボード")
st.write("VIX・ドル円のフィルターを組み合わせた持ち越し戦略のパフォーマンスとリスク（最大DD）を検証します。")

# --- 1. サイドバー：パラメータ調整 ---
st.sidebar.header("⚙️ 戦略の条件設定")
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

# 期間設定
years = list(range(2018, datetime.now().year + 1))
months = list(range(1, 13))

col_y1, col_m1 = st.sidebar.columns(2)
start_year = col_y1.selectbox("開始 年", years, index=len(years)-2)
start_month = col_m1.selectbox("開始 月", months, index=8)

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
def load_night_data():
    t_vix = yf.download("^VIX", period="5y", progress=False)
    time.sleep(1)
    t_usd = yf.download("USDJPY=X", period="5y", progress=False)
    time.sleep(1)
    t_n225 = yf.download("^N225", period="5y", progress=False)

    for df in [t_vix, t_usd, t_n225]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)

    df = pd.DataFrame({
        'VIX_Close': t_vix['Close'],
        'USD_Close': t_usd['Close'],
        'N225_Open': t_n225['Open'],
        'N225_Close': t_n225['Close']
    }).dropna()

    df['USD_pct'] = df['USD_Close'].pct_change() * 100
    # 持ち越しリターン（前日終値から当日寄り付きまで）: (Open - Close.shift(1)) / Close.shift(1)
    df['Night_Return'] = (df['N225_Open'] - df['N225_Close'].shift(1)) / df['N225_Close'].shift(1) * 100
    
    return df.dropna()

with st.spinner("データを安全に取得してシミュレーション中..."):
    df_all = load_night_data()

mask = (df_all.index >= start_date) & (df_all.index < end_date)
df = df_all.loc[mask].copy()

if len(df) == 0:
    st.error("指定された期間のデータがありません。")
    st.stop()

# --- 3. 判定ロジックとドローダウン計算 ---
df['Cond_VIX'] = df['VIX_Close'] < p_vix
df['Cond_USD'] = df['USD_pct'] > p_usd

df['Signal'] = df['Cond_VIX'] & df['Cond_USD']

trades = df[df['Signal']].copy()
win_trades = trades[trades['Night_Return'] > 0]

total_trades = len(trades)
win_count = len(win_trades)
lose_count = total_trades - win_count
win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
total_return = trades['Night_Return'].sum()

# 最大ドローダウン（MDD）の計算
if total_trades > 0:
    trades['Cumulative_Return'] = trades['Night_Return'].cumsum()
    running_max = trades['Cumulative_Return'].cummax()
    drawdown = trades['Cumulative_Return'] - running_max
    max_drawdown = drawdown.min()
else:
    max_drawdown = 0.0

# --- 4. 結果表示 ---
st.header(f"📊 持ち越し戦略検証結果 （期間: {start_year}年{start_month}月 〜 {end_year}年{end_month}月）")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("総トレード", f"{total_trades} 回")
col2.metric("勝率", f"{win_rate:.1f}%" if total_trades > 0 else "0%")
col3.metric("勝敗", f"{win_count}勝/{lose_count}敗")
col4.metric("累積リターン", f"{total_return:+.2f}%")
col5.metric("最大DD", f"{max_drawdown:.2f}%") # 最大ドローダウン表示

st.markdown("---")

if total_trades > 0:
    st.subheader("📈 持ち越し戦略の資産推移グラフ")
    st.line_chart(trades['Cumulative_Return'])
else:
    st.warning("条件に一致するトレードがありませんでした。")

st.subheader("📋 トレード履歴（持ち越し）")
if total_trades > 0:
    display_df = trades[['USD_pct', 'VIX_Close', 'Night_Return']].copy()
    display_df.columns = ['ドル円前日比(%)', 'VIX値', '持ち越しリターン(前日終値→当日寄付)(%)']
    st.dataframe(display_df.sort_index(ascending=False))