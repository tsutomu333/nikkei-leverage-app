import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

st.title("📊 日中デイトレ・バックテスト検証アプリ（NYダウ搭載版）")
st.write("1570（日経レバ）の朝イチ寄り引け戦略を、過去データでシミュレーションします。")

# --- 1. サイドバー：検証条件の設定 ---
st.sidebar.header("⚙️ 検証ルールの設定")
p_gap = st.sidebar.slider("③ 朝の窓開け基準値(%)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
p_dow = st.sidebar.slider("④ 前日NYダウの基準値(%)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
p_vix = st.sidebar.slider("① VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("② ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

st.sidebar.markdown("---")
years = st.sidebar.slider("検証期間（過去何年分？）", min_value=1, max_value=10, value=3)

# --- 2. 過去データの取得 ---
@st.cache_data(ttl=3600)
def load_historical_data(years):
    end_date = datetime.date.today()
    start_date = end_date - relativedelta(years=years)
    
    # 現物1570、VIX、ドル円、NYダウを取得
    t_1570 = yf.download("1570.T", start=start_date, end=end_date, progress=False)
    t_vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    t_usd = yf.download("USDJPY=X", start=start_date, end=end_date, progress=False)
    t_dow = yf.download("^DJI", start=start_date, end=end_date, progress=False)
    
    # マルチインデックスの解除（yfinanceの仕様変更対応）
    for df in [t_1570, t_vix, t_usd, t_dow]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None).normalize()
        
    return t_1570, t_vix, t_usd, t_dow

with st.spinner(f"過去{years}年分の市場データを取得・統合中..."):
    t_1570, t_vix, t_usd, t_dow = load_historical_data(years)

# --- 3. データの結合とズレ（Look-ahead bias）の排除 ---
# 日本市場（1570）のカレンダーをベースにする
df = t_1570[['Open', 'Close']].copy()

# 各種指標を日付で結合（左結合）
df = df.join(t_vix[['Close']].rename(columns={'Close': 'VIX'}), how='left')
df = df.join(t_usd[['Close']].rename(columns={'Close': 'USD'}), how='left')
df = df.join(t_dow[['Close']].rename(columns={'Close': 'DOW'}), how='left')

# 休日などで欠損したデータは前日の値で埋める
df = df.ffill()

# ⚠️最重要：当日の朝9時時点で参照できるのは「前日」のデータのみ。すべて1日（shift）ズラす。
df['VIX_Prev'] = df['VIX'].shift(1)
df['USD_Prev'] = df['USD'].shift(1)
df['USD_Prev2'] = df['USD'].shift(2)
df['DOW_Prev'] = df['DOW'].shift(1)
df['DOW_Prev2'] = df['DOW'].shift(2)

# 各種パーセンテージの計算
df['1570_PrevClose'] = df['Close'].shift(1)
df['Gap_Pct'] = (df['Open'] - df['1570_PrevClose']) / df['1570_PrevClose'] * 100
df['Dow_Pct'] = (df['DOW_Prev'] - df['DOW_Prev2']) / df['DOW_Prev2'] * 100
df['USD_Pct'] = (df['USD_Prev'] - df['USD_Prev2']) / df['USD_Prev2'] * 100

# 1570の日中リターン（朝買って、引けで売った場合の利益率）
df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open'] * 100

# 不要なNaNを削除（最初の数日分）
df = df.dropna()

# --- 4. トレードの判定（フィルター適用） ---
cond_gap = df['Gap_Pct'] >= p_gap
cond_dow = df['Dow_Pct'] >= p_dow
cond_vix = df['VIX_Prev'] < p_vix
cond_usd = df['USD_Pct'] > p_usd

# 全条件をクリアした日（True/False）
df['Trade_Signal'] = cond_gap & cond_dow & cond_vix & cond_usd

# トレードした日のデータだけを抽出
trades = df[df['Trade_Signal']].copy()

# --- 5. 検証結果の表示 ---
st.markdown("---")
st.subheader("🎯 バックテスト検証結果")

if len(trades) > 0:
    # 勝敗の計算
    trades['Win'] = trades['Daily_Return'] > 0
    win_rate = trades['Win'].mean() * 100
    avg_return = trades['Daily_Return'].mean()
    total_return = trades['Daily_Return'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総トレード回数", f"{len(trades)} 回")
    col2.metric("勝率", f"{win_rate:.1f} %")
    col3.metric("1回あたりの平均利益", f"{avg_return:+.2f} %")
    col4.metric("累積リターン(単利)", f"{total_return:+.2f} %")
    
    st.markdown("### 📈 資産推移シミュレーション（累積リターン）")
    # 資産の推移グラフを描画
    trades['Cumulative_Return'] = trades['Daily_Return'].cumsum()
    st.line_chart(trades['Cumulative_Return'])
    
    st.markdown("### 📝 直近のトレード履歴")
    # 見やすいように列を絞って直近10件を表示
    display_cols = ['Gap_Pct', 'Dow_Pct', 'VIX_Prev', 'USD_Pct', 'Daily_Return']
    st.dataframe(trades[display_cols].tail(10).style.format("{:.2f}"))

else:
    st.warning("⚠️ 指定された条件が厳しすぎます。過去のデータで1回もトレード条件を満たしませんでした。スライダーを調整してください。")