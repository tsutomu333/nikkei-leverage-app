import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.title("日経レバ 戦略バックテスト・検証ダッシュボード")
st.write("過去の市場データ（最大30年分）を使って、任意の期間・条件での勝率や損益をシミュレーションします。")

# --- 1. サイドバー：期間設定とパラメータ調整 ---
st.sidebar.header("📅 検証期間の設定")
st.sidebar.write("テストしたい期間の「年月」を指定してください。")

# セレクトボックスや日付選択で年月を指定できるようにする
# 過去30年（1996年〜現在）の年を選択肢にする
years = list(range(1996, datetime.now().year + 1))
months = list(range(1, 13))

col_y1, col_m1 = st.sidebar.columns(2)
start_year = col_y1.selectbox("開始 年", years, index=years.index(2015)) # デフォルト2015年頃から
start_month = col_m1.selectbox("開始 月", months, index=0)

col_y2, col_m2 = st.sidebar.columns(2)
end_year = col_y2.selectbox("終了 年", years, index=len(years)-1)
end_month = col_m2.selectbox("終了 月", months, index=datetime.now().month-1)

start_date = f"{start_year}-{start_month:02d}-01"
# 終了月は月末付近にするため簡易的に翌月1日手前などに調整
if end_month == 12:
    end_date = f"{end_year+1}-01-01"
else:
    end_date = f"{end_year}-{end_month+1:02d}-01"

st.sidebar.markdown("---")
st.sidebar.header("⚙️ エントリー条件（黄金バランス）の調整")
p_nq = st.sidebar.slider("ナスダックの基準値(%)", min_value=-1.0, max_value=2.0, value=0.1, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)
p_sox = st.sidebar.slider("SOX指数の基準値(%)", min_value=-2.0, max_value=3.0, value=0.1, step=0.1)

# --- 2. データの取得（最大30年分） ---
@st.cache_data(ttl=3600)
def load_long_term_data():
    # 30年分のデータを一括取得（Yahoo Finance）
    t_nq = yf.Ticker("NQ=F").history(period="max")
    t_sox = yf.Ticker("^SOX").history(period="max")
    t_usd = yf.Ticker("USDJPY=X").history(period="max")
    t_vix = yf.Ticker("^VIX").history(period="max")
    t_n225 = yf.Ticker("^N225").history(period="max") # 日経平均（検証用）

    # タイムゾーンを外してインデックスを日付のみに合わせる
    for df in [t_nq, t_sox, t_usd, t_vix, t_n225]:
        df.index = df.index.tz_localize(None)

    # 1つのテーブルにマージ
    df = pd.DataFrame({
        'NQ_Close': t_nq['Close'],
        'SOX_Close': t_sox['Close'],
        'USD_Close': t_usd['Close'],
        'VIX_Close': t_vix['Close'],
        'N225_Open': t_n225['Open'],
        'N225_Close': t_n225['Close']
    }).dropna()

    # 変動率（前日比%）の計算
    df['NQ_pct'] = df['NQ_Close'].pct_change() * 100
    df['SOX_pct'] = df['SOX_Close'].pct_change() * 100
    df['USD_pct'] = df['USD_Close'].pct_change() * 100
    
    # 翌日の日経平均のリターン（寄付で買って、翌朝寄付で売る、あるいは当日の夕方買って翌朝売る想定）
    # ここでは簡易的に「翌日のオープンからクローズ（または翌日オープン）」の動きをシミュレーション
    df['Next_Return'] = (df['N225_Close'].shift(-1) - df['N225_Open'].shift(-1)) / df['N225_Open'].shift(-1) * 100

    return df.dropna()

with st.spinner("過去30年分の巨大データを読み込んでバックテストを実行中..."):
    df_all = load_long_term_data()

# ユーザーが指定した期間で絞り込み
mask = (df_all.index >= start_date) & (df_all.index < end_date)
df = df_all.loc[mask].copy()

if len(df) == 0:
    st.error("指定された期間のデータがありません。期間を変更してください。")
    st.stop()

# --- 3. バックテストのシミュレーション計算 ---
df['Cond_NQ'] = df['NQ_pct'] >= p_nq
df['Cond_VIX'] = df['VIX_Close'] < p_vix
df['Cond_USD'] = df['USD_pct'] > p_usd
df['Cond_SOX'] = df['SOX_pct'] >= p_sox

df['Signal'] = df['Cond_NQ'] & df['Cond_VIX'] & df['Cond_USD'] & df['Cond_SOX']

# シグナルがTrueの日の翌日リターンを抽出
trades = df[df['Signal']].copy()
win_trades = trades[trades['Next_Return'] > 0]

total_trades = len(trades)
win_count = len(win_trades)
win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
total_return = trades['Next_Return'].sum()

# --- 4. 結果の表示 ---
st.header(f"📊 バックテスト結果 （期間: {start_year}年{start_month}月 〜 {end_year}年{end_month}月）")

col1, col2, col3, col4 = st.columns(4)
col1.metric("総トレード回数", f"{total_trades} 回")
col2.metric("勝率", f"{win_rate:.1f}%" if total_trades > 0 else "0%")
col3.metric("勝ち回数 / 負け回数", f"{win_count}勝 / {total_trades - win_count}敗")
col4.metric("累積リターン(合算)", f"{total_return:+.2f}%")

st.markdown("---")

# 資産推移のグラフ
if total_trades > 0:
    trades['Cumulative_Return'] = trades['Next_Return'].cumsum()
    st.subheader("📈 資産（累積リターン）の推移グラフ")
    st.line_chart(trades['Cumulative_Return'])
else:
    st.warning("条件に一致するトレードがこの期間にはありませんでした。条件を緩めてみてください。")

st.markdown("---")
st.subheader("📋 該当トレード履歴一覧")
if total_trades > 0:
    display_df = trades[['NQ_pct', 'SOX_pct', 'USD_pct', 'VIX_Close', 'Next_Return']].copy()
    display_df.columns = ['ナスダック変動(%)', 'SOX変動(%)', 'ドル円変動(%)', 'VIX値', '翌日リターン(%)']
    st.dataframe(display_df.sort_index(ascending=False))
else:
    st.info("データがありません。")