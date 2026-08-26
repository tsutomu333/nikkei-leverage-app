import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.title("日経レバ 持ち越し判定アプリ（リアルタイム・ダッシュボード）")
st.write("今日の夕方15:25時点の市場データが、検証済みの『勝ちパターン』を満たしているかを判定します。")

# --- データの取得（直近の動きを見るため短期間取得） ---
@st.cache_data(ttl=60) # 60秒ごとにキャッシュを更新
def get_latest_data():
    t_nq = yf.Ticker("NQ=F").history(period="5d")
    t_sox = yf.Ticker("^SOX").history(period="5d")
    t_usd = yf.Ticker("USDJPY=X").history(period="5d")
    t_vix = yf.Ticker("^VIX").history(period="5d")

    t_nq.index = t_nq.index.tz_localize(None)
    t_sox.index = t_sox.index.tz_localize(None)
    t_usd.index = t_usd.index.tz_localize(None)
    t_vix.index = t_vix.index.tz_localize(None)

    df = pd.DataFrame({
        'NQ_Close': t_nq['Close'],
        'SOX_Close': t_sox['Close'],
        'USD_Close': t_usd['Close'],
        'VIX_Close': t_vix['Close']
    }).dropna()

    df['NQ_pct'] = df['NQ_Close'].pct_change() * 100
    df['SOX_pct'] = df['SOX_Close'].pct_change() * 100
    df['USD_pct'] = df['USD_Close'].pct_change() * 100

    return df.iloc[-1] # 一番直近（今日）のデータを取得

try:
    today_data = get_latest_data()
except Exception as e:
    st.error(f"データの取得に失敗しました: {e}")
    st.stop()

# --- 左側のメニュー（判定基準の調整 / バックテストの黄金バランスを初期値に） ---
st.sidebar.header("⚙️ 判定ルールの設定")
st.sidebar.write("バックテストで検証した基準を初期値にしています。必要に応じて微調整可能です。")

p_nq = st.sidebar.slider("ナスダックの基準値(%)", min_value=-1.0, max_value=2.0, value=0.1, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)
p_sox = st.sidebar.slider("SOX指数の基準値(%)", min_value=-2.0, max_value=3.0, value=0.1, step=0.1)

# --- 今日の数値の取り出し ---
nq_val = today_data['NQ_pct']
vix_val = today_data['VIX_Close']
usd_val = today_data['USD_pct']
sox_val = today_data['SOX_pct']

# --- 条件の判定 ---
cond_nq = nq_val >= p_nq
cond_vix = vix_val < p_vix
cond_usd = usd_val > p_usd
cond_sox = sox_val >= p_sox

is_all_clear = cond_nq and cond_vix and cond_usd and cond_sox

# --- 画面への表示 ---
st.header("🚦 本日のエントリー判定結果")

if is_all_clear:
    st.success("【 判定：買いシグサル点灯 (GO) 】すべての条件クリア！今日の夕方に日経レバを買い、翌朝に売りましょう。")
else:
    st.warning("【 判定：見送り (NO GO) 】条件をクリアしていない項目があります。本日はトレードを控えましょう。")

st.markdown("---")

st.subheader("📋 各指標のクリア状況")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ナスダック変動率", f"{nq_val:+.2f}%", f"基準: ≧{p_nq}%")
    if cond_nq:
        st.info("✅ クリア")
    else:
        st.error("❌ 未達")

with col2:
    st.metric("VIX（恐怖指数）", f"{vix_val:.2f}", f"基準: <{p_vix}")
    if cond_vix:
        st.info("✅ クリア")
    else:
        st.error("❌ 超過（危険）")

with col3:
    st.metric("ドル円変動率", f"{usd_val:+.2f}%", f"基準: >{p_usd}%")
    if cond_usd:
        st.info("✅ クリア")
    else:
        st.error("❌ 下落オーバー")

with col4:
    st.metric("SOX指数変動率", f"{sox_val:+.2f}%", f"基準: ≧{p_sox}%")
    if cond_sox:
        st.info("✅ クリア")
    else:
        st.error("❌ 未達")

st.markdown("---")
st.caption(f"最終データ更新タイミング（直近営業日）： {today_data.name.strftime('%Y-%m-%d %H:%M')}")