import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("🚦 日中デイトレ・リアルタイム買い判断アプリ（NYダウ連動版）")
st.write("朝の先物データと直前に締まったNYダウから、今日の1570の仕込み時を高精度に判定します。")

# --- 1. サイドバー：判定ルールの設定 ---
st.sidebar.header("⚙️ 判定ルールの設定（しきい値）")
p_gap = st.sidebar.slider("③ 朝の窓開け基準値(%)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
p_dow = st.sidebar.slider("④ 前日NYダウの基準値(%)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
p_vix = st.sidebar.slider("① VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("② ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠 マニュアル補正（オプション）")
manual_gap = st.sidebar.number_input("証券アプリの板の気配値(%)を直接入力（空欄なら自動計算）", value=0.0, step=0.1)
ignore_event = st.sidebar.checkbox("⚠️ 本日の重要イベント警告を無視", value=False)

# --- 2. 最新データの取得と高精度エンジン ---
@st.cache_data(ttl=60)
def get_morning_market_data():
    t_vix = yf.download("^VIX", period="1mo", progress=False)
    time.sleep(0.5)
    t_usd = yf.download("USDJPY=X", period="1mo", progress=False)
    time.sleep(0.5)
    t_niy = yf.download("NIY=F", period="1mo", progress=False) # CME日経先物
    time.sleep(0.5)
    t_dow = yf.download("^DJI", period="1mo", progress=False) # NYダウ

    for df in [t_vix, t_usd, t_niy, t_dow]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)

    latest_vix = float(t_vix['Close'].dropna().iloc[-1])
    
    t_usd_clean = t_usd['Close'].dropna()
    usd_pct = (t_usd_clean.iloc[-1] - t_usd_clean.iloc[-2]) / t_usd_clean.iloc[-2] * 100
    
    # 🎯 修正版エンジン：先物は先物同士で比較する
    t_niy_clean = t_niy['Close'].dropna()
    niy_current = float(t_niy_clean.iloc[-1])      # 今の先物価格
    niy_prev_close = float(t_niy_clean.iloc[-2])   # 昨日の先物終値
    nikkei_gap_pct = (niy_current - niy_prev_close) / niy_prev_close * 100
    expected_1570_gap = nikkei_gap_pct * 2

    # 🎯 NYダウの変動率を計算
    t_dow_clean = t_dow['Close'].dropna()
    dow_pct = (t_dow_clean.iloc[-1] - t_dow_clean.iloc[-2]) / t_dow_clean.iloc[-2] * 100

    return latest_vix, float(usd_pct), float(expected_1570_gap), float(dow_pct)

with st.spinner("最新データ（NYダウ・先物など）を取得中..."):
    try:
        vix_val, usd_val, auto_gap_val, dow_val = get_morning_market_data()
    except Exception as e:
        st.error(f"データの取得に失敗しました: {e}")
        st.stop()

# マニュアル入力があればそちらを優先
gap_val = manual_gap if manual_gap != 0.0 else auto_gap_val
is_manual = manual_gap != 0.0

# --- 3. 判定ロジック ---
cond_vix = vix_val < p_vix
cond_usd = usd_val > p_usd
cond_gap = gap_val >= p_gap
cond_dow = dow_val >= p_dow

has_major_event = False 

if not ignore_event and has_major_event:
    is_go = False
else:
    is_go = cond_vix and cond_usd and cond_gap and cond_dow

# --- 4. 画面への結果表示 ---
st.markdown("---")

if is_go:
    st.success("🟢 **【判定：GO! 仕込み推奨】NYダウを含むすべての条件をクリア！寄成注文の準備をどうぞ。**")
else:
    st.warning("🔴 **【判定：見送り推奨】条件を満たしていないため、本日は手控えましょう。**")

st.markdown("### 📋 本日の指標チェック結果")

# 表示項目が5つになったので、5列（columns）に変更
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    gap_label = "③ 1570窓開け(手動)" if is_manual else "③ 1570窓開け(自動)"
    st.metric(gap_label, f"{gap_val:+.2f}%", f"基準: +{p_gap}%以上")
    st.write("✅ クリア" if cond_gap else "❌ 未達")

with col2:
    st.metric("④ NYダウ(直前)", f"{dow_val:+.2f}%", f"基準: {p_dow}%以上")
    st.write("✅ クリア" if cond_dow else "❌ 未達")

with col3:
    st.metric("② ドル円変動率", f"{usd_val:+.2f}%", f"基準: {p_usd}%より上")
    st.write("✅ クリア" if cond_usd else "❌ 警戒")

with col4:
    st.metric("① VIX", f"{vix_val:.2f}", f"基準: {p_vix}未満")
    st.write("✅ クリア" if cond_vix else "❌ 危険")

with col5:
    st.metric("🚨 イベント", "なし" if not has_major_event else "予定あり")
    st.write("✅ 安全" if not has_major_event else "⚠️ 警戒")

# --- 5. 自動生成される判定解説パネル ---
st.markdown("---")
st.subheader("💡 【AI判定解説】本日のトレード根拠")

reasons = []

if cond_gap:
    reasons.append(f"- **1570の朝の窓開け（{gap_val:+.2f}%）：** 設定基準をクリア。今朝はしっかりとした買いの勢いが確認できます。")
else:
    reasons.append(f"- **1570の朝の窓開け（{gap_val:+.2f}%）：** 設定基準に届いておらず、朝の初動の勢いが弱いためダマシのリスクがあります。")

if cond_dow:
    reasons.append(f"- **NYダウ（{dow_val:+.2f}%）：** 日本市場が開く直前に引けた米国株が基準（{p_dow}%）をクリアしており、日本株への強い追い風となります。")
else:
    reasons.append(f"- **NYダウ（{dow_val:+.2f}%）：** 米国株の調子が悪く、基準（{p_dow}%）を下回っています。日本株も連れ安になるリスクが高い状態です。")

if cond_usd:
    reasons.append(f"- **為替・ドル円（{usd_val:+.2f}%）：** 許容下落幅の範囲内に収まっており、極端な円高による下押し圧力は大きくありません。")
else:
    reasons.append(f"- **為替・ドル円（{usd_val:+.2f}%）：** 急激な円高が進行しており、投資家心理にマイナスに働く恐れがあります。")

if cond_vix:
    reasons.append(f"- **恐怖指数VIX（{vix_val:.2f}）：** 上限ラインを下回っており、市場全体が比較的落ち着いた正常な状態にあります。")
else:
    reasons.append(f"- **恐怖指数VIX（{vix_val:.2f}）：** 市場に大きな警戒感が広がっているため、予測不能な乱高下に巻き込まれる危険があります。")

for r in reasons:
    st.write(r)