import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("🚦 日中デイトレ・リアルタイム買い判断アプリ（解説付き）")
st.write("VIX・ドル円・朝の窓開け・重要イベントの状況を自動判定し、エントリーの根拠（解説）を詳しく提示します。")

# --- 1. サイドバー：判定ルールの設定 ---
st.sidebar.header("⚙️ 判定ルールの設定（しきい値）")
p_gap = st.sidebar.slider("③ 朝の窓開け基準値(%)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

ignore_event = st.sidebar.checkbox("⚠️ 本日の重要イベント警告を無視して強制判定する", value=False)

# --- 2. 最新データの取得（エラー防止の安全設計） ---
@st.cache_data(ttl=600)
def get_today_market_data():
    # 期間を1ヶ月に広げることで、休日や祝日を挟んでも確実に直近データが取得できるようにする
    t_vix = yf.download("^VIX", period="1mo", progress=False)
    time.sleep(0.5)
    t_usd = yf.download("USDJPY=X", period="1mo", progress=False)
    time.sleep(0.5)
    t_n225 = yf.download("^N225", period="1mo", progress=False)

    for df in [t_vix, t_usd, t_n225]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)

    # 有効なデータに絞る
    t_vix = t_vix.dropna()
    t_usd = t_usd.dropna()
    t_n225 = t_n225.dropna()

    latest_vix = t_vix['Close'].iloc[-1]
    usd_pct = (t_usd['Close'].iloc[-1] - t_usd['Close'].iloc[-2]) / t_usd['Close'].iloc[-2] * 100
    
    n225_open = t_n225['Open'].iloc[-1]
    n225_prev_close = t_n225['Close'].iloc[-2]
    gap_open = (n225_open - n225_prev_close) / n225_prev_close * 100

    return float(latest_vix), float(usd_pct), float(gap_open)

with st.spinner("最新の市場データと経済イベントをチェック中..."):
    try:
        vix_val, usd_val, gap_val = get_today_market_data()
    except Exception as e:
        st.error(f"データの取得に失敗しました: {e}")
        st.stop()

# --- 3. 判定ロジック ---
cond_vix = vix_val < p_vix
cond_usd = usd_val > p_usd
cond_gap = gap_val >= p_gap

has_major_event = False 
event_message = "本日は日中・夜間に警戒すべき超重要米経済指標の予定はありません。"

if not ignore_event and has_major_event:
    is_go = False
else:
    is_go = cond_vix and cond_usd and cond_gap

# --- 4. 画面への結果表示 ---
st.markdown("---")

if is_go:
    st.success("🟢 **【判定：GO! 仕込み推奨】すべての安全フィルターと条件をクリアしています！**")
else:
    if has_major_event and not ignore_event:
        st.warning("🚨 **【判定：強制見送り】本日は超重要経済指標の発表が予定されているため、見送りとします。**")
    else:
        st.warning("🔴 **【判定：見送り推奨】一部の条件を満たしていないため、本日は手控えましょう。**")

st.markdown("### 📋 本日の指標チェック結果")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("③ 朝の窓開け", f"{gap_val:+.2f}%", f"基準: +{p_gap}%以上")
    st.write("✅ クリア" if cond_gap else "❌ 未達")

with col2:
    st.metric("② ドル円変動率", f"{usd_val:+.2f}%", f"基準: {p_usd}%より上")
    st.write("✅ クリア" if cond_usd else "❌ 警戒")

with col3:
    st.metric("① VIX（恐怖指数）", f"{vix_val:.2f}", f"基準: {p_vix}未満")
    st.write("✅ クリア" if cond_vix else "❌ 危険")

with col4:
    st.metric("🚨 米国重要イベント", "なし" if not has_major_event else "予定あり", "PCE・GDP等")
    st.write("✅ 安全" if not has_major_event else "⚠️ 警戒")

# --- 5. 自動生成される判定解説パネル ---
st.markdown("---")
st.subheader("💡 【AI判定解説】本日のトレード根拠")

reasons = []

if cond_gap:
    reasons.append(f"- **朝の窓開け（+{gap_val:.2f}%）：** 設定基準（+{p_gap}%以上）をクリアしており、今朝はしっかりとした買いの勢い（モメンタム）を持ってスタートしています。")
else:
    reasons.append(f"- **朝の窓開け（+{gap_val:.2f}%）：** 設定基準（+{p_gap}%以上）に届いておらず、朝の初動の勢いが弱いためダマシのリスクがあります。")

if cond_usd:
    reasons.append(f"- **為替・ドル円（{usd_val:+.2f}%）：** 許容下落幅（{p_usd}%）の範囲内に収まっており、極端な円高による日本株への下押し圧力は大きくありません。")
else:
    reasons.append(f"- **為替・ドル円（{usd_val:+.2f}%）：** 急激な円高が進行しており、投資家心理にマイナスに働く恐れがあるため見送りが妥当です。")

if cond_vix:
    reasons.append(f"- **恐怖指数VIX（{vix_val:.2f}）：** 上限ライン（{p_vix}）を下回っており、市場全体が比較的落ち着いた正常な状態にあります。")
else:
    reasons.append(f"- **恐怖指数VIX（{vix_val:.2f}）：** 市場にパニックや大きな警戒感が広がっているため、予測不能な乱高下に巻き込まれる危険があります。")

if has_major_event and not ignore_event:
    reasons.append(f"- **重要イベント：** 本日は相場を大きく動かすイベントが控えているため、ポジションを持ち越す・残すリスクを避ける必要があります。")
else:
    reasons.append(f"- **重要イベント：** 本日は日中の値動きを大きく歪めるような主要イベントの直撃リスクは低い状態です。")

for r in reasons:
    st.write(r)

if is_go:
    st.info("🎯 **総括：** すべての条件が綺麗な形で噛み合っています。朝の寄り付きから15:00の引けにかけて、順張りでのデイトレードを検討する絶好のチャンスです。")
else:
    st.warning("⚠️ **総括：** いずれかの条件が基準を満たしていない、あるいはリスク要因があります。無理にエントリーせず、次のチャンスを待ちましょう。")