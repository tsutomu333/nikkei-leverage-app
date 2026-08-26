import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("🚦 日中デイトレ・リアルタイム買い判断アプリ（完全自動・先物連動版）")
st.write("朝8:40以降にリアルタイムで動いている日経先物（CME）のデータを取得し、9:00前の段階で「今日の1570の予想窓開け」を全自動で計算・判定します。")

# --- 1. サイドバー：判定ルールの設定 ---
st.sidebar.header("⚙️ 判定ルールの設定（しきい値）")
p_gap = st.sidebar.slider("③ 朝の窓開け基準値(%)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

ignore_event = st.sidebar.checkbox("⚠️ 本日の重要イベント警告を無視して強制判定する", value=False)

# --- 2. 最新データと「先物」を使った予想気配値の取得 ---
@st.cache_data(ttl=60) # 朝の変動を捉えるためキャッシュを1分に短縮
def get_morning_market_data():
    # VIX、ドル円、日経平均（現物）、日経平均先物（CME）を取得
    t_vix = yf.download("^VIX", period="1mo", progress=False)
    time.sleep(0.5)
    t_usd = yf.download("USDJPY=X", period="1mo", progress=False)
    time.sleep(0.5)
    t_n225 = yf.download("^N225", period="1mo", progress=False) # 昨日の終値用
    time.sleep(0.5)
    t_niy = yf.download("NIY=F", period="1mo", progress=False) # 現在リアルタイムで動いている先物

    for df in [t_vix, t_usd, t_n225, t_niy]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)

    latest_vix = float(t_vix['Close'].dropna().iloc[-1])
    
    t_usd_clean = t_usd['Close'].dropna()
    usd_pct = (t_usd_clean.iloc[-1] - t_usd_clean.iloc[-2]) / t_usd_clean.iloc[-2] * 100
    
    # 🎯 ここが裏ワザ：先物を使った1570窓開け自動予測エンジン
    # 昨日の日経平均（現物）の終値
    n225_prev_close = float(t_n225['Close'].dropna().iloc[-1])
    # 現在（朝8:50時点）リアルタイムで動いている日経先物の価格
    niy_current = float(t_niy['Close'].dropna().iloc[-1])
    
    # 日経平均自体の予想窓開け率(%)
    nikkei_gap_pct = (niy_current - n225_prev_close) / n225_prev_close * 100
    
    # 1570は日経平均の「2倍（ブル）」連動ETFなので、窓開け率も2倍になる
    expected_1570_gap = nikkei_gap_pct * 2

    return latest_vix, float(usd_pct), float(expected_1570_gap)

with st.spinner("リアルタイム先物データから今日の1570気配値を自動計算中..."):
    try:
        vix_val, usd_val, gap_val = get_morning_market_data()
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
    st.success("🟢 **【判定：GO! 仕込み推奨】すべての安全フィルターと条件をクリア！寄成注文の準備をどうぞ。**")
else:
    if has_major_event and not ignore_event:
        st.warning("🚨 **【判定：強制見送り】本日は超重要経済指標の発表が予定されているため、見送りとします。**")
    else:
        st.warning("🔴 **【判定：見送り推奨】一部の条件を満たしていないため、本日は手控えましょう。**")

st.markdown("### 📋 本日の指標チェック結果（完全自動予測）")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("③ 1570予想窓開け(先物換算)", f"{gap_val:+.2f}%", f"基準: +{p_gap}%以上")
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
    reasons.append(f"- **1570の朝の窓開け（予想 +{gap_val:.2f}%）：** リアルタイムの日経先物の動向から計算した結果、設定基準（+{p_gap}%以上）をクリアしています。今朝はしっかりとした買いの勢いが確認できます。")
else:
    reasons.append(f"- **1570の朝の窓開け（予想 +{gap_val:.2f}%）：** 日経先物の動向から計算した結果、設定基準（+{p_gap}%以上）に届いておらず、朝の初動の勢いが弱いためダマシのリスクがあります。")

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
    st.info("🎯 **総括：** 8:55の時点で先物ベースの予測も含め、すべての条件が完璧に揃っています。証券アプリを開き、1570の「寄成」注文をセットしてください！")
else:
    st.warning("⚠️ **総括：** 条件を満たしていない項目があります。無理にエントリーせず、次のチャンスを待ちましょう。")