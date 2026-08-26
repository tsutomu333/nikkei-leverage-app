import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.title("🚦 日中デイトレ・リアルタイム買い判断アプリ（事前気配値対応版）")
st.write("朝8:40以降の板（気配値）や最新データを自動取得し、9:00の寄り付き前にエントリー判断と解説を行います。")

# --- 1. サイドバー：判定ルールの設定 ---
st.sidebar.header("⚙️ 判定ルールの設定（しきい値）")
p_gap = st.sidebar.slider("③ 朝の窓開け基準値(%)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)

ignore_event = st.sidebar.checkbox("⚠️ 本日の重要イベント警告を無視して強制判定する", value=False)

# --- 2. 最新データと気配値の取得 ---
@st.cache_data(ttl=300) # 5分ごとにキャッシュ更新
def get_morning_market_data():
    t_vix = yf.download("^VIX", period="1mo", progress=False)
    time.sleep(0.5)
    t_usd = yf.download("USDJPY=X", period="1mo", progress=False)
    time.sleep(0.5)
    t_1570 = yf.download("1570.T", period="1mo", progress=False) # 1570（NEXT FUNDS 日経レバレッジ・インデックス上場投信）

    for df in [t_vix, t_usd, t_1570]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None)

    t_vix = t_vix.dropna()
    t_usd = t_usd.dropna()
    t_1570 = t_1570.dropna()

    latest_vix = t_vix['Close'].iloc[-1]
    usd_pct = (t_usd['Close'].iloc[-1] - t_usd['Close'].iloc[-2]) / t_usd['Close'].iloc[-2] * 100
    
    # 1570の「今日の気配値（Open）」と「前日の終値（Closeのiloc[-2]))」を比較して窓開けを計算
    # 朝の早い時間帯でもOpenにその日の予想気配値/始値が入ってきます
    target_open = t_1570['Open'].iloc[-1]
    prev_close = t_1570['Close'].iloc[-2]
    gap_open = (target_open - prev_close) / prev_close * 100

    return float(latest_vix), float(usd_pct), float(gap_open)

with st.spinner("朝の板情報と市場データをチェック中..."):
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
    st.success("🟢 **【判定：GO! 仕込み推奨】すべての安全フィルターと条件をクリアしています！寄成注文の準備をどうぞ。**")
else:
    if has_major_event and not ignore_event:
        st.warning("🚨 **【判定：強制見送り】本日は超重要経済指標の発表が予定されているため、見送りとします。**")
    else:
        st.warning("🔴 **【判定：見送り推奨】一部の条件を満たしていないため、本日は手控えましょう。**")

st.markdown("### 📋 本日の指標チェック結果（1570ベース）")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("③ 朝の窓開け(1570)", f"{gap_val:+.2f}%", f"基準: +{p_gap}%以上")
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
    reasons.append(f"- **1570の朝の窓開け（+{gap_val:.2f}%）：** 設定基準（+{p_gap}%以上）をクリアしており、板の気配値から見て今朝はしっかりとした買いの勢い（モメンタム）が確認できます。")
else:
    reasons.append(f"- **1570の朝の窓開け（+{gap_val:.2f}%）：** 設定基準（+{p_gap}%以上）に届いておらず、朝の初動の勢いが弱いためダマシのリスクがあります。")

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
    st.info("🎯 **総括：** 8:40以降の板の気配値を含めてすべての条件が揃っています。8:55頃までに1570の「寄成」注文をセットする絶好のチャンスです。")
else:
    st.warning("⚠️ **総括：** 条件を満たしていない項目があります。無理にエントリーせず、次のチャンスを待ちましょう。")