import streamlit as st
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import datetime

# --- データ取得用の関数 ---
def get_market_data(ticker_symbol, is_pct=True):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5d")
        if len(hist) < 2: return 0.0
        current_price = hist['Close'].iloc[-1]
        if not is_pct: return round(current_price, 2)
        prev_close = hist['Close'].iloc[-2]
        return round(((current_price - prev_close) / prev_close) * 100, 2)
    except:
        return 0.0

def check_us_macro_events():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(response.content)
        today_str = datetime.datetime.now().strftime("%m-%d-%Y")
        
        event_names = []
        for event in root.findall('event'):
            country = event.find('country').text
            impact = event.find('impact').text
            date_str = event.find('date').text
            if country == 'USD' and impact == 'High' and date_str == today_str:
                event_names.append(event.find('title').text)
        return len(event_names) > 0, event_names
    except:
        return False, ["※取得エラー：手動で確認してください"]

# --- 画面描画 ---
st.title("日経レバ 仕込み判定ダッシュボード")
st.write("15:00〜15:25の間に確認し、明日の寄り付きに向けた仕込みを判定します。")
st.markdown("---")

# --- 左側のメニュー（黄金バランスを初期値にしたカスタム調整） ---
st.sidebar.header("⚙️ 判定ルールの設定")
st.sidebar.write("バックテストで検証した黄金バランスを初期値にしています。")

p_nq = st.sidebar.slider("ナスダックの基準値(%)", min_value=-1.0, max_value=2.0, value=0.1, step=0.1)
p_vix = st.sidebar.slider("VIX(恐怖指数)の上限", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
p_usd = st.sidebar.slider("ドル円の許容下落幅(%)", min_value=-2.0, max_value=0.0, value=-0.5, step=0.1)
p_sox = st.sidebar.slider("SOX指数の基準値(%)", min_value=-2.0, max_value=3.0, value=0.1, step=0.1)

# --- データの取得 ---
with st.spinner("最新のマーケットデータと経済指標を取得中..."):
    has_macro_event, event_list = check_us_macro_events()
    nasdaq_pct = get_market_data("NQ=F", True)
    sox_pct = get_market_data("^SOX", True)
    usdjpy_pct = get_market_data("USDJPY=X", True)
    vix_value = get_market_data("^VIX", False)

# --- 1. 経済指標の自動取得表示 ---
st.header("1. 今夜の米国イベント（自動判定）")
if has_macro_event:
    st.error("🚨 【警告】今夜、以下の重要指標発表が予定されています！強制見送りとします。")
    for e in event_list:
        st.write(f"・ {e}")
else:
    st.success("🟢 今夜は相場を揺るがすような米国の重要指標（CPIやPCEなど）はありません。")

st.caption("※NVIDIAなど「超大型ハイテク株の決算」は自動取得できないため、予定がある場合は下のチェックを入れてください。")
is_tech_earnings = st.checkbox("今夜、メガテック企業の決算発表がある")
st.markdown("---")

# --- 2. マーケットデータの表示 ---
st.header("2. 現在のマーケット動向")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ナスダック先物", f"{nasdaq_pct:+.2f}%")
col2.metric("SOX(半導体)指数", f"{sox_pct:+.2f}%")
col3.metric("ドル円 前日比", f"{usdjpy_pct:+.2f}%")
col4.metric("VIX(恐怖指数)", f"{vix_value:.2f}")
st.markdown("---")

# --- 3. 総合判定結果 ---
st.header("🚦 本日の総合判定結果")

cond_nq = nasdaq_pct >= p_nq
cond_vix = vix_value < p_vix
cond_usd = usdjpy_pct > p_usd
cond_sox = sox_pct >= p_sox
cond_event = not (has_macro_event or is_tech_earnings)

if not cond_event:
    st.error("🚨 【総合判定：見送り】今夜は重要イベントがあります。指標ギャンブルを回避します。")
elif not cond_vix:
    st.error(f"🚨 【総合判定：見送り】VIXが {vix_value} と危険水域（{p_vix}以上）です。")
elif not cond_usd:
    st.warning(f"🟡 【総合判定：見送り】ドル円が {usdjpy_pct}% と許容下落幅（{p_usd}%）を超えて円高に振れています。")
elif not cond_sox:
    st.warning(f"🟡 【総合判定：見送り】SOX指数が {sox_pct}% と基準値（{p_sox}%）未満です。")
elif not cond_nq:
    st.warning(f"🟡 【総合判定：見送り】ナスダックが {nasdaq_pct}% と基準値（{p_nq}%）未満です。")
else:
    st.success("🟢 【総合判定：GO! 仕込み推奨】すべての安全フィルターをクリアしました！明日の朝9時に利確しましょう。")

st.markdown("---")
st.subheader("📋 各指標のクリア状況")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("ナスダック", f"{nasdaq_pct:+.2f}%", f"基準: ≧{p_nq}%")
    st.info("✅ クリア" if cond_nq else "❌ 未達")
with c2:
    st.metric("VIX指数", f"{vix_value:.2f}", f"基準: <{p_vix}")
    st.info("✅ クリア" if cond_vix else "❌ 超過")
with c3:
    st.metric("ドル円", f"{usdjpy_pct:+.2f}%", f"基準: >{p_usd}%")
    st.info("✅ クリア" if cond_usd else "❌ 未達")
with c4:
    st.metric("SOX指数", f"{sox_pct:+.2f}%", f"基準: ≧{p_sox}%")
    st.info("✅ クリア" if cond_sox else "❌ 未達")

st.markdown("---")
st.header("📊 各指標の詳細と解説（学習用）")

# --- 詳細解説セクション ---
st.subheader("① 米国重要イベント")
if not cond_event:
    st.error("❌ 本日の状態：重要イベントあり（危険）")
else:
    st.success("✅ 本日の状態：イベントなし（安全）")
st.write("**【なぜ見るの？】** CPIやPCE、巨大企業の決算などの発表直後は、プロの投資家でも予測不可能なギャンブル相場になります。過去の法則が一切通用しなくなるため、これを避けるのが勝率アップの絶対条件です。")

st.subheader("② VIX（恐怖指数）")
if cond_vix:
    st.success(f"✅ 本日の状態：{vix_value}（安全圏）")
else:
    st.error(f"❌ 本日の状態：{vix_value}（{p_vix}以上の危険水域）")
st.write("**【なぜ見るの？】** 投資家のパニック度合いを示します。通常は15前後ですが、数値が跳ね上がると相場のボラティリティ（価格の上下動）が激しくなり、夜間に突然の暴落が起きやすくなります。")

st.subheader("③ ドル円（為替）の前日比")
if cond_usd:
    st.success(f"✅ 本日の状態：{usdjpy_pct}%（基準クリア）")
else:
    st.warning(f"❌ 本日の状態：{usdjpy_pct}%（急激な円高）")
st.write("**【なぜ見るの？】** 日経平均を構成する主力企業は輸出企業が多いため、「円高」は利益が減るマイナス要因として嫌われます。米国株が上がっていても、円高が進んでいると日経平均は伸び悩みます。")

st.subheader("④ SOX（半導体）指数")
if cond_sox:
    st.success(f"✅ 本日の状態：{sox_pct}%（基準クリア）")
else:
    st.warning(f"❌ 本日の状態：{sox_pct}%（基準値未満）")
st.write("**【なぜ見るの？】** 現在の日経平均は、東京エレクトロンやアドバンテストといった「半導体関連株」の動きに指数全体が大きく引っ張られる構造になっています。日経を買うなら、その大黒柱である半導体が元気であることが重要です。")

st.subheader("⑤ ナスダック100先物")
if cond_nq:
    st.success(f"✅ 本日の状態：{nasdaq_pct}%（基準クリア）")
else:
    st.warning(f"❌ 本日の状態：{nasdaq_pct}%（基準値未満）")
st.write("**【なぜ見るの？】** 日経平均の翌朝のスタート位置を決める「メインエンジン」です。これが基準値以上の強い勢いを持っている時にだけ乗ることで、ダマシを減らし、勝てる確率の高い波だけを狙うことができます。")