import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

st.set_page_config(page_title="B_翌朝寄り付き研究", page_icon="🔬", layout="wide")

JST = ZoneInfo("Asia/Tokyo")

TICKERS = {
    "1570": "1570.T",
    "日経225": "^N225",
    "TOPIX": "^TOPX",
    "NASDAQ100": "NQ=F",
    "S&P500": "^GSPC",
    "SOX": "^SOX",
    "NYダウ": "^DJI",
    "VIX": "^VIX",
    "ドル円": "USDJPY=X",
    "米10年金利": "^TNX",
}

FEATURE_LABELS = {
    "n225_ret": "前日日経225騰落率",
    "n225_close_pos": "前日日経225終値位置",
    "topix_ret": "前日TOPIX騰落率",
    "nq_ret": "前回米国NASDAQ100騰落率",
    "sp_ret": "前回米国S&P500騰落率",
    "sox_ret": "前回米国SOX騰落率",
    "dow_ret": "前回米国NYダウ騰落率",
    "vix_level": "前回米国VIX水準",
    "usd_ret": "前回ドル円騰落率",
    "tnx_ret": "前回米10年金利変化率",
    "us_breadth": "米国4指数プラス数",
    "nq_usd_combo": "NASDAQ100騰落率＋ドル円騰落率",
}

@st.cache_data(ttl=900)
def download_daily(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")

def series_close(df):
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["Close"], errors="coerce").dropna()

def build_research(start, end):
    # The backtest deliberately uses only data that was already available
    # by the Japanese cash-session close. US features are the PREVIOUS US
    # session, never the same day's US close (look-ahead avoidance).
    raw = {k: download_daily(v, start, end) for k, v in TICKERS.items()}

    jp = pd.DataFrame(index=raw["1570"].index)
    etf = raw["1570"]
    n225 = raw["日経225"]
    topix = raw["TOPIX"]

    for c in ["Open", "High", "Low", "Close"]:
        if c in etf.columns:
            jp[f"etf_{c.lower()}"] = pd.to_numeric(etf[c], errors="coerce")

    jp["n225_close"] = series_close(n225)
    if not n225.empty:
        jp["n225_prev_close"] = jp["n225_close"].shift(1)
        jp["n225_ret"] = jp["n225_close"].pct_change() * 100
        high = pd.to_numeric(n225.get("High"), errors="coerce")
        low = pd.to_numeric(n225.get("Low"), errors="coerce")
        jp["n225_close_pos"] = ((jp["n225_close"] - low) / (high - low)).replace([np.inf, -np.inf], np.nan)

    jp["topix_ret"] = series_close(topix).pct_change() * 100

    us_map = {
        "nq_ret": "NASDAQ100", "sp_ret": "S&P500",
        "sox_ret": "SOX", "dow_ret": "NYダウ"
    }
    for feat, key in us_map.items():
        s = series_close(raw[key])
        # shift(1): the last completed US session before Japanese date D.
        jp[feat] = s.pct_change().shift(1) * 100

    vix = series_close(raw["VIX"])
    usd = series_close(raw["ドル円"])
    tnx = series_close(raw["米10年金利"])

    jp["vix_level"] = vix.shift(1)
    jp["usd_ret"] = usd.pct_change().shift(1) * 100
    jp["tnx_ret"] = tnx.pct_change().shift(1) * 100

    jp["us_breadth"] = sum([
        (jp["nq_ret"] > 0).astype(int),
        (jp["sp_ret"] > 0).astype(int),
        (jp["sox_ret"] > 0).astype(int),
        (jp["dow_ret"] > 0).astype(int),
    ])
    jp["nq_usd_combo"] = jp["nq_ret"] + jp["usd_ret"]

    # Target: buy 1570 at today's close, sell at next trading day's open.
    jp["target_next_open_ret"] = (jp["etf_open"].shift(-1) / jp["etf_close"] - 1) * 100
    jp["target_up"] = jp["target_next_open_ret"] > 0

    return jp.dropna(subset=["etf_close", "etf_open", "target_next_open_ret"])

def screen_features(df, min_n):
    rows = []
    for feat, label in FEATURE_LABELS.items():
        if feat not in df.columns:
            continue
        s = df[feat].dropna()
        if len(s) < min_n:
            continue

        for q, direction, symbol in [
            (0.20, "low", "≤"),
            (0.30, "low", "≤"),
            (0.40, "low", "≤"),
            (0.60, "high", "≥"),
            (0.70, "high", "≥"),
            (0.80, "high", "≥"),
        ]:
            threshold = s.quantile(q)
            if direction == "low":
                mask = df[feat] <= threshold
            else:
                mask = df[feat] >= threshold
            x = df.loc[mask, "target_next_open_ret"].dropna()
            if len(x) < min_n:
                continue
            mean_ret = x.mean()
            up_rate = (x > 0).mean() * 100
            rows.append({
                "情報": label,
                "条件": f"{symbol} {threshold:.3f}" + ("以下" if direction=="low" else "以上"),
                "件数": len(x),
                "上昇率": up_rate,
                "平均リターン": mean_ret,
                "中央値": x.median(),
                "最大勝ち": x.max(),
                "最大負け": x.min(),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["スコア"] = out["平均リターン"] * np.sqrt(out["件数"])
    return out.sort_values(["平均リターン", "上昇率"], ascending=False)

def calc_stats(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return {}
    wins = x[x > 0]
    losses = x[x <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {
        "n": len(x),
        "win_rate": (x > 0).mean() * 100,
        "mean": x.mean(),
        "median": x.median(),
        "pf": pf,
        "cum": ((1 + x/100).prod() - 1) * 100,
    }

st.title("🔬 翌朝寄り付き研究アプリ")
st.caption("「前日15:25時点で何が翌朝の1570寄り付きに効くのか」を探すための研究用アプリです。")
st.info("⚠️ 重要：バックテストは未来情報を使わない設計です。米国株は「その日本日の前に終了した米国セッション」を使用します。15:25時点の米国先物リアルタイム値を長期間バックテストするには、別の高頻度ヒストリカルデータが必要です。")

with st.sidebar:
    st.header("🔧 研究条件")
    start_date = st.date_input("開始日", datetime(2018,1,1).date())
    end_date = st.date_input("終了日", datetime.now(JST).date())
    min_n = st.number_input("最低サンプル数", min_value=20, max_value=200, value=40, step=10)
    run = st.button("🔄 研究データを取得", type="primary")

if run or "research_df" not in st.session_state:
    with st.spinner("過去データを取得・整形しています…"):
        try:
            df = build_research(start_date.isoformat(), end_date.isoformat())
            st.session_state["research_df"] = df
        except Exception as e:
            st.error(f"データ取得・計算でエラーが発生しました：{e}")
            st.stop()
else:
    df = st.session_state["research_df"]

if df.empty:
    st.error("データが取得できませんでした。期間やYahoo Finance側の提供状況を確認してください。")
    st.stop()

st.subheader("① まず全体像")
stats = calc_stats(df["target_next_open_ret"])
c = st.columns(5)
c[0].metric("対象件数", f"{stats['n']:,}件")
c[1].metric("翌朝上昇率", f"{stats['win_rate']:.1f}%")
c[2].metric("平均寄り付きリターン", f"{stats['mean']:+.3f}%")
c[3].metric("中央値", f"{stats['median']:+.3f}%")
c[4].metric("複利", f"{stats['cum']:+.2f}%")

st.caption("対象は「1570を日本市場の終値で買い、次の営業日の寄り付きで売る」単純モデル。手数料・スリッページ・税金は未反映。")

st.subheader("② 何が効いている？")
result = screen_features(df, int(min_n))
if result.empty:
    st.warning("条件を満たす分析結果がありません。最低サンプル数を下げてください。")
else:
    display = result.copy()
    for col in ["上昇率"]:
        display[col] = display[col].map(lambda x: f"{x:.1f}%")
    for col in ["平均リターン", "中央値", "最大勝ち", "最大負け"]:
        display[col] = display[col].map(lambda x: f"{x:+.3f}%")
    display["スコア"] = result["スコア"].map(lambda x: f"{x:+.3f}")
    st.dataframe(display.head(30), use_container_width=True, hide_index=True)

    best = result.iloc[0]
    st.success(f"現時点の上位候補：**{best['情報']} / {best['条件']}** → 件数 {int(best['件数'])}、上昇率 {best['上昇率']:.1f}%、平均リターン {best['平均リターン']:+.3f}%")

st.subheader("③ 候補条件を選んで詳しく見る")
labels = [v for v in FEATURE_LABELS.values()]
label_to_feat = {v:k for k,v in FEATURE_LABELS.items()}
selected_label = st.selectbox("分析する情報", labels)
feat = label_to_feat[selected_label]
q = st.slider("条件の分位点", 0.05, 0.95, 0.70, 0.05)
direction = st.radio("条件方向", ["以上", "以下"], horizontal=True)
threshold = df[feat].quantile(q if direction=="以上" else 1-q)
mask = df[feat] >= threshold if direction=="以上" else df[feat] <= threshold
x = df.loc[mask, "target_next_open_ret"]
s2 = calc_stats(x)
if s2:
    cc = st.columns(5)
    cc[0].metric("条件該当数", f"{s2['n']:,}")
    cc[1].metric("上昇率", f"{s2['win_rate']:.1f}%")
    cc[2].metric("平均", f"{s2['mean']:+.3f}%")
    cc[3].metric("中央値", f"{s2['median']:+.3f}%")
    cc[4].metric("Profit Factor", f"{s2['pf']:.2f}" if np.isfinite(s2["pf"]) else "—")
    st.write(f"**判定閾値：{threshold:.4f}**")
    chart = pd.DataFrame({"翌朝1570寄り付きリターン": x.values})
    st.line_chart(chart)

st.subheader("④ 時系列で確認")
st.line_chart(df.set_index(df.index)["target_next_open_ret"].rolling(20).mean())

st.subheader("⑤ 研究上の注意")
st.markdown("""
- **相関と因果は別物**です。過去に効いたから将来も効くとは限りません。
- 条件を何百通りも試して「一番良いもの」を選ぶと、過学習になりやすいです。
- 本番採用前に、期間を分けた**アウト・オブ・サンプル検証**を行います。
- まずは「何が効くか」を発見し、その後にAアプリの判定条件へ戻します。
- 目標は勝率ではなく、**翌朝リターンの期待値が安定してプラスになる情報**を見つけることです。
""")

st.caption(f"最終更新表示：{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')} JST")
