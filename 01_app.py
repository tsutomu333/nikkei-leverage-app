import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="日経レバ 1泊判定", page_icon="📈", layout="wide")
JST = ZoneInfo("Asia/Tokyo")

st.title("📈 日経レバ 1泊トレード判定")
st.caption("前日の大引けで1570を買い、翌営業日の寄り付きで売る戦略。未来の情報を使わないことを最優先に設計。")

@st.cache_data(ttl=300)
def yf_daily(ticker, period="10y"):
    try:
        x = yf.download(ticker, period=period, auto_adjust=False,
                        progress=False, threads=False)
        if x.empty: return pd.DataFrame()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x.dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def yf_intraday(ticker, period="1d", interval="5m"):
    try:
        x = yf.download(ticker, period=period, interval=interval,
                        auto_adjust=False, progress=False, threads=False)
        if x.empty: return pd.DataFrame()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_convert(JST)
        return x.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def latest_price(ticker):
    x = yf_intraday(ticker)
    if x.empty: return np.nan, None
    x = x.dropna(subset=["Close"])
    if x.empty: return np.nan, None
    return float(x["Close"].iloc[-1]), x.index[-1]

def latest_daily_change(ticker):
    x = yf_daily(ticker, "5d")
    if len(x) < 2: return np.nan
    return float((x["Close"].iloc[-1] / x["Close"].iloc[-2] - 1) * 100)

@st.cache_data(ttl=300)
def macro_events():
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        today = datetime.now(JST).strftime("%m-%d-%Y")
        events = []
        for e in root.findall("event"):
            if (e.findtext("country","") == "USD"
                and e.findtext("impact","") == "High"
                and e.findtext("date","") == today):
                events.append(e.findtext("title",""))
        return True, events
    except Exception as exc:
        return False, [str(exc)]

# ---- settings ----
st.sidebar.header("⚙️ 判定条件")
p_nq = st.sidebar.number_input("NASDAQ100先物 基準(%)", -3.0, 5.0, 0.1, 0.1)
p_sox = st.sidebar.number_input("SOX 基準(%)", -3.0, 5.0, 0.1, 0.1)
p_usd = st.sidebar.number_input("ドル円 前日比の下限(%)", -3.0, 1.0, -0.5, 0.1)
p_vix = st.sidebar.number_input("VIX 上限", 10.0, 50.0, 20.0, 0.5)

# ---- live ----
st.header("① 本日の判定")
st.caption(f"日本時間 {datetime.now(JST):%Y-%m-%d %H:%M:%S}")

with st.spinner("最新データを取得中..."):
    nq, nq_ts = latest_price("NQ=F")
    sox, sox_ts = latest_price("^SOX")
    usd, usd_ts = latest_price("JPY=X")
    fx5 = yf_daily("JPY=X", "5d")
    vix5 = yf_daily("^VIX", "5d")
    ok_event, events = macro_events()

prev_fx = float(fx5["Close"].iloc[-2]) if len(fx5) >= 2 else np.nan
usd_pct = (usd / prev_fx - 1) * 100 if not np.isnan(usd) and prev_fx else np.nan
vix = float(vix5["Close"].iloc[-1]) if not vix5.empty else np.nan

c = st.columns(4)
c[0].metric("NASDAQ100先物", "—" if np.isnan(nq) else f"{nq:+.2f}%")
c[1].metric("SOX", "—" if np.isnan(sox) else f"{sox:+.2f}%")
c[2].metric("ドル円 前日比", "—" if np.isnan(usd_pct) else f"{usd_pct:+.2f}%")
c[3].metric("VIX", "—" if np.isnan(vix) else f"{vix:.2f}")

cond_nq = not np.isnan(nq) and nq >= p_nq
cond_sox = not np.isnan(sox) and sox >= p_sox
cond_usd = not np.isnan(usd_pct) and usd_pct > p_usd
cond_vix = not np.isnan(vix) and vix < p_vix
cond_event = ok_event and not events

st.markdown("---")
if not ok_event:
    st.error("🔴 見送り：米国イベント情報を取得できません。安全側に倒します。")
elif events:
    st.error("🔴 見送り：今夜の重要な米国イベントがあります。")
    for e in events: st.write("・" + e)
elif all([cond_nq, cond_sox, cond_usd, cond_vix]):
    st.success("🟢 買い候補：設定したフィルターをすべてクリア。")
else:
    st.warning("🟡 見送り：フィルター未達。")

status = pd.DataFrame({
    "指標":["NASDAQ100先物","SOX","ドル円","VIX","米国重要イベント"],
    "現在値":[
        "取得失敗" if np.isnan(nq) else f"{nq:+.2f}%",
        "取得失敗" if np.isnan(sox) else f"{sox:+.2f}%",
        "取得失敗" if np.isnan(usd_pct) else f"{usd_pct:+.2f}%",
        "取得失敗" if np.isnan(vix) else f"{vix:.2f}",
        "なし" if ok_event and not events else ("あり" if events else "不明")],
    "判定":["○" if cond_nq else "×","○" if cond_sox else "×",
           "○" if cond_usd else "×","○" if cond_vix else "×",
           "○" if cond_event else "×"]})
st.dataframe(status, use_container_width=True, hide_index=True)
st.info("⚠️ 買い候補は翌朝上昇を保証しません。実運用では少額のフォワードテストを推奨します。")

# ---- backtest ----
st.markdown("---")
st.header("② バックテスト")
st.write("1570.Tそのものを対象にし、翌朝の寄り付きリターンを複利で計算します。米国系特徴量は1営業日ラグを取り、未来の情報を使わない設計です。")

years = list(range(2018, datetime.now(JST).year + 1))
a,b,c,d = st.columns(4)
sy = a.selectbox("開始年", years, index=max(0,len(years)-4))
sm = b.selectbox("開始月", range(1,13), index=0)
ey = c.selectbox("終了年", years, index=len(years)-1)
em = d.selectbox("終了月", range(1,13), index=datetime.now(JST).month-1)

if st.button("🔄 バックテストを実行", type="primary"):
    with st.spinner("データを取得して検証中..."):
        etf = yf_daily("1570.T","10y")
        n225 = yf_daily("^N225","10y")
        vixd = yf_daily("^VIX","10y")
        fx = yf_daily("JPY=X","10y")
        nqd = yf_daily("NQ=F","10y")
        soxd = yf_daily("^SOX","10y")

    if any(x.empty for x in [etf,n225,vixd,fx,nqd,soxd]):
        st.error("必要なデータを取得できませんでした。")
    else:
        bt = pd.DataFrame(index=etf.index)
        bt["close"] = etf["Close"]
        bt["next_open"] = etf["Open"].shift(-1)
        # Conservative daily proxy: only completed prior US-session data.
        bt["nq_pct"] = nqd["Close"].pct_change().reindex(bt.index).shift(1)*100
        bt["sox_pct"] = soxd["Close"].pct_change().reindex(bt.index).shift(1)*100
        bt["usd_pct"] = fx["Close"].pct_change().reindex(bt.index).shift(1)*100
        bt["vix"] = vixd["Close"].reindex(bt.index).shift(1)
        bt["jp_pct"] = n225["Close"].pct_change().reindex(bt.index)*100
        bt["1570_day_pct"] = etf["Close"].pct_change()*100

        start = pd.Timestamp(f"{sy}-{sm:02d}-01")
        end = pd.Timestamp(f"{ey}-{em:02d}-01") + pd.offsets.MonthBegin(1)
        bt = bt[(bt.index >= start) & (bt.index < end)].dropna()

        bt["signal"] = ((bt.nq_pct >= p_nq) & (bt.sox_pct >= p_sox) &
                        (bt.usd_pct > p_usd) & (bt.vix < p_vix))
        trades = bt[bt.signal].copy()
        trades["return_pct"] = (trades.next_open / trades.close - 1)*100
        trades = trades.dropna()

        if trades.empty:
            st.warning("条件に一致するトレードがありません。")
        else:
            equity = (1 + trades.return_pct/100).cumprod()
            peak = equity.cummax()
            dd = equity/peak - 1
            win = trades.return_pct > 0
            wins, losses = int(win.sum()), int((~win).sum())
            wr = wins/len(trades)*100
            avg_win = trades.loc[win,"return_pct"].mean()
            avg_loss = trades.loc[~win,"return_pct"].mean()
            expectancy = wr/100*avg_win + (1-wr/100)*avg_loss
            pf = (trades.loc[win,"return_pct"].sum() /
                  abs(trades.loc[~win,"return_pct"].sum())) if losses else np.inf

            cols = st.columns(7)
            cols[0].metric("トレード数",len(trades))
            cols[1].metric("勝率",f"{wr:.1f}%")
            cols[2].metric("複利リターン",f"{(equity.iloc[-1]-1)*100:+.2f}%")
            cols[3].metric("期待値/回",f"{expectancy:+.3f}%")
            cols[4].metric("平均勝ち",f"{avg_win:+.3f}%")
            cols[5].metric("平均負け",f"{avg_loss:+.3f}%")
            cols[6].metric("最大DD",f"{dd.min()*100:.2f}%")
            st.caption(f"Profit Factor: {pf:.2f}  | 手数料・スリッページ・税金は未反映")

            st.line_chart(pd.DataFrame({"資産曲線":equity.values},index=trades.index))

            show = trades[["nq_pct","sox_pct","usd_pct","vix","return_pct"]].copy()
            show.columns=["NQ前回比%","SOX前回比%","ドル円前回比%","VIX","翌朝1570寄付リターン%"]
            st.dataframe(show.sort_index(ascending=False), use_container_width=True)

st.markdown("---")
st.caption("Yahoo Finance等のデータを利用。高頻度履歴の制約上、長期バックテストは保守的な日次プロキシです。投資助言ではありません。")
