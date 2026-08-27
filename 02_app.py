import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="日経レバ 1泊判定 02", page_icon="📈", layout="wide")
JST = ZoneInfo("Asia/Tokyo")

st.title("📈 日経レバ 1泊トレード判定")
st.caption("前日の大引けで1570を買い、翌営業日の寄り付きで売る戦略。15:25頃に利用できる情報だけで判定します。")

# =========================================================
# データ取得
# =========================================================
@st.cache_data(ttl=300)
def yf_daily(ticker, period="10y"):
    try:
        x = yf.download(
            ticker, period=period, auto_adjust=False,
            progress=False, threads=False
        )
        if x.empty:
            return pd.DataFrame()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x.index = pd.to_datetime(x.index).tz_localize(None)
        return x.dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def yf_intraday(ticker, period="1d", interval="5m"):
    try:
        x = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False
        )
        if x.empty:
            return pd.DataFrame()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_convert(JST)
        return x.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def current_vs_prev_close_pct(ticker):
    """現在値を直近の日次終値と比較して％を返す。"""
    intraday = yf_intraday(ticker)
    daily = yf_daily(ticker, "5d")

    if intraday.empty or daily.empty:
        return np.nan, np.nan, None

    intraday = intraday.dropna(subset=["Close"])
    if intraday.empty or len(daily) < 2:
        return np.nan, np.nan, None

    current = float(intraday["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2])
    pct = (current / prev_close - 1.0) * 100.0
    return pct, current, intraday.index[-1]

def latest_daily_pct(ticker):
    x = yf_daily(ticker, "5d")
    if len(x) < 2:
        return np.nan
    return float((x["Close"].iloc[-1] / x["Close"].iloc[-2] - 1) * 100)

@st.cache_data(ttl=300)
def macro_events():
    """取得失敗は『イベントなし』にしない。"""
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        today = datetime.now(JST).strftime("%m-%d-%Y")
        events = []

        for e in root.findall("event"):
            country = e.findtext("country", "")
            impact = e.findtext("impact", "")
            date_str = e.findtext("date", "")
            title = e.findtext("title", "")
            if country == "USD" and impact == "High" and date_str == today:
                events.append(title)

        return True, events
    except Exception as exc:
        return False, [f"取得エラー: {exc}"]

# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("⚙️ 判定条件")
p_nq = st.sidebar.number_input(
    "NASDAQ100先物 基準(%)", -3.0, 5.0, 0.1, 0.1
)
p_sox = st.sidebar.number_input(
    "SOX 基準(%)", -3.0, 5.0, 0.1, 0.1
)
p_usd = st.sidebar.number_input(
    "ドル円 前日比の下限(%)", -3.0, 1.0, -0.5, 0.1
)
p_vix = st.sidebar.number_input(
    "VIX 上限", 10.0, 50.0, 20.0, 0.5
)

st.sidebar.markdown("---")
st.sidebar.write("**売買ルール**")
st.sidebar.write("判定：15:20〜15:25頃")
st.sidebar.write("買い：当日大引け")
st.sidebar.write("売り：翌営業日寄り付き")

# =========================================================
# 本日の判定
# =========================================================
st.header("① 本日の判定")
st.caption(f"日本時間 {datetime.now(JST):%Y-%m-%d %H:%M:%S}")

with st.spinner("最新データを取得中..."):
    nq_pct, nq_price, nq_ts = current_vs_prev_close_pct("NQ=F")
    sox_pct, sox_price, sox_ts = current_vs_prev_close_pct("^SOX")

    # USDJPYはYahooでは JPY=X が「USD/JPY」
    usd_pct = latest_daily_pct("JPY=X")

    vix_df = yf_daily("^VIX", "5d")
    vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else np.nan

    ok_event, events = macro_events()

c = st.columns(4)
c[0].metric(
    "NASDAQ100先物",
    "取得失敗" if np.isnan(nq_pct) else f"{nq_pct:+.2f}%"
)
c[1].metric(
    "SOX",
    "取得失敗" if np.isnan(sox_pct) else f"{sox_pct:+.2f}%"
)
c[2].metric(
    "ドル円 前日比",
    "取得失敗" if np.isnan(usd_pct) else f"{usd_pct:+.2f}%"
)
c[3].metric(
    "VIX",
    "取得失敗" if np.isnan(vix) else f"{vix:.2f}"
)

cond_nq = not np.isnan(nq_pct) and nq_pct >= p_nq
cond_sox = not np.isnan(sox_pct) and sox_pct >= p_sox
cond_usd = not np.isnan(usd_pct) and usd_pct > p_usd
cond_vix = not np.isnan(vix) and vix < p_vix
cond_event = ok_event and not events

st.markdown("---")

if not ok_event:
    st.error("🔴 見送り：米国イベント情報を取得できません。安全側に倒します。")
elif events:
    st.error("🔴 見送り：今夜の重要な米国イベントがあります。")
    for e in events:
        st.write("・" + e)
elif all([cond_nq, cond_sox, cond_usd, cond_vix]):
    st.success("🟢 買い候補：設定したフィルターをすべてクリア。")
else:
    st.warning("🟡 見送り：フィルター未達。")

status = pd.DataFrame({
    "指標": [
        "NASDAQ100先物", "SOX", "ドル円", "VIX", "米国重要イベント"
    ],
    "現在値": [
        "取得失敗" if np.isnan(nq_pct) else f"{nq_pct:+.2f}%",
        "取得失敗" if np.isnan(sox_pct) else f"{sox_pct:+.2f}%",
        "取得失敗" if np.isnan(usd_pct) else f"{usd_pct:+.2f}%",
        "取得失敗" if np.isnan(vix) else f"{vix:.2f}",
        "なし" if ok_event and not events else (
            "あり" if events else "不明"
        )
    ],
    "判定": [
        "○" if cond_nq else "×",
        "○" if cond_sox else "×",
        "○" if cond_usd else "×",
        "○" if cond_vix else "×",
        "○" if cond_event else "×"
    ]
})
st.dataframe(status, use_container_width=True, hide_index=True)

st.info(
    "⚠️ 「買い候補」は翌朝上昇を保証しません。"
    "バックテストとフォワードテストで優位性を確認してください。"
)

# =========================================================
# バックテスト
# =========================================================
st.markdown("---")
st.header("② バックテスト")
st.write(
    "1570.Tを実際の売買対象にします。"
    "『当日大引け→翌営業日寄り』を1トレードとし、"
    "米国系データは当日15:25頃にすでに分かっている"
    "直前の米国セッションを対応させます。"
)

years = list(range(2018, datetime.now(JST).year + 1))
a, b, c2, d = st.columns(4)
sy = a.selectbox("開始年", years, index=max(0, len(years) - 4))
sm = b.selectbox("開始月", range(1, 13), index=0)
ey = c2.selectbox("終了年", years, index=len(years) - 1)
em = d.selectbox("終了月", range(1, 13), index=datetime.now(JST).month - 1)

run_bt = st.button("🔄 バックテストを実行", type="primary")

if run_bt:
    with st.spinner("1570・市場データを取得して検証中..."):
        etf = yf_daily("1570.T", "10y")
        n225 = yf_daily("^N225", "10y")
        vixd = yf_daily("^VIX", "10y")
        fx = yf_daily("JPY=X", "10y")
        nqd = yf_daily("NQ=F", "10y")
        soxd = yf_daily("^SOX", "10y")

    if any(x.empty for x in [etf, n225, vixd, fx, nqd, soxd]):
        st.error("必要なデータを取得できませんでした。")
    else:
        bt = pd.DataFrame(index=etf.index)
        bt["close"] = etf["Close"]
        bt["next_open"] = etf["Open"].shift(-1)

        # -------------------------------------------------
        # 日本の日付 D に対して、
        # 「D の15:25時点で既知」の米国データは
        # 原則として米国の直前セッション D-1。
        # そのデータを D に対応付ける。
        # -------------------------------------------------
        def prior_us_session_to_japan_date(series, japan_index):
            s = series.copy()
            s.index = pd.to_datetime(s.index).normalize() + pd.Timedelta(days=1)
            return s.reindex(japan_index)

        nq_prior = nqd["Close"].pct_change() * 100
        sox_prior = soxd["Close"].pct_change() * 100
        vix_prior = vixd["Close"]
        fx_prior = fx["Close"].pct_change() * 100

        bt["nq_pct"] = prior_us_session_to_japan_date(
            nq_prior, bt.index
        )
        bt["sox_pct"] = prior_us_session_to_japan_date(
            sox_prior, bt.index
        )
        bt["vix"] = prior_us_session_to_japan_date(
            vix_prior, bt.index
        )
        bt["usd_pct"] = prior_us_session_to_japan_date(
            fx_prior, bt.index
        )

        # 日本市場で当日大引けまでに分かっている情報
        bt["n225_pct"] = n225["Close"].pct_change().reindex(bt.index) * 100
        bt["1570_day_pct"] = etf["Close"].pct_change() * 100

        start = pd.Timestamp(f"{sy}-{sm:02d}-01")
        end = (
            pd.Timestamp(f"{ey}-{em:02d}-01")
            + pd.offsets.MonthBegin(1)
        )
        bt = bt[(bt.index >= start) & (bt.index < end)].dropna()

        bt["signal"] = (
            (bt["nq_pct"] >= p_nq)
            & (bt["sox_pct"] >= p_sox)
            & (bt["usd_pct"] > p_usd)
            & (bt["vix"] < p_vix)
        )

        trades = bt[bt["signal"]].copy()
        trades["return_pct"] = (
            trades["next_open"] / trades["close"] - 1
        ) * 100
        trades = trades.dropna()

        if trades.empty:
            st.warning("条件に一致するトレードがありません。")
        else:
            equity = (1 + trades["return_pct"] / 100).cumprod()
            peak = equity.cummax()
            dd = equity / peak - 1

            win = trades["return_pct"] > 0
            wins = int(win.sum())
            losses = int((~win).sum())
            n = len(trades)

            win_rate = wins / n * 100
            avg_win = (
                trades.loc[win, "return_pct"].mean()
                if wins else 0.0
            )
            avg_loss = (
                trades.loc[~win, "return_pct"].mean()
                if losses else 0.0
            )
            expectancy = (
                win_rate / 100 * avg_win
                + (1 - win_rate / 100) * avg_loss
            )

            profit_factor = (
                trades.loc[win, "return_pct"].sum()
                / abs(trades.loc[~win, "return_pct"].sum())
                if losses else np.inf
            )

            # 最大連敗
            loss_streak = 0
            max_loss_streak = 0
            for is_win in win.tolist():
                if is_win:
                    loss_streak = 0
                else:
                    loss_streak += 1
                    max_loss_streak = max(max_loss_streak, loss_streak)

            total_return = (equity.iloc[-1] - 1) * 100
            max_dd = dd.min() * 100

            st.subheader("検証結果")

            # 7列だと画面幅によって数字が省略されるため、
            # 2段にして確実に全文字を表示。
            r1 = st.columns(4)
            r1[0].metric("トレード数", f"{n:,} 回")
            r1[1].metric("勝率", f"{win_rate:.1f}%")
            r1[2].metric("勝ち / 負け", f"{wins} / {losses}")
            r1[3].metric("最大連敗", f"{max_loss_streak} 回")

            r2 = st.columns(4)
            r2[0].metric("複利リターン", f"{total_return:+.2f}%")
            r2[1].metric("期待値 / 回", f"{expectancy:+.3f}%")
            r2[2].metric("平均勝ち", f"{avg_win:+.3f}%")
            r2[3].metric("平均負け", f"{avg_loss:+.3f}%")

            r3 = st.columns(2)
            r3[0].metric("最大DD", f"{max_dd:.2f}%")
            r3[1].metric(
                "Profit Factor",
                "∞" if np.isinf(profit_factor)
                else f"{profit_factor:.2f}"
            )

            st.caption(
                "※複利リターンは各トレードのリターンを連続して"
                "再投資した場合。手数料・スリッページ・税金は未反映。"
            )

            st.subheader("📈 資産曲線")
            chart = pd.DataFrame(
                {"資産": equity.values},
                index=trades.index
            )
            st.line_chart(chart)

            st.subheader("📋 トレード履歴")
            show = trades[
                ["nq_pct", "sox_pct", "usd_pct", "vix", "return_pct"]
            ].copy()
            show.columns = [
                "NQ前回米国セッション比(%)",
                "SOX前回米国セッション比(%)",
                "ドル円前回比(%)",
                "VIX",
                "翌朝1570寄付リターン(%)"
            ]
            st.dataframe(
                show.sort_index(ascending=False),
                use_container_width=True
            )

            st.subheader("🔎 判定条件別の参考")
            st.write(
                "現在は固定ルール版です。次版以降で、"
                "条件別の勝率・期待値・上昇確率を比較し、"
                "『単純なGO/見送り』から確率判定へ進化させます。"
            )

st.markdown("---")
st.caption(
    "データ：Yahoo Finance等。無料データには時間足・履歴の制約があります。"
    "本アプリは研究・検証用であり、投資助言ではありません。"
)
