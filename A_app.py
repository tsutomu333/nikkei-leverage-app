import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import time

st.set_page_config(page_title="日経レバ1泊トレード判定", page_icon="📈", layout="wide")

JST = timezone(timedelta(hours=9))

# =========================================================
# 共通データ取得
# =========================================================
@st.cache_data(ttl=900)
def get_latest_market_data():
    """
    本日の判定用。
    「現在値の前日比」を各銘柄から取得する。
    yfinance の日次履歴だけでなく、可能なら intraday を優先する。
    """
    def pct_change(ticker):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
            if hist.empty or len(hist) < 2:
                return np.nan
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if len(close) < 2:
                return np.nan
            return float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        except Exception:
            return np.nan

    def current_vix():
        try:
            t = yf.Ticker("^VIX")
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
            if hist.empty:
                return np.nan
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            return float(close.iloc[-1]) if len(close) else np.nan
        except Exception:
            return np.nan

    # NQ=F / SOX / USDJPY は直近日次値を利用。
    # 判定画面の表示値は「直近取得できた値」であることを明記する。
    return {
        "nasdaq": pct_change("NQ=F"),
        "sox": pct_change("^SOX"),
        "usd": pct_change("USDJPY=X"),
        "vix": current_vix(),
    }


@st.cache_data(ttl=900)
def check_us_macro_events():
    """FairEconomy の今週カレンダーから、当日のUSD Highイベントを確認。"""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        today_us = datetime.now(JST).strftime("%m-%d-%Y")
        events = []

        for event in root.findall("event"):
            country = event.findtext("country", "")
            impact = event.findtext("impact", "")
            date_str = event.findtext("date", "")
            title = event.findtext("title", "")

            if country == "USD" and impact == "High" and date_str == today_us:
                events.append(title)

        return True, events
    except Exception:
        # 取得失敗時は「イベントなし」と断定しない
        return False, ["取得エラー：手動で米国重要指標を確認してください"]


# =========================================================
# バックテスト
# =========================================================
@st.cache_data(ttl=3600)
def load_backtest_data():
    """
    日本営業日を基準に、直前の米国セッションのデータを割り当てる。
    取引は 1570.T の「日本営業日 t の始値 / 前営業日終値」を使う。
    実運用の「前日大引け買い→翌朝寄り売り」に合わせ、
    signal_date の条件で翌営業日の寄り付きリターンを検証する。

    重要:
    - 日本営業日 t の signal は、t-1 の米国セッション終値を利用。
    - そのため、翌営業日の結果を条件に混ぜない。
    """
    tickers = {
        "nq": "NQ=F",
        "sox": "^SOX",
        "usd": "USDJPY=X",
        "vix": "^VIX",
        "etf": "1570.T",
    }

    data = {}
    for key, ticker in tickers.items():
        x = yf.download(
            ticker,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x.index = pd.to_datetime(x.index).tz_localize(None)
        data[key] = x

    # 日本ETFの営業日を基準にする
    etf = data["etf"].copy()
    etf = etf[["Open", "Close"]].rename(
        columns={"Open": "ETF_Open", "Close": "ETF_Close"}
    )
    etf.index = pd.to_datetime(etf.index).normalize()

    # 米国側は日付だけをキーにする
    us = pd.DataFrame(index=pd.Index([], dtype="datetime64[ns]"))
    for key in ["nq", "sox", "usd", "vix"]:
        x = data[key]
        if key == "vix":
            s = x["Close"].rename("VIX_Close")
        else:
            s = x["Close"].rename(f"{key}_Close")
        s.index = pd.to_datetime(s.index).normalize()
        us = us.join(s, how="outer")

    # 日本営業日 t に、直前の米国日次セッションを割り当てる。
    # merge_asof の direction='backward' で t より前の日を取得する。
    jp = etf.reset_index().rename(columns={"index": "JP_Date"})
    us2 = us.reset_index().rename(columns={"index": "US_Date"})
    jp["US_Date"] = jp["JP_Date"] - pd.Timedelta(days=1)

    merged = pd.merge_asof(
        jp.sort_values("US_Date"),
        us2.sort_values("US_Date"),
        on="US_Date",
        direction="backward",
        allow_exact_matches=True,
    )

    # 米国データの日付が日本日付より未来にならないことを明示的に確認
    merged["US_Date"] = pd.to_datetime(merged["US_Date"])
    merged["JP_Date"] = pd.to_datetime(merged["JP_Date"])
    merged = merged[merged["US_Date"] < merged["JP_Date"]].copy()

    # 各米国指標の前日比
    for col, out in [
        ("nq_Close", "NASDAQ_pct"),
        ("sox_Close", "SOX_pct"),
        ("usd_Close", "USD_pct"),
    ]:
        merged[out] = merged[col].pct_change() * 100

    # 取引リターン:
    # signal日 t の大引けで買い、翌日本営業日 t+1 の寄りで売る。
    # したがって signal日の「翌営業日ETF_Open / signal日のETF_Close - 1」
    merged["Next_Open"] = merged["ETF_Open"].shift(-1)
    merged["Night_Return"] = (
        (merged["Next_Open"] - merged["ETF_Close"]) / merged["ETF_Close"] * 100
    )

    # 直前の米国セッションを識別するため、US_Dateも保存
    merged = merged.dropna(
        subset=["NASDAQ_pct", "SOX_pct", "USD_pct", "VIX_Close", "ETF_Close", "Next_Open"]
    ).copy()

    return merged


def run_backtest(df, p_nq, p_sox, p_usd, p_vix, start_date, end_date):
    x = df[(df["JP_Date"] >= pd.Timestamp(start_date)) &
           (df["JP_Date"] < pd.Timestamp(end_date))].copy()

    x["Cond_NQ"] = x["NASDAQ_pct"] >= p_nq
    x["Cond_SOX"] = x["SOX_pct"] >= p_sox
    x["Cond_USD"] = x["USD_pct"] > p_usd
    x["Cond_VIX"] = x["VIX_Close"] < p_vix
    x["Signal"] = x["Cond_NQ"] & x["Cond_SOX"] & x["Cond_USD"] & x["Cond_VIX"]

    trades = x[x["Signal"]].copy()

    if trades.empty:
        return trades, {
            "count": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "expectancy": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "total_compound": 0, "mdd": 0, "max_losing_streak": 0
        }

    trades["Win"] = trades["Night_Return"] > 0

    # 1トレード=1単位資産として複利
    trades["Equity"] = (1 + trades["Night_Return"] / 100).cumprod()
    peak = trades["Equity"].cummax()
    dd = trades["Equity"] / peak - 1
    mdd = float(dd.min() * 100)

    wins = trades.loc[trades["Night_Return"] > 0, "Night_Return"]
    losses = trades.loc[trades["Night_Return"] <= 0, "Night_Return"]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # 最大連敗
    losing = (trades["Night_Return"] <= 0).astype(int).tolist()
    max_streak = cur = 0
    for v in losing:
        if v:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    result = {
        "count": len(trades),
        "wins": int((trades["Night_Return"] > 0).sum()),
        "losses": int((trades["Night_Return"] <= 0).sum()),
        "win_rate": float((trades["Night_Return"] > 0).mean() * 100),
        "expectancy": float(trades["Night_Return"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
        "profit_factor": pf,
        "total_compound": float((trades["Equity"].iloc[-1] - 1) * 100),
        "mdd": mdd,
        "max_losing_streak": max_streak,
    }
    return trades, result


# =========================================================
# UI
# =========================================================
st.title("📈 日経レバ1泊トレード判定")
st.caption("前日の大引けで1570を買い、翌営業日の寄り付きで売る。15:25頃に利用できる情報だけで判定する設計です。")

st.sidebar.header("⚙️ 判定条件")

# Session State に「今日の数値」を保存
for k, default in {
    "copy_nq": 0.10,
    "copy_sox": 0.10,
    "copy_usd": -0.50,
    "copy_vix": 20.00,
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

p_nq = st.sidebar.number_input(
    "NASDAQ100先物 基準(%)", value=float(st.session_state["copy_nq"]),
    step=0.05, format="%.2f", key="p_nq"
)
p_sox = st.sidebar.number_input(
    "SOX基準(%)", value=float(st.session_state["copy_sox"]),
    step=0.05, format="%.2f", key="p_sox"
)
p_usd = st.sidebar.number_input(
    "ドル円 前日比以下制限(%)", value=float(st.session_state["copy_usd"]),
    step=0.05, format="%.2f", key="p_usd"
)
p_vix = st.sidebar.number_input(
    "VIX上限", value=float(st.session_state["copy_vix"]),
    step=0.5, format="%.2f", key="p_vix"
)

st.sidebar.markdown("---")
st.sidebar.subheader("売買ルール")
st.sidebar.write("判定：15:20〜15:25頃")
st.sidebar.write("買い：当日大引けで1570")
st.sidebar.write("売り：翌営業日寄り付き")

# ---------------------------------------------------------
# 今日の判定
# ---------------------------------------------------------
with st.spinner("最新データを取得中..."):
    latest = get_latest_market_data()
    event_ok, event_list = check_us_macro_events()

nasdaq_pct = latest["nasdaq"]
sox_pct = latest["sox"]
usd_pct = latest["usd"]
vix_value = latest["vix"]

st.header("① 本日の判定")
st.caption(f"日本時間 {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("NASDAQ100先物", f"{nasdaq_pct:+.2f}%" if pd.notna(nasdaq_pct) else "取得不可")
c2.metric("SOX", f"{sox_pct:+.2f}%" if pd.notna(sox_pct) else "取得不可")
c3.metric("ドル円 前日比", f"{usd_pct:+.2f}%" if pd.notna(usd_pct) else "取得不可")
c4.metric("VIX", f"{vix_value:.2f}" if pd.notna(vix_value) else "取得不可")

cond_nq = pd.notna(nasdaq_pct) and nasdaq_pct >= p_nq
cond_sox = pd.notna(sox_pct) and sox_pct >= p_sox
cond_usd = pd.notna(usd_pct) and usd_pct > p_usd
cond_vix = pd.notna(vix_value) and vix_value < p_vix

macro_error = (not event_ok and len(event_list) == 1 and "取得エラー" in event_list[0])
event_has_high = event_ok and len(event_list) > 0
cond_event = (not event_has_high) and (not macro_error)

if macro_error:
    st.warning("⚠️ 米国重要指標カレンダーを取得できませんでした。安全側のためGO判定にはしません。")
elif event_has_high:
    st.error("🚨 今夜の米国重要イベントあり。見送り。")
    for e in event_list:
        st.write(f"・{e}")

all_clear = cond_nq and cond_sox and cond_usd and cond_vix and cond_event

if all_clear:
    st.success("🟢 買い候補：設定したフィルターをすべてクリア。")
else:
    st.warning("🟡 見送り：少なくとも1つのフィルターをクリアしていません。")

st.subheader("各指標のクリア状況")
status = pd.DataFrame({
    "指標": ["NASDAQ100先物", "SOX", "ドル円", "VIX", "米国重要イベント"],
    "現在値": [
        f"{nasdaq_pct:+.2f}%" if pd.notna(nasdaq_pct) else "取得不可",
        f"{sox_pct:+.2f}%" if pd.notna(sox_pct) else "取得不可",
        f"{usd_pct:+.2f}%" if pd.notna(usd_pct) else "取得不可",
        f"{vix_value:.2f}" if pd.notna(vix_value) else "取得不可",
        "あり" if event_has_high else ("取得エラー" if macro_error else "なし"),
    ],
    "判定": [
        "○" if cond_nq else "×",
        "○" if cond_sox else "×",
        "○" if cond_usd else "×",
        "○" if cond_vix else "×",
        "○" if cond_event else "×",
    ]
})
st.dataframe(status, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# 今日の実測値をバックテスト条件へコピー
# ---------------------------------------------------------
st.markdown("---")
st.header("🔵 今日の数値をバックテスト条件にコピー")

st.write("今日15:25頃に実際に確認した数値を、そのまま過去データの検索条件として使えます。")

if pd.notna(nasdaq_pct) and pd.notna(sox_pct) and pd.notna(usd_pct) and pd.notna(vix_value):
    st.info(
        f"コピーする条件：NASDAQ ≥ {nasdaq_pct:.2f}% / "
        f"SOX ≥ {sox_pct:.2f}% / ドル円 > {usd_pct:.2f}% / VIX < {vix_value:.2f}"
    )

    if st.button("📋 本日の数値を条件にコピー", type="primary"):
        # 次の再実行でサイドバーに反映
        st.session_state["copy_nq"] = round(float(nasdaq_pct), 2)
        st.session_state["copy_sox"] = round(float(sox_pct), 2)
        st.session_state["copy_usd"] = round(float(usd_pct), 2)
        st.session_state["copy_vix"] = round(float(vix_value), 2)

        # ウィジェットキーにも即時反映
        st.session_state["p_nq"] = st.session_state["copy_nq"]
        st.session_state["p_sox"] = st.session_state["copy_sox"]
        st.session_state["p_usd"] = st.session_state["copy_usd"]
        st.session_state["p_vix"] = st.session_state["copy_vix"]

        st.success("✅ 本日の実測値をバックテスト条件へコピーしました。")
        st.rerun()
else:
    st.warning("現在値が揃っていないため、コピーできません。")

# ---------------------------------------------------------
# バックテスト
# ---------------------------------------------------------
st.markdown("---")
st.header("② バックテスト")
st.write("日本営業日を基準に、直前の米国セッションのデータで条件判定し、翌営業日の1570寄り付きまでを検証します。")

years = list(range(2018, datetime.now(JST).year + 1))
c1, c2, c3, c4 = st.columns(4)
start_year = c1.selectbox("開始年", years, index=max(0, len(years)-4))
start_month = c2.selectbox("開始月", range(1, 13), index=0)
end_year = c3.selectbox("終了年", years, index=len(years)-1)
end_month = c4.selectbox("終了月", range(1, 13), index=datetime.now(JST).month-1)

start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01")
end_date = pd.Timestamp(
    f"{end_year+1}-01-01" if end_month == 12 else f"{end_year}-{end_month+1:02d}-01"
)

if st.button("🔄 バックテストを実行", type="primary"):
    with st.spinner("過去データを取得して検証中..."):
        try:
            bt_df = load_backtest_data()
            trades, result = run_backtest(
                bt_df, p_nq, p_sox, p_usd, p_vix, start_date, end_date
            )
            st.session_state["bt_trades"] = trades
            st.session_state["bt_result"] = result
            st.session_state["bt_params"] = (p_nq, p_sox, p_usd, p_vix)
            st.session_state["bt_period"] = (start_date, end_date)
        except Exception as e:
            st.error(f"バックテストでエラーが発生しました：{e}")

if "bt_result" in st.session_state:
    result = st.session_state["bt_result"]
    trades = st.session_state["bt_trades"]

    st.subheader("検証結果")
    if result["count"] == 0:
        st.warning("条件に一致するトレードがありませんでした。")
    else:
        a, b, c, d, e = st.columns(5)
        a.metric("取引数", f'{result["count"]}回')
        b.metric("勝率", f'{result["win_rate"]:.1f}%')
        c.metric("期待値/回", f'{result["expectancy"]:+.3f}%')
        d.metric("Profit Factor", f'{result["profit_factor"]:.2f}')
        e.metric("最大DD", f'{result["mdd"]:.2f}%')

        a, b, c, d = st.columns(4)
        a.metric("勝ち / 負け", f'{result["wins"]} / {result["losses"]}')
        b.metric("平均勝ち", f'{result["avg_win"]:+.3f}%')
        c.metric("平均負け", f'{result["avg_loss"]:+.3f}%')
        d.metric("最大連敗", f'{result["max_losing_streak"]}回')

        st.metric("複利リターン（手数料等未反映）", f'{result["total_compound"]:+.2f}%')

        chart_df = trades[["JP_Date", "Equity"]].set_index("JP_Date")
        st.subheader("📈 資産推移")
        st.line_chart(chart_df)

        st.subheader("📋 トレード履歴")
        display = trades[
            ["JP_Date", "US_Date", "NASDAQ_pct", "SOX_pct", "USD_pct", "VIX_Close", "ETF_Close", "Next_Open", "Night_Return"]
        ].copy()
        display.columns = [
            "判定日", "使用した米国セッション", "NASDAQ前日比(%)", "SOX前日比(%)",
            "ドル円前日比(%)", "VIX", "1570判定日終値", "翌営業日寄り", "1泊リターン(%)"
        ]
        display = display.sort_values("判定日", ascending=False)
        st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("※バックテストは市場データの取得仕様・時差・休場日等の影響を受けます。実運用前に証券会社の実際の約定時刻・1570の寄り付き価格と照合してください。")
