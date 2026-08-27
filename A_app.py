import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

st.set_page_config(page_title="日経レバ1泊トレード判定", layout="wide")

# =========================================================
# 共通
# =========================================================
DEFAULTS = {
    "p_nq": 0.10,
    "p_sox": 0.10,
    "p_usd": -0.50,
    "p_vix": 20.0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data(ttl=900)
def get_market_data(ticker_symbol, is_pct=True):
    try:
        hist = yf.Ticker(ticker_symbol).history(period="5d", auto_adjust=False)
        if len(hist) < 2:
            return 0.0
        current = float(hist["Close"].iloc[-1])
        if not is_pct:
            return round(current, 2)
        prev = float(hist["Close"].iloc[-2])
        return round((current - prev) / prev * 100, 2)
    except Exception:
        return 0.0


@st.cache_data(ttl=900)
def check_us_macro_events():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        today = datetime.now().strftime("%m-%d-%Y")
        events = []

        for event in root.findall("event"):
            country = event.findtext("country", "")
            impact = event.findtext("impact", "")
            date_str = event.findtext("date", "")
            title = event.findtext("title", "")

            if country == "USD" and impact == "High" and date_str == today:
                events.append(title)

        return bool(events), events
    except Exception:
        return False, ["※経済指標カレンダー取得エラー。手動確認してください"]


@st.cache_data(ttl=3600)
def load_backtest_data():
    """日経225の翌営業日寄り付きリターンを、前営業日までに利用可能な情報で判定する。

    日本営業日 t の大引けで買い、翌営業日 t+1 の寄りで売る。
    米国指標は「日本営業日 t の15:25時点で既知」となるよう、
    直近の米国取引日の終値を1営業日分シフトして使用する。
    """

    tickers = {
        "N225": "^N225",
        "NQ": "NQ=F",
        "SOX": "^SOX",
        "USD": "USDJPY=X",
        "VIX": "^VIX",
    }

    raw = {}
    for name, ticker in tickers.items():
        raw[name] = yf.download(
            ticker, period="8y", auto_adjust=False,
            progress=False, threads=False
        )

    def close_series(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    def open_series(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Open"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    n225_close = close_series(raw["N225"])
    n225_open = open_series(raw["N225"])
    nq_close = close_series(raw["NQ"])
    sox_close = close_series(raw["SOX"])
    usd_close = close_series(raw["USD"])
    vix_close = close_series(raw["VIX"])

    # 日本営業日を基準に、直近の米国データを「前日まで」に固定する。
    df = pd.DataFrame(index=n225_close.index)
    df["N225_Close"] = n225_close
    df["Next_N225_Open"] = n225_open.shift(-1)

    us = pd.DataFrame({
        "NQ_Close": nq_close,
        "SOX_Close": sox_close,
        "USD_Close": usd_close,
        "VIX_Close": vix_close,
    })

    # 日本の日付に対して、その日以前の米国終値をasofで対応。
    # さらに「日本の大引け時点で確実に分かる」ことを優先し、
    # 同日米国終値は原則まだ存在しないため、前日米国終値を使う。
    us = us.sort_index()
    jp_dates = pd.DataFrame({"jp_date": df.index}).sort_values("jp_date")

    for col in us.columns:
        tmp = us[[col]].copy()
        tmp["us_date"] = tmp.index
        tmp = tmp.reset_index(drop=True).sort_values("us_date")

        # 日本日付より前の米国取引日をマッチ
        matched = pd.merge_asof(
            jp_dates,
            tmp,
            left_on="jp_date",
            right_on="us_date",
            direction="backward",
            allow_exact_matches=False,
        )
        df[col] = matched[col].to_numpy()

    # 米国各指標の前日比（米国側の日次変化率）
    for col in ["NQ_Close", "SOX_Close", "USD_Close"]:
        df[col + "_pct"] = df[col].pct_change() * 100

    # 未来情報を使わないため、翌朝寄りの結果は評価専用。
    df["Night_Return"] = (
        (df["Next_N225_Open"] - df["N225_Close"])
        / df["N225_Close"] * 100
    )

    return df.dropna(subset=[
        "N225_Close", "Next_N225_Open",
        "NQ_Close_pct", "SOX_Close_pct",
        "USD_Close_pct", "VIX_Close"
    ])


# =========================================================
# 今日のデータを先に取得
# =========================================================
with st.spinner("最新のマーケットデータを取得中..."):
    has_macro_event, event_list = check_us_macro_events()
    nasdaq_pct = get_market_data("NQ=F", True)
    sox_pct = get_market_data("^SOX", True)
    usdjpy_pct = get_market_data("USDJPY=X", True)
    vix_value = get_market_data("^VIX", False)

# =========================================================
# 「本日の数値を条件にコピー」は、サイドバーのウィジェットより
# 前に実行する。これが今回のエラー修正の核心。
# =========================================================
copy_nq = round(nasdaq_pct, 2)
copy_sox = round(sox_pct, 2)
copy_usd = round(usdjpy_pct, 2)
copy_vix = round(vix_value, 2)

st.title("📈 日経レバ1泊トレード判定")
st.caption(
    "前日の大引けで1570を買い、翌営業日の寄り付きで売る戦略。"
    "15:25頃までに利用できる情報だけで判定します。"
)

st.info(
    f"コピーする条件：NASDAQ ≥ {copy_nq:.2f}% / "
    f"SOX ≥ {copy_sox:.2f}% / ドル円 > {copy_usd:.2f}% / "
    f"VIX < {copy_vix:.2f}"
)

if st.button("📋 本日の数値を条件にコピー", type="primary"):
    # まだサイドバーのウィジェットを生成していないので安全に更新できる。
    st.session_state["p_nq"] = copy_nq
    st.session_state["p_sox"] = copy_sox
    st.session_state["p_usd"] = copy_usd
    st.session_state["p_vix"] = copy_vix
    st.rerun()

# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("⚙️ 判定条件")
st.sidebar.number_input(
    "NASDAQ100先物 基準(%)",
    min_value=-5.0, max_value=5.0, step=0.01,
    key="p_nq"
)
st.sidebar.number_input(
    "SOX基準(%)",
    min_value=-5.0, max_value=5.0, step=0.01,
    key="p_sox"
)
st.sidebar.number_input(
    "ドル円 前日比以下制限(%)",
    min_value=-5.0, max_value=5.0, step=0.01,
    key="p_usd"
)
st.sidebar.number_input(
    "VIX上限",
    min_value=5.0, max_value=60.0, step=0.01,
    key="p_vix"
)

p_nq = float(st.session_state["p_nq"])
p_sox = float(st.session_state["p_sox"])
p_usd = float(st.session_state["p_usd"])
p_vix = float(st.session_state["p_vix"])

# =========================================================
# 今日の判定
# =========================================================
st.header("① 本日の判定")
st.caption("日本時間 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

cols = st.columns(4)
cols[0].metric("NASDAQ100先物", f"{nasdaq_pct:+.2f}%")
cols[1].metric("SOX", f"{sox_pct:+.2f}%")
cols[2].metric("ドル円 前日比", f"{usdjpy_pct:+.2f}%")
cols[3].metric("VIX", f"{vix_value:.2f}")

cond_nq = nasdaq_pct >= p_nq
cond_sox = sox_pct >= p_sox
cond_usd = usdjpy_pct > p_usd
cond_vix = vix_value < p_vix
cond_event = not has_macro_event

if not cond_event:
    st.error("🚨 見送り：今夜は重要な米国経済イベントがあります。")
elif not cond_vix:
    st.error(f"🚨 見送り：VIX {vix_value:.2f} ≥ {p_vix:.2f}")
elif not cond_usd:
    st.warning(f"🟡 見送り：ドル円 {usdjpy_pct:+.2f}% ≤ {p_usd:+.2f}%")
elif not cond_sox:
    st.warning(f"🟡 見送り：SOX {sox_pct:+.2f}% < {p_sox:.2f}%")
elif not cond_nq:
    st.warning(f"🟡 見送り：NASDAQ100先物 {nasdaq_pct:+.2f}% < {p_nq:.2f}%")
else:
    st.success("🟢 買い候補：設定したフィルターをすべてクリア。")

st.subheader("各条件のクリア状況")
status = pd.DataFrame({
    "指標": ["NASDAQ100先物", "SOX", "ドル円", "VIX", "米国重要イベント"],
    "現在値": [
        f"{nasdaq_pct:+.2f}%",
        f"{sox_pct:+.2f}%",
        f"{usdjpy_pct:+.2f}%",
        f"{vix_value:.2f}",
        "あり" if has_macro_event else "なし",
    ],
    "判定": [
        "○" if cond_nq else "×",
        "○" if cond_sox else "×",
        "○" if cond_usd else "×",
        "○" if cond_vix else "×",
        "○" if cond_event else "×",
    ],
})
st.dataframe(status, use_container_width=True, hide_index=True)

if has_macro_event:
    st.warning("今夜の重要イベント：" + " / ".join(event_list))

# =========================================================
# 今日の判定を学ぶ：詳細解説
# =========================================================
st.markdown("---")
st.header("📚 今日の判定を理解する")
st.write(
    "この欄は、毎日の判定結果を読むだけで4つの市場指標の意味と、"
    "なぜ『買い候補／見送り』になるのかを学べるようにしています。"
)

with st.expander("① NASDAQ100先物：米国ハイテク株の勢いを見る", expanded=True):
    st.write(
        f"**今日の値：{nasdaq_pct:+.2f}% ／ 基準：{p_nq:+.2f}%以上 → {'クリア' if cond_nq else '未達'}**"
    )
    st.write(
        "NASDAQ100先物は、米国の大型ハイテク企業を中心とした市場の方向感を確認するために使います。"
        "日経平均も半導体・ハイテク株の影響を受けやすいため、翌朝の日経の寄り付き方向を考える材料になります。"
    )
    st.write(
        "このアプリでは『前日の米国市場が強いほど、翌朝の日経も上向きになりやすい』という仮説を、"
        "NASDAQ100先物の前日比で検証しています。"
    )
    st.info(
        "📌 読み方：基準以上なら『米国ハイテク株の追い風あり』。ただし、NASDAQだけで買いを決めるのではなく、"
        "SOX・ドル円・VIXも同時に確認します。"
    )

with st.expander("② SOX指数：半導体株の強さを見る", expanded=True):
    st.write(
        f"**今日の値：{sox_pct:+.2f}% ／ 基準：{p_sox:+.2f}%以上 → {'クリア' if cond_sox else '未達'}**"
    )
    st.write(
        "SOX指数（フィラデルフィア半導体株指数）は、半導体関連株の動きを見る代表的な指数です。"
        "日経平均には東京エレクトロン、アドバンテストなど半導体関連の影響が大きいため、"
        "NASDAQとは別に半導体セクターの強さを確認します。"
    )
    st.write(
        "NASDAQが上昇していても半導体だけが弱ければ、日経の上昇が続かない可能性があります。"
        "そのため、この戦略では『NASDAQ＋SOX』の両方が基準を満たすことを条件にしています。"
    )
    st.info(
        "📌 読み方：SOXが基準以上なら『日経の主力半導体株にも追い風がある』と考える材料になります。"
    )

with st.expander("③ ドル円：円高・円安が日経に与える影響を見る", expanded=True):
    st.write(
        f"**今日の値：{usdjpy_pct:+.2f}% ／ 基準：{p_usd:+.2f}%より大きい → {'クリア' if cond_usd else '未達'}**"
    )
    st.write(
        "ドル円は『1ドルが何円か』を表します。ドル円が下がると円高、上がると円安です。"
        "日経平均には輸出企業が多いため、急激な円高は株価の重しになりやすい傾向があります。"
    )
    st.write(
        "この戦略では、米国株が強くても急な円高が進んでいる日は見送る、という考え方です。"
        "ただし、円安なら必ず日経が上がるという意味ではありません。"
    )
    st.info(
        "📌 読み方：基準を下回るほど円高方向の警戒が強くなります。『米国株は強いが為替が悪い』という日を除外する役割です。"
    )

with st.expander("④ VIX：市場の不安・変動の大きさを見る", expanded=True):
    st.write(
        f"**今日の値：{vix_value:.2f} ／ 上限：{p_vix:.2f}未満 → {'クリア' if cond_vix else '未達'}**"
    )
    st.write(
        "VIXは、S&P500オプション市場から算出される、将来の株価変動に対する市場の警戒感を示す代表的な指数です。"
        "一般にVIXが高いほど、市場が不安定で値動きが大きくなっていることを示します。"
    )
    st.write(
        "翌朝までの1泊取引では、夜間に大きく相場が動くことが最大のリスクの一つです。"
        "そこでVIXが一定水準以上の日は、他の条件が良くても見送る設計にしています。"
    )
    st.info(
        "📌 読み方：VIXは『低ければ必ず上がる』という指標ではありません。"
        "この戦略では主に『危険な相場環境を避けるフィルター』として使います。"
    )

with st.expander("⑤ 米国重要イベント：予測しにくい夜を避ける", expanded=True):
    if has_macro_event:
        st.write("**今日：重要イベントあり → 見送り**")
    else:
        st.write("**今日：取得できた範囲では重要イベントなし → イベント条件はクリア**")
    st.write(
        "CPI、PCE、雇用統計、FOMCなど、市場を大きく動かす可能性のある米国イベントがある夜は、"
        "通常の日とは値動きの性質が変わることがあります。"
        "このアプリでは、そうした『イベントで一気に動くリスク』を避ける考え方を採用しています。"
    )
    st.warning(
        "⚠️ 経済指標カレンダーの自動取得には限界があります。重要イベントが『なし』と表示されても、"
        "実際の売買前には必ず主要経済指標と大型ハイテク企業の決算を手動確認してください。"
    )

with st.expander("⑥ 4条件をどう組み合わせている？", expanded=True):
    st.write(
        "この戦略は『上がる指標を1つ当てる』のではなく、異なる役割のフィルターを重ねています。"
    )
    st.markdown(
        "- **NASDAQ100先物** → 米国ハイテク株の方向感\n"
        "- **SOX** → 半導体セクターの強さ\n"
        "- **ドル円** → 円高による日経への逆風をチェック\n"
        "- **VIX** → 不安定な相場を避ける\n"
        "- **重要イベント** → 夜間の急変リスクを避ける"
    )
    if cond_event and cond_nq and cond_sox and cond_usd and cond_vix:
        st.success("🟢 現在は5つのフィルターをすべてクリアしています。『買い候補』という判定です。")
    else:
        failed = []
        if not cond_event: failed.append("米国重要イベント")
        if not cond_nq: failed.append("NASDAQ100先物")
        if not cond_sox: failed.append("SOX")
        if not cond_usd: failed.append("ドル円")
        if not cond_vix: failed.append("VIX")
        st.warning("🟡 未達のフィルター：" + "、".join(failed) + "。この戦略では見送りです。")

with st.expander("⑦ バックテストの数字をどう読む？"):
    st.write(
        "バックテストは『過去に同じ条件なら、翌営業日の寄り付きまで持った場合どうなったか』を見るものです。"
    )
    st.markdown(
        "- **取引数**：条件を満たした日数。少なすぎると偶然の影響が大きくなります。\n"
        "- **勝率**：翌朝のリターンがプラスだった割合。\n"
        "- **期待値 / 1回**：1回仕掛けたときの平均リターン。勝率だけでなく、勝ち幅・負け幅も反映されます。\n"
        "- **平均勝ち／平均負け**：1回の勝ち・負けが平均で何％だったか。\n"
        "- **複利リターン**：各トレードのリターンを順番に複利で積み上げた結果。\n"
        "- **最大DD**：資産が直前の最高水準からどれだけ落ち込んだか。『どの程度の苦しい期間があり得たか』を見る数字です。"
    )
    st.info(
        "📌 大切なのは『勝率だけ』ではありません。勝率・期待値・最大DD・取引回数を一緒に見て、"
        "実運用の結果と長期的に比較します。"
    )

# =========================================================
# バックテスト
# =========================================================
st.markdown("---")
st.header("② バックテスト")
st.write(
    "本日の判定条件と同じ4条件を過去データに適用します。"
    "各日の大引け時点で既知だった米国データを使い、"
    "翌営業日の1570相当の寄り付き方向を検証します。"
)

years = list(range(2018, datetime.now().year + 1))
c1, c2, c3, c4 = st.columns(4)
start_year = c1.selectbox("開始年", years, index=max(0, len(years)-4))
start_month = c2.selectbox("開始月", list(range(1, 13)), index=0)
end_year = c3.selectbox("終了年", years, index=len(years)-1)
end_month = c4.selectbox("終了月", list(range(1, 13)), index=datetime.now().month-1)

if st.button("🔄 バックテストを実行", type="secondary"):
    with st.spinner("過去データを取得して検証中..."):
        bt = load_backtest_data()

    start_date = pd.Timestamp(start_year, start_month, 1)
    if end_month == 12:
        end_date = pd.Timestamp(end_year + 1, 1, 1)
    else:
        end_date = pd.Timestamp(end_year, end_month + 1, 1)

    bt = bt[(bt.index >= start_date) & (bt.index < end_date)].copy()

    bt["Signal"] = (
        (bt["NQ_Close_pct"] >= p_nq)
        & (bt["SOX_Close_pct"] >= p_sox)
        & (bt["USD_Close_pct"] > p_usd)
        & (bt["VIX_Close"] < p_vix)
    )

    trades = bt[bt["Signal"]].copy()

    if len(trades) == 0:
        st.warning("条件に一致するトレードがありませんでした。")
    else:
        r = trades["Night_Return"]
        wins = int((r > 0).sum())
        losses = int((r <= 0).sum())
        win_rate = wins / len(r) * 100
        cumulative = (1 + r / 100).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative / peak - 1) * 100
        max_dd = float(dd.min())
        total_return = float((cumulative.iloc[-1] - 1) * 100)
        avg = float(r.mean())
        avg_win = float(r[r > 0].mean()) if wins else 0
        avg_loss = float(r[r <= 0].mean()) if losses else 0
        profit_factor = (
            float(r[r > 0].sum() / abs(r[r <= 0].sum()))
            if losses and r[r <= 0].sum() != 0 else float("inf")
        )
        max_loss_streak = 0
        cur = 0
        for x in r:
            if x <= 0:
                cur += 1
                max_loss_streak = max(max_loss_streak, cur)
            else:
                cur = 0

        # 7項目を1列に詰めると、Streamlitの画面幅によって
        # 「+0.001%」などが「+0.0...」と省略表示されるため、
        # 4項目＋3項目の2段に分けて、数字を最後まで表示する。
        m1 = st.columns(4)
        m1[0].metric("取引数", f"{len(r)}回")
        m1[1].metric("勝率", f"{win_rate:.1f}%")
        m1[2].metric("複利リターン", f"{total_return:+.2f}%")
        m1[3].metric("最大DD", f"{max_dd:.2f}%")

        m2 = st.columns(3)
        m2[0].metric("期待値 / 1回", f"{avg:+.3f}%")
        m2[1].metric("平均勝ち", f"{avg_win:+.3f}%")
        m2[2].metric("平均負け", f"{avg_loss:+.3f}%")

        st.caption(
            f"勝敗 {wins}勝 / {losses}敗 ｜ 利益率(Profit Factor) "
            f"{profit_factor:.2f} ｜ 最大連敗 {max_loss_streak}回"
        )

        equity = pd.DataFrame({"資産倍率": cumulative}, index=trades.index)
        st.line_chart(equity)

        history = trades[[
            "NQ_Close_pct", "SOX_Close_pct",
            "USD_Close_pct", "VIX_Close", "Night_Return"
        ]].copy()
        history.columns = [
            "NASDAQ100先物(%)", "SOX(%)", "ドル円(%)",
            "VIX", "翌朝寄りリターン(%)"
        ]
        st.subheader("📋 トレード履歴")
        st.dataframe(
            history.sort_index(ascending=False),
            use_container_width=True
        )

st.markdown("---")
st.caption(
    "※これは統計的な検証用アプリです。実際の売買では、1570の板・スリッページ・"
    "手数料・寄り付き約定価格との差などが発生します。"
)
