import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime, timedelta
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

FEATURES = {
    "n225_ret": ("当日日経225騰落率", "日経225"),
    "n225_close_pos": ("当日日経225終値位置", "日経225"),
    "topix_ret": ("当日TOPIX騰落率", "TOPIX"),
    "nq_ret": ("前回米国NASDAQ100騰落率", "NASDAQ100"),
    "sp_ret": ("前回米国S&P500騰落率", "S&P500"),
    "sox_ret": ("前回米国SOX騰落率", "SOX"),
    "dow_ret": ("前回米国NYダウ騰落率", "NYダウ"),
    "vix_level": ("前回米国VIX水準", "VIX"),
    "usd_ret": ("前回ドル円騰落率", "ドル円"),
    "tnx_ret": ("前回米10年金利変化率", "米10年金利"),
    "us_breadth": ("米国4指数プラス数", "米国4指数"),
    "nq_usd_combo": ("NASDAQ100騰落率＋ドル円騰落率", "NASDAQ×ドル円"),
}

EXPLANATIONS = {
    "n225_ret": "エントリー当日（1570を大引けで買う日）の日経225騰落です。大きく下げた後に翌朝反発しやすいか、逆に上昇の勢いが続きやすいかを見ます。",
    "n225_close_pos": "エントリー当日の高値〜安値のどこで日経225が引けたかです。1に近いほど高値圏、0に近いほど安値圏で引けています。",
    "topix_ret": "日経225より広い日本株全体の、エントリー当日の動きです。日本市場全体の強弱が翌朝に残るかを見ます。",
    "nq_ret": "直前に終了した米国NASDAQ100の騰落です。日本のハイテク・半導体株に波及しやすい情報です。",
    "sp_ret": "直前のS&P500です。米国株全体のリスクオン・リスクオフを表す代表的な情報です。",
    "sox_ret": "米国半導体株の動きです。日経平均への寄与が大きい日本の半導体株との関連を調べます。",
    "dow_ret": "米国大型株の代表指数です。NASDAQとは違う業種も含むため、米国市場の広がりを確認できます。",
    "vix_level": "市場の不安の大きさです。低いほど安全とは限らず、高い局面で翌朝反発しやすい可能性も調べます。",
    "usd_ret": "ドル円の変化です。プラスは円安方向、マイナスは円高方向。輸出株の多い日本市場との関係を見ます。",
    "tnx_ret": "米10年金利の変化です。金利上昇・低下が株式市場の評価や為替を通じて翌朝に影響するかを見ます。",
    "us_breadth": "NASDAQ100・S&P500・SOX・NYダウのうち何指数が上昇したかです。米国株の上昇・下落の『広がり』を見ます。",
    "nq_usd_combo": "NASDAQ100とドル円を足した簡易指標です。米ハイテク株と円安・円高が同時に日本株へどう効くかを見ます。",
}

@st.cache_data(ttl=1800)
def download_daily(ticker, start, end):
    df = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, threads=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index().dropna(how="all")

def close_s(df):
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["Close"], errors="coerce").dropna()

def map_prior_us_feature(jp_dates, s, name):
    """日本日付Dに対し、Dより前に終了している直近の米国セッション値を割り当てる。"""
    left = pd.DataFrame({"jp_date": pd.to_datetime(jp_dates).normalize()}).sort_values("jp_date")
    right = pd.DataFrame({
        "us_date": pd.to_datetime(s.index).normalize(),
        name: pd.to_numeric(s.values, errors="coerce")
    }).dropna().sort_values("us_date")
    m = pd.merge_asof(
        left, right, left_on="jp_date", right_on="us_date",
        direction="backward", allow_exact_matches=False
    )
    return pd.Series(m[name].values, index=left["jp_date"].values)

@st.cache_data(ttl=3600)
def build_research(start, end):
    start_dt = pd.Timestamp(start) - pd.Timedelta(days=20)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=3)
    raw = {k: download_daily(v, start_dt.date().isoformat(), end_dt.date().isoformat())
           for k, v in TICKERS.items()}

    etf = raw["1570"]
    if etf.empty:
        return pd.DataFrame()

    jp = pd.DataFrame(index=etf.index.copy())
    jp.index.name = "date"
    jp["etf_open"] = pd.to_numeric(etf["Open"], errors="coerce")
    jp["etf_close"] = pd.to_numeric(etf["Close"], errors="coerce")

    # 日本市場情報（当日の大引けまでに既知）
    n225 = raw["日経225"].reindex(jp.index)
    topix = raw["TOPIX"].reindex(jp.index)
    jp["n225_close"] = pd.to_numeric(n225.get("Close"), errors="coerce")
    jp["n225_ret"] = jp["n225_close"].pct_change() * 100
    hi = pd.to_numeric(n225.get("High"), errors="coerce")
    lo = pd.to_numeric(n225.get("Low"), errors="coerce")
    jp["n225_close_pos"] = ((jp["n225_close"] - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    jp["topix_ret"] = close_s(topix).reindex(jp.index).pct_change() * 100

    # 米国市場情報：日本日付Dより前に終了した直近米国セッション
    us_returns = {
        "nq_ret": close_s(raw["NASDAQ100"]).pct_change() * 100,
        "sp_ret": close_s(raw["S&P500"]).pct_change() * 100,
        "sox_ret": close_s(raw["SOX"]).pct_change() * 100,
        "dow_ret": close_s(raw["NYダウ"]).pct_change() * 100,
        "usd_ret": close_s(raw["ドル円"]).pct_change() * 100,
        "tnx_ret": close_s(raw["米10年金利"]).pct_change() * 100,
    }
    for feat, s in us_returns.items():
        jp[feat] = map_prior_us_feature(jp.index, s, feat).reindex(jp.index).values

    vix = close_s(raw["VIX"])
    jp["vix_level"] = map_prior_us_feature(jp.index, vix, "vix_level").reindex(jp.index).values

    jp["us_breadth"] = (
        (jp["nq_ret"] > 0).astype(int)
        + (jp["sp_ret"] > 0).astype(int)
        + (jp["sox_ret"] > 0).astype(int)
        + (jp["dow_ret"] > 0).astype(int)
    )
    jp["nq_usd_combo"] = jp["nq_ret"] + jp["usd_ret"]

    # 目的変数：当日大引け買い → 翌営業日寄り売り
    jp["next_open"] = jp["etf_open"].shift(-1)
    jp["target_next_open_ret"] = (jp["next_open"] / jp["etf_close"] - 1) * 100
    jp["target_up"] = jp["target_next_open_ret"] > 0

    jp = jp[(jp.index >= pd.Timestamp(start)) & (jp.index <= pd.Timestamp(end))]
    return jp.dropna(subset=["etf_close", "next_open", "target_next_open_ret"])

def stats(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return {"n":0,"win":np.nan,"mean":np.nan,"median":np.nan,"std":np.nan,"pf":np.nan,"cum":np.nan}
    wins = x[x > 0]
    losses = x[x <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {
        "n": len(x),
        "win": (x > 0).mean() * 100,
        "mean": x.mean(),
        "median": x.median(),
        "std": x.std(ddof=1) if len(x) > 1 else np.nan,
        "pf": pf,
        "cum": ((1 + x/100).prod() - 1) * 100,
    }

def welch_t(mean1, std1, n1, mean2, std2, n2):
    """条件付き部分集合の平均 と 全体（無条件）平均 を比較するWelchのt統計量。
    部分集合は全体に内包される（独立でない）ため厳密な検定ではないが、
    『改善幅がサンプルの散らばりに対してどれだけ大きいか』を測る目安として使う。"""
    if any(pd.isna(v) for v in [mean1, std1, n1, mean2, std2, n2]) or n1 < 2 or n2 < 2:
        return np.nan
    se = np.sqrt((std1**2)/n1 + (std2**2)/n2)
    if se == 0 or np.isnan(se):
        return np.nan
    return (mean1 - mean2) / se

def candidate_mask(df, feat, direction, threshold):
    if direction == "low":
        return df[feat] <= threshold
    return df[feat] >= threshold

QUANTILES = [
    (0.20, "low"), (0.30, "low"), (0.40, "low"),
    (0.60, "high"), (0.70, "high"), (0.80, "high"),
]

def bonferroni_z(n_tests, alpha=0.05):
    """72条件（特徴量×分位点）を同時に検定するため、有意水準をBonferroni補正する。
    補正後のalphaに対応する両側z臨界値を返す。"""
    n_tests = max(int(n_tests), 1)
    adj_alpha = alpha / n_tests
    return float(scipy_stats.norm.ppf(1 - adj_alpha / 2))

def discover_candidates(train, valid, min_n, use_correction=True):
    base_tr = stats(train["target_next_open_ret"])
    base_va = stats(valid["target_next_open_ret"])
    rows = []

    total_tests = len(FEATURES) * len(QUANTILES)
    z_crit = bonferroni_z(total_tests) if use_correction else 1.0
    valid_min_n = max(30, min_n // 2)

    for feat, (label, _) in FEATURES.items():
        s = train[feat].dropna()
        if len(s) < min_n:
            continue

        for q, direction in QUANTILES:
            threshold = s.quantile(q)
            mtr = candidate_mask(train, feat, direction, threshold)
            mva = candidate_mask(valid, feat, direction, threshold)

            tr = stats(train.loc[mtr, "target_next_open_ret"])
            va = stats(valid.loc[mva, "target_next_open_ret"])

            if tr["n"] < min_n:
                continue

            symbol = "≤" if direction == "low" else "≥"
            cond = f"{symbol} {threshold:.3f}"

            train_edge = tr["mean"] - base_tr["mean"]
            valid_edge = va["mean"] - base_va["mean"] if va["n"] else np.nan

            # 検証期間の条件付き平均が「無条件平均」から統計的にどれだけ離れているか。
            # 72条件を同時に見ているため、単に valid_edge > 0 だけでは偶然一致が混ざりやすい。
            # Bonferroni補正したz臨界値を超えるものだけを厳しく「再現」とみなす。
            t_valid = welch_t(va["mean"], va["std"], va["n"], base_va["mean"], base_va["std"], base_va["n"])
            significant = (not np.isnan(t_valid)) and abs(t_valid) >= z_crit

            enough_n = va["n"] >= valid_min_n

            if enough_n and train_edge > 0 and valid_edge > 0 and significant:
                verdict = "🟢 再現候補"
            elif enough_n and train_edge > 0 and valid_edge > 0:
                verdict = "🟡 改善はあるが有意性不足"
            elif train_edge > 0 and enough_n:
                verdict = "🟡 検証で再現せず"
            else:
                verdict = "⚪ 参考"

            rows.append({
                "feature": feat,
                "情報": label,
                "条件": cond,
                "direction": direction,
                "threshold": threshold,
                "発見件数": tr["n"],
                "発見上昇率": tr["win"],
                "発見平均": tr["mean"],
                "発見改善": train_edge,
                "検証件数": va["n"],
                "検証上昇率": va["win"],
                "検証平均": va["mean"],
                "検証改善": valid_edge,
                "検証PF": va["pf"],
                "検証t値": t_valid,
                "判定": verdict,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, base_tr, base_va, total_tests, z_crit

    # 発見と検証の両方で基準超えした候補を優先
    out["再現スコア"] = (
        out["発見改善"].clip(lower=-9) +
        out["検証改善"].fillna(-9).clip(lower=-9)
    )
    out = out.sort_values(
        ["判定", "再現スコア", "検証平均", "発見平均"],
        ascending=[True, False, False, False]
    )
    # 文字列順にならないよう独自順序で再ソート
    rank = {"🟢 再現候補":0, "🟡 改善はあるが有意性不足":1, "🟡 検証で再現せず":2, "⚪ 参考":3}
    out["_rank"] = out["判定"].map(rank).fillna(9)
    out = out.sort_values(["_rank","再現スコア"], ascending=[True,False]).drop(columns="_rank")
    return out, base_tr, base_va, total_tests, z_crit

def plain_explanation(row, base_tr, base_va):
    feat = row["feature"]
    direction_text = "以下" if row["direction"] == "low" else "以上"
    label = row["情報"]
    t = row["threshold"]
    return (
        f"**{label} が {t:.3f}{direction_text}**だった日を取り出した条件です。  \n"
        f"発見期間では平均 {row['発見平均']:+.3f}%（無条件 {base_tr['mean']:+.3f}%）、"
        f"検証期間では平均 {row['検証平均']:+.3f}%（無条件 {base_va['mean']:+.3f}%）でした。  \n"
        f"{EXPLANATIONS.get(feat,'')}  \n"
        f"**{row['判定']}**：これはまだ売買ルールではなく、Aアプリに入れる前の研究候補です。"
    )

# =========================================================
# UI
# =========================================================
st.title("🔬 B_翌朝寄り付き研究アプリ")
st.caption("『何が翌朝の1570に効くのか』を、発見期間と検証期間を分けて調べます。")

st.info(
    "このBアプリの目的は『一番良かった条件を見つける』ことではなく、"
    "**過去の前半で見つけた条件が、後半の未使用データでも再現するか**を確認することです。"
)

today = datetime.now(JST).date()

with st.sidebar:
    st.header("🔧 研究条件")
    st.markdown("**① 条件を発見する期間**")
    discover_start = st.date_input("発見開始", datetime(2018,1,1).date(), key="ds")
    discover_end = st.date_input("発見終了", datetime(2023,12,31).date(), key="de")

    st.markdown("**② 答え合わせする期間**")
    validate_start = st.date_input("検証開始", datetime(2024,1,1).date(), key="vs")
    validate_end = st.date_input("検証終了", today, key="ve")

    min_n = st.number_input("発見期間の最低サンプル数", 20, 300, 80, 10)

    st.markdown("**③ 判定の厳しさ**")
    use_correction = st.checkbox(
        "多重検定を補正する（Bonferroni・推奨）", value=True,
        help="72個の条件（特徴量×分位点）を同時に試すため、補正しないと偶然良く見える条件が"
             "🟢再現候補に紛れ込みやすくなります。オフにすると単純に「両期間で改善>0」だけで判定します。"
    )
    run = st.button("🔄 研究を実行", type="primary")

if discover_end >= validate_start:
    st.error(
        "発見期間と検証期間が重なっています（または順序が逆です）。"
        "検証期間は発見期間より後の日付にしてください。"
    )
    st.stop()

all_start = min(discover_start, validate_start)
all_end = max(discover_end, validate_end)

if run or "B02_df" not in st.session_state:
    with st.spinner("研究データを取得・整理しています…"):
        try:
            st.session_state["B02_df"] = build_research(all_start.isoformat(), all_end.isoformat())
        except Exception as e:
            st.error(f"データ取得エラー：{e}")
            st.stop()

df = st.session_state["B02_df"]
if df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

train = df[(df.index >= pd.Timestamp(discover_start)) & (df.index <= pd.Timestamp(discover_end))].copy()
valid = df[(df.index >= pd.Timestamp(validate_start)) & (df.index <= pd.Timestamp(validate_end))].copy()

result, base_tr, base_va, total_tests, z_crit = discover_candidates(
    train, valid, int(min_n), use_correction=use_correction
)

if use_correction:
    st.caption(
        f"🧪 多重検定補正：今回は {total_tests} 個の条件を同時に試すため、"
        f"検証期間での改善は目安として概ね z ≥ {z_crit:.2f} 相当（Bonferroni補正後の有意水準）"
        "を満たすものだけを🟢再現候補としています。"
    )

# 1. やさしい全体像
st.header("① まず、何を比べているの？")
st.write(
    "毎日無条件で1570を大引けに買って翌朝寄りで売った場合を**基準**にします。"
    "その基準より『特定の条件の日だけ買う』ほうが良くなるかを調べています。"
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("🔎 発見期間")
    st.caption(f"{discover_start} ～ {discover_end}")
    a = st.columns(3)
    a[0].metric("件数", f"{base_tr['n']:,}")
    a[1].metric("無条件の上昇率", f"{base_tr['win']:.1f}%")
    a[2].metric("無条件の平均", f"{base_tr['mean']:+.3f}%")
with c2:
    st.subheader("✅ 検証期間")
    st.caption(f"{validate_start} ～ {validate_end}")
    a = st.columns(3)
    a[0].metric("件数", f"{base_va['n']:,}")
    a[1].metric("無条件の上昇率", f"{base_va['win']:.1f}%")
    a[2].metric("無条件の平均", f"{base_va['mean']:+.3f}%")

st.caption(
    "例：検証期間の無条件平均が +0.10% なら、候補条件の検証平均が +0.25% なら『+0.15%改善』と見ます。"
)

# 2. 今日の研究で分かったこと
st.header("② 今日の研究で分かったこと")

if result.empty:
    st.warning("候補がありません。期間または最低サンプル数を調整してください。")
else:
    reproducible = result[result["判定"] == "🟢 再現候補"].copy()

    if reproducible.empty:
        st.warning(
            "今回は『発見期間で良く、検証期間でも基準より良かった』条件が見つかりませんでした。"
            "これは失敗ではなく、過学習を避けるための重要な結果です。"
        )
    else:
        top = reproducible.iloc[0]
        st.success(
            f"⭐ 最も注目する再現候補：**{top['情報']} / {top['条件']}**  "
            f"｜検証 {int(top['検証件数'])}件・上昇率 {top['検証上昇率']:.1f}%・"
            f"平均 {top['検証平均']:+.3f}%"
        )
        st.markdown(plain_explanation(top, base_tr, base_va))

        st.subheader("注目候補 TOP5")
        for _, row in reproducible.head(5).iterrows():
            with st.expander(
                f"{row['情報']} / {row['条件']}  → 検証平均 {row['検証平均']:+.3f}% "
                f"（改善 {row['検証改善']:+.3f}%）"
            ):
                st.markdown(plain_explanation(row, base_tr, base_va))

# 3. 比較表
st.header("③ 発見期間と検証期間を並べて見る")
st.write(
    "ここがBアプリの中心です。**発見期間だけ良い条件は信用しません。検証期間でも同じ方向に改善するか**を見ます。"
)

if not result.empty:
    show = result.copy()
    cols = [
        "判定","情報","条件",
        "発見件数","発見上昇率","発見平均","発見改善",
        "検証件数","検証上昇率","検証平均","検証改善","検証PF","検証t値"
    ]
    show = show[cols].head(40)

    for col in ["発見上昇率","検証上昇率"]:
        show[col] = show[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}%")
    for col in ["発見平均","発見改善","検証平均","検証改善"]:
        show[col] = show[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.3f}%")
    show["検証PF"] = show["検証PF"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    show["検証t値"] = show["検証t値"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

# 4. 表の読み方
st.header("④ 表はこう読む")
st.markdown("""
**🟢 再現候補**  
発見期間で無条件より良く、検証期間でも無条件より良く、
かつ多重検定補正後の基準でも統計的に意味のある差でした。  
→ 次に詳しく調べる価値があります。

**🟡 改善はあるが有意性不足**  
両期間で改善はプラスですが、条件の数（72通り）を考慮すると偶然の範囲内かもしれません。  
→ 参考にはなりますが、単独では採用根拠として弱いです。

**🟡 検証で再現せず**  
発見期間では良かったのに、未使用の検証期間では基準を上回りませんでした。  
→ 過去に偶然よく見えただけの可能性があります。

**⚪ 参考**  
現時点では優位性が弱い条件です。

**「改善」**  
その期間の無条件平均と比べて、条件を付けることで平均リターンが何%上がったかです。  
ここを特に重視します。

**「検証t値」**  
検証期間の条件付き平均が、無条件平均からサンプルのばらつきに対してどれだけ離れているかの目安です。  
絶対値が大きいほど「偶然とは考えにくい差」であることを示します（厳密な独立検定ではなく目安）。
""")

# 5. 個別候補を学ぶ
st.header("⑤ 気になる候補を1つ選んで勉強する")
if not result.empty:
    options = [
        f"{i+1}. {r['情報']} / {r['条件']} / {r['判定']}"
        for i, (_, r) in enumerate(result.head(40).iterrows())
    ]
    selected = st.selectbox("候補", options)
    idx = int(selected.split(".")[0]) - 1
    row = result.head(40).iloc[idx]
    st.markdown(plain_explanation(row, base_tr, base_va))

    x1 = train.loc[candidate_mask(train, row["feature"], row["direction"], row["threshold"]), "target_next_open_ret"]
    x2 = valid.loc[candidate_mask(valid, row["feature"], row["direction"], row["threshold"]), "target_next_open_ret"]

    cc = st.columns(2)
    with cc[0]:
        st.subheader("発見期間")
        st.metric("平均", f"{x1.mean():+.3f}%")
        st.metric("上昇率", f"{(x1>0).mean()*100:.1f}%")
    with cc[1]:
        st.subheader("検証期間")
        st.metric("平均", f"{x2.mean():+.3f}%" if len(x2) else "—")
        st.metric("上昇率", f"{(x2>0).mean()*100:.1f}%" if len(x2) else "—")

# 6. 次に何をするか
st.header("⑥ 次に何をすればいい？")
if not result.empty and not result[result["判定"] == "🟢 再現候補"].empty:
    st.write(
        "まず🟢再現候補を2〜3個に絞ります。次の版では、その候補を**2条件組み合わせても再現するか**を調べます。"
        "それでも安定していれば、初めてAアプリへの採用候補にします。"
    )
else:
    st.write(
        "今回は強い再現候補がありません。条件の閾値を細かく探しすぎず、"
        "次は特徴量そのもの（日経先物など）を追加する方が有効です。"
    )

st.warning(
    "⚠️ 重要：Bアプリの結果だけで売買条件を変更しません。"
    "『発見 → 未使用期間で検証 → 実運用でフォワード確認』の順で進めます。"
)
st.caption(f"最終更新：{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')} JST")
