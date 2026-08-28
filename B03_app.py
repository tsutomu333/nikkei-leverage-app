import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="B03_2条件組み合わせ研究", page_icon="🧩", layout="wide")
JST = ZoneInfo("Asia/Tokyo")

# =========================================================
# B_app.py（B02.1）と同じ基礎ロジック
# 既存のB_app.pyは変更せず、このファイルは独立して動く新バージョンです。
# 目的：単一条件の🟢再現候補どうしを2条件組み合わせ（AND）にしても、
#       発見期間・検証期間の両方で再現するかを検証します。
# =========================================================

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

    n225 = raw["日経225"].reindex(jp.index)
    topix = raw["TOPIX"].reindex(jp.index)
    jp["n225_close"] = pd.to_numeric(n225.get("Close"), errors="coerce")
    jp["n225_ret"] = jp["n225_close"].pct_change() * 100
    hi = pd.to_numeric(n225.get("High"), errors="coerce")
    lo = pd.to_numeric(n225.get("Low"), errors="coerce")
    jp["n225_close_pos"] = ((jp["n225_close"] - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    jp["topix_ret"] = close_s(topix).reindex(jp.index).pct_change() * 100

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

    jp["next_open"] = jp["etf_open"].shift(-1)
    jp["target_next_open_ret"] = (jp["next_open"] / jp["etf_close"] - 1) * 100
    jp["target_up"] = jp["target_next_open_ret"] > 0

    jp = jp[(jp.index >= pd.Timestamp(start)) & (jp.index <= pd.Timestamp(end))]
    return jp.dropna(subset=["etf_close", "next_open", "target_next_open_ret"])

def norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p <= 0 or p >= 1:
        return np.nan
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

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
    n_tests = max(int(n_tests), 1)
    adj_alpha = alpha / n_tests
    return float(norm_ppf(1 - adj_alpha / 2))

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
                "feature": feat, "情報": label, "条件": cond,
                "direction": direction, "threshold": threshold,
                "発見件数": tr["n"], "発見平均": tr["mean"], "発見改善": train_edge,
                "検証件数": va["n"], "検証平均": va["mean"], "検証改善": valid_edge,
                "検証t値": t_valid, "判定": verdict,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, base_tr, base_va, total_tests, z_crit

    out["再現スコア"] = out["発見改善"].clip(lower=-9) + out["検証改善"].fillna(-9).clip(lower=-9)
    rank = {"🟢 再現候補":0, "🟡 改善はあるが有意性不足":1, "🟡 検証で再現せず":2, "⚪ 参考":3}
    out["_rank"] = out["判定"].map(rank).fillna(9)
    out = out.sort_values(["_rank","再現スコア"], ascending=[True,False]).drop(columns="_rank")
    return out, base_tr, base_va, total_tests, z_crit

# =========================================================
# ここからB03独自：2条件組み合わせ（異なる特徴量のAND）検証
# =========================================================

def pick_combo_pool(single_result, max_pool=10):
    """組み合わせ対象のプールを選ぶ。
    まず🟢再現候補を優先。2個未満なら🟡改善はあるが有意性不足も加える。
    それでも少なければ再現スコア上位で埋める。特徴量が重複しないよう
    同一特徴量は最良の1条件だけ残す。"""
    if single_result.empty:
        return single_result.iloc[0:0], "no_single_candidates"

    green = single_result[single_result["判定"] == "🟢 再現候補"]
    yellow_ok = single_result[single_result["判定"] == "🟡 改善はあるが有意性不足"]

    pool = green.copy()
    source = "green_only"
    if len(pool) < 2:
        pool = pd.concat([pool, yellow_ok]).drop_duplicates()
        source = "green_plus_yellow"
    if len(pool) < 2:
        pool = single_result[single_result["発見改善"] > 0].copy()
        source = "fallback_top_positive"

    # 特徴量が重複する場合は再現スコアが高い方だけ残す
    pool = pool.sort_values("再現スコア", ascending=False).drop_duplicates(subset="feature", keep="first")
    pool = pool.head(max_pool)
    return pool, source

def discover_combo_candidates(train, valid, pool, base_tr, base_va, min_n, use_correction=True):
    rows = []
    pool = pool.reset_index(drop=True)
    pairs = list(combinations(range(len(pool)), 2))
    total_tests = max(len(pairs), 1)
    z_crit = bonferroni_z(total_tests) if use_correction else 1.0
    valid_min_n = max(20, min_n // 3)

    for i, j in pairs:
        r1 = pool.iloc[i]
        r2 = pool.iloc[j]
        if r1["feature"] == r2["feature"]:
            continue

        mtr = candidate_mask(train, r1["feature"], r1["direction"], r1["threshold"]) & \
              candidate_mask(train, r2["feature"], r2["direction"], r2["threshold"])
        mva = candidate_mask(valid, r1["feature"], r1["direction"], r1["threshold"]) & \
              candidate_mask(valid, r2["feature"], r2["direction"], r2["threshold"])

        tr = stats(train.loc[mtr, "target_next_open_ret"])
        va = stats(valid.loc[mva, "target_next_open_ret"])

        if tr["n"] < min_n:
            continue

        train_edge = tr["mean"] - base_tr["mean"]
        valid_edge = va["mean"] - base_va["mean"] if va["n"] else np.nan

        t_valid = welch_t(va["mean"], va["std"], va["n"], base_va["mean"], base_va["std"], base_va["n"])
        significant = (not np.isnan(t_valid)) and abs(t_valid) >= z_crit
        enough_n = va["n"] >= valid_min_n

        # 単体条件それぞれの検証平均より、組み合わせが上回っているか（付加価値の目安）
        beats_single = (
            (not pd.isna(va["mean"])) and
            (not pd.isna(r1["検証平均"])) and (not pd.isna(r2["検証平均"])) and
            va["mean"] > max(r1["検証平均"], r2["検証平均"])
        )

        if enough_n and train_edge > 0 and valid_edge > 0 and significant:
            verdict = "🟢 2条件再現候補"
        elif enough_n and train_edge > 0 and valid_edge > 0:
            verdict = "🟡 改善はあるが有意性不足"
        elif train_edge > 0 and enough_n:
            verdict = "🟡 検証で再現せず"
        else:
            verdict = "⚪ 参考"

        rows.append({
            "組み合わせ": f"{r1['情報']}{r1['条件']} × {r2['情報']}{r2['条件']}",
            "特徴量1": r1["feature"], "条件1": r1["条件"],
            "特徴量2": r2["feature"], "条件2": r2["条件"],
            "発見件数": tr["n"], "発見平均": tr["mean"], "発見改善": train_edge,
            "検証件数": va["n"], "検証平均": va["mean"], "検証改善": valid_edge,
            "検証t値": t_valid, "単体より改善": beats_single, "判定": verdict,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, total_tests, z_crit

    out["再現スコア"] = out["発見改善"].clip(lower=-9) + out["検証改善"].fillna(-9).clip(lower=-9)
    rank = {"🟢 2条件再現候補":0, "🟡 改善はあるが有意性不足":1, "🟡 検証で再現せず":2, "⚪ 参考":3}
    out["_rank"] = out["判定"].map(rank).fillna(9)
    out = out.sort_values(["_rank","再現スコア"], ascending=[True,False]).drop(columns="_rank")
    return out, total_tests, z_crit

# =========================================================
# UI
# =========================================================
st.title("🧩 B03_2条件組み合わせ研究")
st.caption("B02で見つかった単一条件の🟢再現候補どうしを2条件組み合わせ（AND）にしても、発見期間・検証期間の両方で再現するかを検証します。")

st.info(
    "B_app.py（B02）は変更していません。これは別バージョンとして追加した新しいアプリです。\n\n"
    "手順：①B02と同じロジックで単一条件を発見・検証 → ②有望だった単一条件どうしのペア（異なる特徴量のAND）を作成 "
    "→ ③そのペアも発見期間・検証期間の両方で無条件平均より改善し、統計的にも有意かを確認します。"
)

today = datetime.now(JST).date()

with st.sidebar:
    st.header("🔧 研究条件")
    discover_start = st.date_input("発見開始", datetime(2018,1,1).date(), key="ds")
    discover_end = st.date_input("発見終了", datetime(2023,12,31).date(), key="de")
    validate_start = st.date_input("検証開始", datetime(2024,1,1).date(), key="vs")
    validate_end = st.date_input("検証終了", today, key="ve")
    min_n = st.number_input("発見期間の最低サンプル数（単一条件）", 20, 300, 80, 10)
    max_pool = st.number_input("組み合わせに使う単一候補の最大数", 2, 20, 10, 1)
    use_correction = st.checkbox("多重検定を補正する（Bonferroni・推奨）", value=True)
    run = st.button("🔄 研究を実行", type="primary")

if discover_end >= validate_start:
    st.error("発見期間と検証期間が重なっています。検証期間は発見期間より後の日付にしてください。")
    st.stop()

all_start = min(discover_start, validate_start)
all_end = max(discover_end, validate_end)

if run or "B03_df" not in st.session_state:
    with st.spinner("研究データを取得・整理しています…"):
        try:
            st.session_state["B03_df"] = build_research(all_start.isoformat(), all_end.isoformat())
        except Exception as e:
            st.error(f"データ取得エラー：{e}")
            st.stop()

df = st.session_state["B03_df"]
if df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

train = df[(df.index >= pd.Timestamp(discover_start)) & (df.index <= pd.Timestamp(discover_end))].copy()
valid = df[(df.index >= pd.Timestamp(validate_start)) & (df.index <= pd.Timestamp(validate_end))].copy()

single_result, base_tr, base_va, single_tests, single_z = discover_candidates(
    train, valid, int(min_n), use_correction=use_correction
)

st.header("① 単一条件の発見・検証（B02と同じロジック）")
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔎 発見期間")
    st.caption(f"{discover_start} ～ {discover_end}")
    a = st.columns(3)
    a[0].metric("件数", f"{base_tr['n']:,}")
    a[1].metric("無条件の平均", f"{base_tr['mean']:+.3f}%")
with c2:
    st.subheader("✅ 検証期間")
    st.caption(f"{validate_start} ～ {validate_end}")
    a = st.columns(3)
    a[0].metric("件数", f"{base_va['n']:,}")
    a[1].metric("無条件の平均", f"{base_va['mean']:+.3f}%")

if single_result.empty:
    st.warning("単一条件の候補が見つかりませんでした。期間や最低サンプル数を見直してください。")
    st.stop()

n_single_green = int((single_result["判定"] == "🟢 再現候補").sum())
st.write(f"単一条件の🟢再現候補：**{n_single_green}件**（{single_tests}条件中、Bonferroni補正 z ≥ {single_z:.2f}）")

with st.expander("単一条件の結果を全て見る"):
    show1 = single_result.drop(columns=["feature","direction","threshold","再現スコア"])
    st.dataframe(show1, use_container_width=True, hide_index=True)

st.header("② 2条件組み合わせ（AND）の発見・検証")

pool, pool_source = pick_combo_pool(single_result, max_pool=int(max_pool))

if len(pool.drop_duplicates(subset="feature")) < 2:
    st.warning(
        "組み合わせを作るための単一条件候補が2件未満でした（特徴量が重複しないもの）。"
        "期間や最低サンプル数を調整して単一条件候補を増やしてください。"
    )
    st.stop()

pool_note = {
    "green_only": "🟢再現候補のみでペアを作成しました。",
    "green_plus_yellow": "🟢再現候補が2件未満だったため、🟡改善はあるが有意性不足も加えてペアを作成しました。",
    "fallback_top_positive": "🟢🟡が少なかったため、発見期間で改善が見られた上位条件でペアを作成しました（参考扱い）。",
}.get(pool_source, "")
st.caption(f"組み合わせ候補プール：{len(pool)}件の単一条件（特徴量の重複なし）。{pool_note}")

with st.expander("組み合わせに使った単一条件プールを見る"):
    st.dataframe(
        pool[["情報","条件","発見平均","発見改善","検証平均","検証改善","判定"]],
        use_container_width=True, hide_index=True
    )

combo_result, combo_tests, combo_z = discover_combo_candidates(
    train, valid, pool, base_tr, base_va, int(min_n), use_correction=use_correction
)

if combo_result.empty:
    st.warning("条件を満たす2条件組み合わせが見つかりませんでした（サンプル数不足の可能性があります）。")
else:
    st.caption(
        f"🧪 多重検定補正：今回試した2条件組み合わせは{combo_tests}通り。"
        f"Bonferroni補正後、検証期間の改善が概ね z ≥ {combo_z:.2f} 相当を満たすものだけを🟢2条件再現候補としています。"
    )

    green_combo = combo_result[combo_result["判定"] == "🟢 2条件再現候補"]

    if green_combo.empty:
        st.warning(
            "単一条件では良かったものも、2条件を組み合わせると「発見期間・検証期間の両方で有意に改善」という"
            "基準を満たす組み合わせは見つかりませんでした。これは過学習を避けられている、という意味で悪い結果ではありません。"
        )
    else:
        top = green_combo.iloc[0]
        st.success(
            f"⭐ 最も注目する2条件組み合わせ：**{top['組み合わせ']}**  \n"
            f"検証 {int(top['検証件数'])}件・平均 {top['検証平均']:+.3f}%（無条件比 {top['検証改善']:+.3f}%）・"
            f"検証t値 {top['検証t値']:.2f}"
        )
        st.subheader("🟢 2条件再現候補 一覧")
        show2 = green_combo.drop(columns=["特徴量1","特徴量2","再現スコア"])
        for col in ["発見平均","発見改善","検証平均","検証改善"]:
            show2[col] = show2[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.3f}%")
        show2["検証t値"] = show2["検証t値"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        st.dataframe(show2, use_container_width=True, hide_index=True)

    st.subheader("全組み合わせ結果（上位40件）")
    show_all = combo_result.drop(columns=["特徴量1","特徴量2","再現スコア"]).head(40).copy()
    for col in ["発見平均","発見改善","検証平均","検証改善"]:
        show_all[col] = show_all[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.3f}%")
    show_all["検証t値"] = show_all["検証t値"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(show_all, use_container_width=True, hide_index=True)

st.header("③ 表の読み方")
st.markdown("""
**🟢 2条件再現候補**
発見期間・検証期間の両方で無条件平均より改善し、組み合わせ数を考慮したBonferroni補正後でも
統計的に意味のある差でした。単一条件より一段階慎重な基準です。

**単体より改善**
組み合わせた検証期間の平均が、単一条件それぞれの検証平均より良いかどうかです。
Falseの場合、組み合わせても単一条件以上の効果は出ていない（条件を増やす意味が薄い）ことを示します。

**「検証t値」**
検証期間の条件付き平均が、無条件平均からどれだけ離れているかの目安（厳密な独立検定ではありません）。
""")

st.warning(
    "⚠️ 重要：この結果だけで売買条件を変更しません。"
    "『発見 → 未使用期間で検証 → 実運用でフォワード確認』の順で進めます。"
)
st.caption(f"最終更新：{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')} JST")
