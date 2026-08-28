import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from itertools import combinations
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="C4_日中デイトレ研究（気象データ版）", page_icon="🌦️", layout="wide")
JST = ZoneInfo("Asia/Tokyo")

# =========================================================
# C4_app.py：C_app.py/C2_app.py/C3_app.pyと同じ手法（発見期間と検証期間の分離＋
# Bonferroni補正）で、市場の指標ではなく「東京の天気」という、普通は想定しない
# 変わった切り口を追加して日中デイトレ戦略（寄り付き買い→大引け売り）を検証する
# 追加調査アプリ。C_app.py・C2_app.py・C3_app.py・main_app.py・day_backtest.pyは
# 変更していません。
#
# 背景：学術研究には「Good Day Sunshine（好天効果）」として知られる、
# 取引所所在地の日照時間と株式リターンの間に弱い相関があるという報告がある
# （Hirshleifer & Shumway, 2003ほか）。効果は小さく議論もあるが、
# VIXやドル円のような「誰でも思いつく指標」とは違い、市場参加者に
# 織り込まれにくい変わった切り口として検証する価値がある。
#
# C3_app.pyとの違い：単一条件の候補を21種類→26種類に拡張。
# 追加した5指標（気象データはOpen-Meteo歴史データAPIから取得。株価データとは別ソース）：
#   - prev_sunshine_min ：前日（東京）の日照時間（分）
#   - prev_precip_mm     ：前日（東京）の降水量（mm）
#   - prev_temp_max      ：前日（東京）の最高気温（℃）
#   - morning_precip_mm  ：当日朝6-9時（東京・寄り付き前）の降水量（mm）
#   - morning_temp       ：当日朝6-9時（東京・寄り付き前）の平均気温（℃）
# いずれも「その日の寄り付き前」に確定している情報のみを使用。
# 天気データが取得できない場合、該当指標は自動的にスキップされます。
# =========================================================

TICKERS = {
    "1570": "1570.T",
    "NASDAQ100": "NQ=F",
    "S&P500": "^GSPC",
    "SOX": "^SOX",
    "NYダウ": "^DJI",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "ドル円": "USDJPY=X",
    "米10年金利": "^TNX",
    "原油": "CL=F",
    "金": "GC=F",
    "米国債ETF": "TLT",
}

TOKYO_LAT = 35.6762
TOKYO_LON = 139.6503

FEATURES = {
    "gap_pct": ("当日1570寄り付き窓開け率", "1570"),
    "nq_ret": ("前回米国NASDAQ100騰落率", "NASDAQ100"),
    "sp_ret": ("前回米国S&P500騰落率", "S&P500"),
    "sox_ret": ("前回米国SOX騰落率", "SOX"),
    "dow_ret": ("前回米国NYダウ騰落率", "NYダウ"),
    "vix_level": ("前回米国VIX水準", "VIX"),
    "usd_ret": ("前回ドル円騰落率", "ドル円"),
    "tnx_ret": ("前回米10年金利変化率", "米10年金利"),
    "us_breadth": ("米国4指数プラス数", "米国4指数"),
    "nq_usd_combo": ("NASDAQ100騰落率＋ドル円騰落率", "NASDAQ×ドル円"),
    "oil_ret": ("前回WTI原油騰落率", "原油"),
    "gold_ret": ("前回金騰落率", "金"),
    "vix_ret": ("前回VIX騰落率（水準ではなく変化率）", "VIX"),
    "bond_ret": ("前回米国債ETF(TLT)騰落率", "米国債ETF"),
    "weekday": ("当日の曜日（月=0～金=4）", "カレンダー"),
    "us_range_pct": ("前回米国NASDAQ100の値幅（高値-安値/終値）", "NASDAQ100"),
    "vix_term": ("VIX期間構造（VIX3M÷VIX）", "VIX期間構造"),
    "streak_us": ("米国NASDAQ100の連続陽線・陰線日数（符号付き）", "NASDAQ100"),
    "realized_vol_1570": ("1570自身の直近10日ボラティリティ", "1570"),
    "prior_day_ret": ("1570自身の前日の日中リターン", "1570"),
    "vix_shock": ("VIXの変化率の絶対値（急変動の大きさ）", "VIX"),
    "prev_sunshine_min": ("前日（東京）の日照時間（分）", "天気"),
    "prev_precip_mm": ("前日（東京）の降水量（mm）", "天気"),
    "prev_temp_max": ("前日（東京）の最高気温（℃）", "天気"),
    "morning_precip_mm": ("当日朝6-9時（東京）の降水量（mm）", "天気"),
    "morning_temp": ("当日朝6-9時（東京）の平均気温（℃）", "天気"),
}

EXPLANATIONS = {
    "gap_pct": "当日、1570が前日終値に対してどれだけ窓を開けて始まったかです。main_app.pyでは先物（NIY=F）から寄り付き前に推定しますが、ここではバックテストのため実際の始値を使う近似をしています（day_backtest.pyと同じ考え方）。",
    "nq_ret": "直前に終了した米国NASDAQ100の騰落です。寄り付き前に確定している情報です。",
    "sp_ret": "直前のS&P500です。米国株全体のリスクオン・リスクオフを表します。",
    "sox_ret": "米国半導体株の動きです。",
    "dow_ret": "main_app.pyが使うNYダウの直前騰落そのものです。",
    "vix_level": "main_app.pyが使うVIX水準そのものです。",
    "usd_ret": "main_app.pyが使うドル円の直前騰落そのものです。",
    "tnx_ret": "米10年金利の直前変化率です。",
    "us_breadth": "NASDAQ100・S&P500・SOX・NYダウのうち何指数が上昇したかです。",
    "nq_usd_combo": "NASDAQ100とドル円を足した簡易指標です。",
    "oil_ret": "直前のWTI原油先物の騰落です。資源・インフレ懸念や世界景気に対するリスク心理を表します。",
    "gold_ret": "直前の金先物の騰落です。株式市場のリスク回避度合いを表す代表的な指標です。",
    "vix_ret": "VIXの「水準」ではなく「前々日からの変化率」です。急上昇・急低下そのものが投資家心理の変化を表すことがあります。",
    "bond_ret": "米国長期国債ETF(TLT)の直前騰落です。債券が買われている（金利低下）かどうかで、リスク回避姿勢を確認します。",
    "weekday": "当日が何曜日かです。曜日そのものは寄り付き前から分かっている情報なので、当日の値として使えます。",
    "us_range_pct": "直前の米国NASDAQ100の「値幅（高値と安値の差）」です。方向ではなく、その日どれだけ荒れたか＝ボラティリティの大きさを表します。",
    "vix_term": "VIX（1ヶ月先の予想変動率）とVIX3M（3ヶ月先）の比率です。通常は3ヶ月先の方が高い（コンタンゴ）ですが、市場が不安定になると逆転（バックワーデーション）します。",
    "streak_us": "米国NASDAQ100が何日連続で上昇（または下落）しているかです。プラスなら連続陽線の日数、マイナスなら連続陰線の日数を表し、トレンドの続きやすさ・反転しやすさを見ます。",
    "realized_vol_1570": "1570自身の直近10日間の値動きの大きさ（標準偏差）です。当日の値は含めず、過去の確定した値動きだけから計算しています。",
    "prior_day_ret": "1570自身が前日の日中デイトレ（寄り付き→大引け）でどれだけ動いたかです。前日の勢いが続くか、反転しやすいかを見ます。",
    "vix_shock": "VIXが前々日からどれだけ急に変化したかを、方向を問わず（絶対値で）見た指標です。急上昇・急低下のどちらも「ショック」として扱います。",
    "prev_sunshine_min": "前日、東京でどれだけ日が照っていたか（分）です。「晴れた日は株が上がりやすい」という学術研究（Good Day Sunshine効果）にちなんだ指標です。",
    "prev_precip_mm": "前日、東京でどれだけ雨・雪が降ったか（mm）です。悪天候が投資家心理に与える影響を見ます。",
    "prev_temp_max": "前日の東京の最高気温です。暑さ・寒さが取引意欲に影響するかを見ます。",
    "morning_precip_mm": "当日の朝6時〜9時（寄り付き前）に東京でどれだけ雨が降ったかです。通勤時の悪天候が当日の投資家心理に与える影響を見ます。",
    "morning_temp": "当日の朝6時〜9時（寄り付き前）の東京の平均気温です。",
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

@st.cache_data(ttl=3600)
def download_weather_daily(start, end):
    """東京の日次気象データ（Open-Meteo歴史データAPI）。取得失敗時は空のDataFrameを返す。"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": TOKYO_LAT,
            "longitude": TOKYO_LON,
            "start_date": start,
            "end_date": end,
            "daily": "sunshine_duration,precipitation_sum,temperature_2m_max",
            "timezone": "Asia/Tokyo",
        }
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        d = js.get("daily", {})
        if "time" not in d:
            return pd.DataFrame()
        df = pd.DataFrame(d)
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
        df = df.set_index("date")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def download_weather_hourly(start, end):
    """東京の時間別気象データ（Open-Meteo歴史データAPI）。取得失敗時は空のDataFrameを返す。"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": TOKYO_LAT,
            "longitude": TOKYO_LON,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,precipitation",
            "timezone": "Asia/Tokyo",
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        h = js.get("hourly", {})
        if "time" not in h:
            return pd.DataFrame()
        df = pd.DataFrame(h)
        df["datetime"] = pd.to_datetime(df["time"])
        df["date"] = df["datetime"].dt.normalize()
        df["hour"] = df["datetime"].dt.hour
        return df
    except Exception:
        return pd.DataFrame()

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

def compute_streak(sign_series):
    """符号付き連続日数：+1が3日連続なら+3、-1が2日連続なら-2。符号が変わったらリセット。"""
    out = []
    cur = 0.0
    prev = 0.0
    for s in sign_series.values:
        if pd.isna(s) or s == 0:
            cur = 0.0
            prev = 0.0
        elif s == prev:
            cur += s
        else:
            cur = s
            prev = s
        out.append(cur)
    return pd.Series(out, index=sign_series.index)

@st.cache_data(ttl=3600)
def build_research(start, end):
    start_dt = pd.Timestamp(start) - pd.Timedelta(days=30)
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

    # 当日の窓開け率：前日終値→当日始値（寄り付き執行の近似。実運用は先物から推定）
    prev_close = jp["etf_close"].shift(1)
    jp["gap_pct"] = (jp["etf_open"] - prev_close) / prev_close * 100

    # 当日の曜日（月=0～金=4）。寄り付き前から分かっている当日自身の情報。
    jp["weekday"] = jp.index.dayofweek

    # 米国市場情報：日本日付Dより前に終了した直近米国セッション（寄り付き前に確定済み）
    us_returns = {
        "nq_ret": close_s(raw["NASDAQ100"]).pct_change() * 100,
        "sp_ret": close_s(raw["S&P500"]).pct_change() * 100,
        "sox_ret": close_s(raw["SOX"]).pct_change() * 100,
        "dow_ret": close_s(raw["NYダウ"]).pct_change() * 100,
        "usd_ret": close_s(raw["ドル円"]).pct_change() * 100,
        "tnx_ret": close_s(raw["米10年金利"]).pct_change() * 100,
        "oil_ret": close_s(raw["原油"]).pct_change() * 100,
        "gold_ret": close_s(raw["金"]).pct_change() * 100,
        "vix_ret": close_s(raw["VIX"]).pct_change() * 100,
        "bond_ret": close_s(raw["米国債ETF"]).pct_change() * 100,
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

    # --- 手の込んだ指標（C3から） ---
    nq_df = raw["NASDAQ100"]
    if not nq_df.empty and "High" in nq_df.columns and "Low" in nq_df.columns:
        nq_range = (pd.to_numeric(nq_df["High"], errors="coerce") - pd.to_numeric(nq_df["Low"], errors="coerce")) \
                   / pd.to_numeric(nq_df["Close"], errors="coerce") * 100
        nq_range = nq_range.dropna()
        jp["us_range_pct"] = map_prior_us_feature(jp.index, nq_range, "us_range_pct").reindex(jp.index).values
    else:
        jp["us_range_pct"] = np.nan

    vix3m = close_s(raw["VIX3M"])
    if not vix3m.empty:
        vix3m_mapped = map_prior_us_feature(jp.index, vix3m, "vix3m_level").reindex(jp.index).values
        with np.errstate(divide="ignore", invalid="ignore"):
            jp["vix_term"] = vix3m_mapped / jp["vix_level"].values
    else:
        jp["vix_term"] = np.nan

    nq_close_full = close_s(raw["NASDAQ100"])
    if not nq_close_full.empty:
        nq_sign = np.sign(nq_close_full.diff())
        nq_streak = compute_streak(nq_sign)
        jp["streak_us"] = map_prior_us_feature(jp.index, nq_streak, "streak_us").reindex(jp.index).values
    else:
        jp["streak_us"] = np.nan

    etf_ret = jp["etf_close"].pct_change() * 100
    jp["realized_vol_1570"] = etf_ret.rolling(10).std().shift(1)

    # 目的変数：当日寄り付き買い → 当日大引け売り（日中デイトレ）
    jp["day_ret"] = (jp["etf_close"] - jp["etf_open"]) / jp["etf_open"] * 100

    jp["prior_day_ret"] = jp["day_ret"].shift(1)
    jp["vix_shock"] = jp["vix_ret"].abs()

    # --- ここから「天気」指標（Open-Meteo歴史データAPI、株価データとは別ソース） ---
    weather_daily = download_weather_daily(start_dt.date().isoformat(), end_dt.date().isoformat())
    if not weather_daily.empty and "sunshine_duration" in weather_daily.columns:
        sunshine_min = pd.to_numeric(weather_daily["sunshine_duration"], errors="coerce") / 60.0
        jp["prev_sunshine_min"] = map_prior_us_feature(jp.index, sunshine_min.dropna(), "prev_sunshine_min").reindex(jp.index).values
    else:
        jp["prev_sunshine_min"] = np.nan
    if not weather_daily.empty and "precipitation_sum" in weather_daily.columns:
        precip = pd.to_numeric(weather_daily["precipitation_sum"], errors="coerce")
        jp["prev_precip_mm"] = map_prior_us_feature(jp.index, precip.dropna(), "prev_precip_mm").reindex(jp.index).values
    else:
        jp["prev_precip_mm"] = np.nan
    if not weather_daily.empty and "temperature_2m_max" in weather_daily.columns:
        tmax = pd.to_numeric(weather_daily["temperature_2m_max"], errors="coerce")
        jp["prev_temp_max"] = map_prior_us_feature(jp.index, tmax.dropna(), "prev_temp_max").reindex(jp.index).values
    else:
        jp["prev_temp_max"] = np.nan

    weather_hourly = download_weather_hourly(start_dt.date().isoformat(), end_dt.date().isoformat())
    if not weather_hourly.empty and "hour" in weather_hourly.columns:
        morning = weather_hourly[(weather_hourly["hour"] >= 6) & (weather_hourly["hour"] < 9)]
        if "precipitation" in morning.columns and not morning.empty:
            morning_precip = pd.to_numeric(morning["precipitation"], errors="coerce")
            morning_precip_by_date = morning.assign(_p=morning_precip).groupby("date")["_p"].sum()
            jp["morning_precip_mm"] = morning_precip_by_date.reindex(jp.index).values
        else:
            jp["morning_precip_mm"] = np.nan
        if "temperature_2m" in morning.columns and not morning.empty:
            morning_temp = pd.to_numeric(morning["temperature_2m"], errors="coerce")
            morning_temp_by_date = morning.assign(_t=morning_temp).groupby("date")["_t"].mean()
            jp["morning_temp"] = morning_temp_by_date.reindex(jp.index).values
        else:
            jp["morning_temp"] = np.nan
    else:
        jp["morning_precip_mm"] = np.nan
        jp["morning_temp"] = np.nan

    jp = jp[(jp.index >= pd.Timestamp(start)) & (jp.index <= pd.Timestamp(end))]
    needed = ["etf_open", "etf_close", "gap_pct", "day_ret", "vix_level", "usd_ret", "dow_ret"]
    return jp.dropna(subset=needed)

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

def discover_candidates(train, valid, target_col, min_n, use_correction=True):
    base_tr = stats(train[target_col])
    base_va = stats(valid[target_col])
    rows = []

    total_tests = len(FEATURES) * len(QUANTILES)
    z_crit = bonferroni_z(total_tests) if use_correction else 1.0
    valid_min_n = max(30, min_n // 2)

    for feat, (label, _) in FEATURES.items():
        if feat not in train.columns:
            continue
        s = train[feat].dropna()
        if len(s) < min_n:
            continue

        for q, direction in QUANTILES:
            threshold = s.quantile(q)
            mtr = candidate_mask(train, feat, direction, threshold)
            mva = candidate_mask(valid, feat, direction, threshold)

            tr = stats(train.loc[mtr, target_col])
            va = stats(valid.loc[mva, target_col])

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

def pick_combo_pool(single_result, max_pool=10):
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
    pool = pool.sort_values("再現スコア", ascending=False).drop_duplicates(subset="feature", keep="first")
    pool = pool.head(max_pool)
    return pool, source

def discover_combo_candidates(train, valid, target_col, pool, base_tr, base_va, min_n, use_correction=True):
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

        tr = stats(train.loc[mtr, target_col])
        va = stats(valid.loc[mva, target_col])

        if tr["n"] < min_n:
            continue

        train_edge = tr["mean"] - base_tr["mean"]
        valid_edge = va["mean"] - base_va["mean"] if va["n"] else np.nan

        t_valid = welch_t(va["mean"], va["std"], va["n"], base_va["mean"], base_va["std"], base_va["n"])
        significant = (not np.isnan(t_valid)) and abs(t_valid) >= z_crit
        enough_n = va["n"] >= valid_min_n

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

def plain_explanation(row, base_tr, base_va):
    feat = row["feature"]
    direction_text = "以下" if row["direction"] == "low" else "以上"
    label = row["情報"]
    t = row["threshold"]
    return (
        f"**{label} が {t:.3f}{direction_text}**だった日を取り出した条件です。  \n"
        f"発見期間では平均 {row['発見平均']:+.3f}%（無条件 {base_tr['mean']:+.3f}%）、"
        f"検証期間では平均 {row['検証平均']:+.3f}%（無条件 {base_va['mean']:+.3f}%）でした。  \n"
        f"{EXPLANATIONS.get(feat,'')}"
    )

# =========================================================
# UI
# =========================================================
st.title("🌦️ C4_日中デイトレ研究（気象データ版）")
st.caption("C_app.py/C2_app.py/C3_app.pyと同じ発見期間・検証期間の分離＋多重検定補正で、東京の天気という変わった切り口を追加して検証します。")

st.info(
    "C_app.py・C2_app.py・C3_app.py・main_app.py・day_backtest.pyは変更していません。これは追加調査専用の新アプリです。\n\n"
    "対象戦略：当日の1570を**寄り付きで買い、大引けで売る**（日中デイトレ）。\n"
    "判断材料は寄り付き前に確定している情報のみ使用（当日の日経225・TOPIXの終値は使いません）。\n\n"
    "「晴れた日は株が上がりやすい」という学術研究（Good Day Sunshine効果）にちなみ、"
    "東京の天気データ（Open-Meteo歴史データAPI、株価データとは別ソース）を追加しました。"
    "取得に失敗した場合、天気の指標は自動的にスキップされます。"
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

if run or "C4_df" not in st.session_state:
    with st.spinner("研究データを取得・整理しています（株価データ＋気象データ）…"):
        try:
            st.session_state["C4_df"] = build_research(all_start.isoformat(), all_end.isoformat())
        except Exception as e:
            st.error(f"データ取得エラー：{e}")
            st.stop()

df = st.session_state["C4_df"]
if df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

train = df[(df.index >= pd.Timestamp(discover_start)) & (df.index <= pd.Timestamp(discover_end))].copy()
valid = df[(df.index >= pd.Timestamp(validate_start)) & (df.index <= pd.Timestamp(validate_end))].copy()

TARGET = "day_ret"
base_tr = stats(train[TARGET])
base_va = stats(valid[TARGET])

st.header("① まず、何を比べているの？")
st.write("毎日無条件で1570を寄り付きに買って大引けで売った場合を基準にします。天気を含む追加指標が、この基準よりも良い日だけを選び出せているかを検証します。")

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

st.subheader("🌦️ 天気データの取得状況")
weather_cols = ["prev_sunshine_min", "prev_precip_mm", "prev_temp_max", "morning_precip_mm", "morning_temp"]
wc = st.columns(len(weather_cols))
for i, col in enumerate(weather_cols):
    non_na = df[col].notna().mean() * 100 if col in df.columns else 0.0
    wc[i].metric(FEATURES[col][0], f"{non_na:.0f}% 取得")

# =========================================================
# ② 単一条件の発見・検証（天気を含む26特徴量）
# =========================================================
st.header("② 単一条件の発見・検証（天気を含む26特徴量）")
st.caption("C3_app.pyの21指標に加え、前日の東京の日照時間・降水量・最高気温、当日朝6-9時の降水量・気温の5つの天気指標を追加しています。")

single_result, base_tr2, base_va2, single_tests, single_z = discover_candidates(
    train, valid, TARGET, int(min_n), use_correction=use_correction
)

if single_result.empty:
    st.warning("単一条件の候補が見つかりませんでした。期間や最低サンプル数を見直してください。")
else:
    n_single_green = int((single_result["判定"] == "🟢 再現候補").sum())
    st.write(f"単一条件の🟢再現候補：**{n_single_green}件**（{single_tests}条件中、Bonferroni補正 z ≥ {single_z:.2f}）")

    with st.expander("単一条件の結果を全て見る"):
        show1 = single_result.drop(columns=["feature","direction","threshold","再現スコア"])
        st.dataframe(show1, use_container_width=True, hide_index=True)

    reproducible = single_result[single_result["判定"] == "🟢 再現候補"]
    if not reproducible.empty:
        top = reproducible.iloc[0]
        st.success(f"⭐ 最も注目する単一条件：**{top['情報']} / {top['条件']}**")
        st.markdown(plain_explanation(top, base_tr, base_va))

# =========================================================
# ③ 2条件組み合わせ（AND）の発見・検証
# =========================================================
st.header("③ 2条件組み合わせ（AND）の発見・検証")

if not single_result.empty:
    pool, pool_source = pick_combo_pool(single_result, max_pool=int(max_pool))

    if len(pool.drop_duplicates(subset="feature")) < 2:
        st.warning("組み合わせを作るための単一条件候補が2件未満でした。")
    else:
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
            train, valid, TARGET, pool, base_tr, base_va, int(min_n), use_correction=use_correction
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
                st.warning("2条件を組み合わせても、統計的に有意な再現候補は見つかりませんでした。")
            else:
                top = green_combo.iloc[0]
                st.success(
                    f"⭐ 最も注目する2条件組み合わせ：**{top['組み合わせ']}**  \n"
                    f"検証 {int(top['検証件数'])}件・平均 {top['検証平均']:+.3f}%（無条件比 {top['検証改善']:+.3f}%）・"
                    f"検証t値 {top['検証t値']:.2f}"
                )
                show2 = green_combo.drop(columns=["特徴量1","特徴量2","再現スコア"])
                st.dataframe(show2, use_container_width=True, hide_index=True)

            st.subheader("全組み合わせ結果（上位40件）")
            show_all = combo_result.drop(columns=["特徴量1","特徴量2","再現スコア"]).head(40).copy()
            st.dataframe(show_all, use_container_width=True, hide_index=True)

st.header("④ 表の読み方")
st.markdown("""
**②③** はB02/B03/C_app.py/C2_app.py/C3_app.pyと同じく、多数の条件を同時に試すため
Bonferroni補正した厳しい基準で「🟢再現候補」を判定しています。

**「改善」** はその期間の無条件平均と比べて、条件を付けることで平均リターンが何%上がったかです。
""")

st.warning(
    "⚠️ 重要：この結果だけでmain_app.pyの判定ロジックを変更しません。"
    "『発見 → 未使用期間で検証 → 実運用でフォワード確認』の順で進めます。"
    "また、backtestは実際の約定条件（手数料・スリッページ・寄り付きの実際の約定価格）を完全には再現しません。"
)
st.caption(f"最終更新：{datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')} JST")
