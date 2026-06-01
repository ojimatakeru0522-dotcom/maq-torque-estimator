#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MA-Q csv + 身長 + 体重だけで肘関節内反ピークトルクを推定するStreamlitアプリ

対応形式:
- 1つのCSV内に、同一計測日の複数投球が時系列で並んでいる形式
- 「== yy0 ==============」は静止データとして除外
- 「== yy1 ==============」など、yy0以外のブロックを各投球データとして推定

重要:
CB.csv は毎回アップロードしない。
この app.py と同じ階層に data フォルダを作り、
その中に固定用の CB.csv を置く。

構成:
app.py
requirements.txt
data/
└── CB.csv
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 固定CBファイル
# =========================================================
CB_FIXED_PATH = os.path.join("data", "CB.csv")


# =========================================================
# 1. 基本関数
# =========================================================
def acc_to_velo(ax, ay, az, fps: float):
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)

    dt = 1.0 / fps
    n = len(ax)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vz = np.zeros(n)

    for i in range(1, n):
        vx[i] = vx[i - 1] + 0.5 * (ax[i - 1] + ax[i]) * dt
        vy[i] = vy[i - 1] + 0.5 * (ay[i - 1] + ay[i]) * dt
        vz[i] = vz[i - 1] + 0.5 * (az[i - 1] + az[i]) * dt

    return np.vstack((vx, vy, vz)).T


def add_global_coords(
    df: pd.DataFrame,
    fps: float = 500.0,
    acc_cols=("hiacc_x", "hiacc_y", "hiacc_z"),
    gyro_cols=("gyro_x", "gyro_y", "gyro_z"),
    cb_csv_path: Optional[str] = None,
    eps: float = 1e-9
) -> pd.DataFrame:
    for c in (*acc_cols, *gyro_cols):
        if c not in df.columns:
            raise ValueError(f"Column missing for global transform: {c}")

    if len(df) < 131:
        raise ValueError(f"MA-Q csvが短すぎます。少なくとも131行必要ですが、{len(df)}行です。")

    if cb_csv_path is None:
        raise ValueError("CB.csv の固定パスが指定されていません。")

    if not os.path.exists(cb_csv_path):
        raise FileNotFoundError(
            f"固定CB.csvが見つかりません: {cb_csv_path}\n"
            "app.py と同じ階層に data フォルダを作り、その中に CB.csv を置いてください。"
        )

    # X軸: 擬似速度の平均方向
    maq_velo = acc_to_velo(
        df[acc_cols[0]].to_numpy(),
        df[acc_cols[1]].to_numpy(),
        df[acc_cols[2]].to_numpy(),
        fps=fps
    )

    maq_velo_mean = maq_velo[110:131, :].mean(axis=0)

    if np.linalg.norm(maq_velo_mean) < eps:
        raise ValueError("進行方向ベクトルがゼロに近いです。")

    X = maq_velo_mean / np.linalg.norm(maq_velo_mean)

    # 固定CB.csvから重力方向を推定
    df_cb = pd.read_csv(cb_csv_path, encoding="cp932")

    if df_cb.shape[0] < 103 or df_cb.shape[1] < 26:
        raise ValueError(
            f"CB.csvのサイズが不足しています。必要: 103行以上・26列以上, "
            f"実際: {df_cb.shape[0]}行・{df_cb.shape[1]}列"
        )

    acc_data = df_cb.iloc[3:103, [23, 24, 25]].apply(pd.to_numeric, errors="coerce").to_numpy()

    norms = np.linalg.norm(acc_data, axis=1)
    norms[norms == 0] = np.nan

    unit_a = acc_data / norms[:, np.newaxis]
    gravity_vec = np.nanmean(unit_a, axis=0)

    if np.any(~np.isfinite(gravity_vec)) or np.linalg.norm(gravity_vec) < eps:
        raise ValueError("CB.csvから重力方向を推定できませんでした。")

    gravity_vec = gravity_vec / np.linalg.norm(gravity_vec)
    G = -gravity_vec

    # Y軸 = X × G
    Y = np.cross(X, G)

    if np.linalg.norm(Y) < eps:
        Y = np.cross(X, np.array([0, 1, 0]))

    if np.linalg.norm(Y) < eps:
        raise ValueError("Y軸を定義できませんでした。")

    Y = Y / np.linalg.norm(Y)

    # Z軸 = X × Y
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)

    maq_global = np.stack([X, Y, Z], axis=-1)

    acc_local = df[list(acc_cols)].to_numpy(dtype=float)
    gyro_local = df[list(gyro_cols)].to_numpy(dtype=float)

    acc_global = acc_local @ maq_global
    gyro_global = gyro_local @ maq_global

    # 元コードに合わせてジャイロZを反転
    gyro_global[:, 2] *= -1

    acc_g_df = pd.DataFrame(acc_global, columns=[f"{c}_g" for c in acc_cols])
    gyro_g_df = pd.DataFrame(gyro_global, columns=[f"{c}_g" for c in gyro_cols])

    return pd.concat([df.reset_index(drop=True), acc_g_df, gyro_g_df], axis=1)


def fix_gyro_sign_flips(
    df: pd.DataFrame,
    cols: Iterable[str] = ("gyro_x", "gyro_y", "gyro_z"),
    threshold: float = 2000.0,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    corrected = df.copy()
    flip_indices: Dict[str, List[int]] = {}

    for c in cols:
        s = pd.to_numeric(corrected[c], errors="coerce")
        prev = s.shift(1)

        big_jump = (s - prev).abs() >= threshold
        sign_prev = np.sign(prev)
        sign_curr = np.sign(s)
        sign_flipped = (sign_prev * sign_curr == -1)

        trigger = (big_jump & sign_flipped).fillna(False)
        flip_indices[c] = list(s.index[trigger])

        corrected_series = s.copy()
        for idx in flip_indices[c]:
            corrected_series.loc[idx:] *= -1.0

        corrected[c] = corrected_series

    return corrected, flip_indices


def detect_release_timing(
    df: pd.DataFrame,
    cols: Iterable[str] = ("hiacc_x_g", "hiacc_y_g", "hiacc_z_g"),
    base_start: int = 100,
    base_end: int = 130,
    multiplier: float = 2.0,
) -> Tuple[Optional[int], Optional[float]]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    if base_end > len(df):
        raise ValueError(f"base_end ({base_end}) exceeds dataframe length ({len(df)})")

    norm = np.sqrt(np.sum(df.loc[:, cols].astype(float) ** 2, axis=1))
    base_mean = norm.iloc[base_start:base_end].mean()

    for i in range(base_start - 1, -1, -1):
        if norm.iloc[i] > base_mean * multiplier:
            return i, float(norm.iloc[i])

    return None, None


def count_frames_after_release(df: pd.DataFrame, release_idx: Optional[int]) -> Optional[int]:
    if release_idx is None:
        return None
    return len(df) - release_idx - 1


def drop_after_release(df: pd.DataFrame, release_idx: Optional[int]) -> pd.DataFrame:
    if release_idx is None:
        return df.copy()
    return df.iloc[:release_idx + 1].copy()


def split_custom(df: pd.DataFrame, parts):
    n = len(df)
    splits = []
    start = 0

    for size in parts:
        if size <= 0:
            continue

        end = min(start + size, n)
        splits.append(df.iloc[start:end])
        start = end

        if start >= n:
            break

    if start < n:
        splits.append(df.iloc[start:])

    return splits


# =========================================================
# 2. 特徴量作成・トルク推定
# =========================================================
def make_features_from_last82(last82: pd.DataFrame, mass: float, height: float, abc: int = 52) -> Dict[str, float]:
    splits = split_custom(last82, [abc, 82 - abc])

    feat: Dict[str, float] = {
        "mass": float(mass),
        "height": float(height),
    }

    for i, sub_df in enumerate(splits, start=1):
        axis_cols = [
            c for c in sub_df.columns
            if (
                ("hiacc_" in c and "_g" in c)
                or ("gyro_" in c)
            )
        ]

        for col in axis_cols:
            col_data = pd.to_numeric(sub_df[col], errors="coerce").dropna()
            feat[f"{col}_mean_part{i}"] = float(col_data.mean())
            feat[f"{col}_range_part{i}"] = float(col_data.max() - col_data.min())

    return feat


def calc_peak_torque_from_features(feat: Dict[str, float]) -> float:
    required = [
        "height",
        "hiacc_x_g_mean_part1",
        "mass",
        "hiacc_x_g_mean_part2",
        "hiacc_z_g_range_part1",
        "hiacc_y_g_range_part1",
        "gyro_x_range_part1",
        "gyro_z_range_part1",
        "gyro_y_range_part1",
        "gyro_x_range_part2",
        "gyro_z_range_part2",
        "gyro_y_range_part2",
    ]

    missing = [k for k in required if k not in feat or not np.isfinite(feat[k])]
    if missing:
        raise ValueError(f"回帰式に必要な特徴量が作成できていません: {missing}")

    peak_torque = (
        116.750927
        - 85.207859 * feat["height"]
        - 1.198912 * feat["hiacc_x_g_mean_part1"]
        + 0.888097 * feat["mass"]
        + 0.462018 * feat["hiacc_x_g_mean_part2"]
        - 0.216850 * feat["hiacc_z_g_range_part1"]
        - 0.179015 * feat["hiacc_y_g_range_part1"]
        + 0.005488 * feat["gyro_x_range_part1"]
        + 0.003471 * feat["gyro_z_range_part1"]
        + 0.003259 * feat["gyro_y_range_part1"]
        + 0.001968 * feat["gyro_x_range_part2"]
        - 0.000953 * feat["gyro_z_range_part2"]
        + 0.000228 * feat["gyro_y_range_part2"]
    )

    return float(peak_torque)


# =========================================================
# 3. 複数投球CSVの読み込み・推定
# =========================================================
def _decode_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()

    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="ignore")


def load_trials_from_day_csv(uploaded_file) -> Dict[int, pd.DataFrame]:
    """
    同一計測日のCSVから投球データだけを抽出する。

    想定:
    == yy0 ==============  は静止データ
    == yy1 ==============  など yy0以外は投球データ
    yy0 が再度出たら、前の投球を確定して次の試技へ進む
    """
    text = _decode_uploaded_file(uploaded_file)
    lines = text.splitlines()

    header = None
    current_rows: List[List[str]] = []
    trials_rows: List[List[List[str]]] = []
    in_throw = False

    for raw_line in lines:
        line = raw_line.rstrip("\n\r")

        if not line.strip():
            continue

        if line.startswith("投球年"):
            header = line.split("\t")
            continue

        if line.startswith("== yy0"):
            if current_rows:
                trials_rows.append(current_rows)
                current_rows = []
            in_throw = False
            continue

        if line.startswith("== yy"):
            # yy0以外のブロックは投球データとして扱う
            in_throw = True
            continue

        if not in_throw:
            continue

        cols = line.split("\t")

        # 数値データ行以外を除外
        if len(cols) < 6:
            continue

        try:
            int(float(cols[0]))
        except ValueError:
            continue

        current_rows.append(cols)

    if current_rows:
        trials_rows.append(current_rows)

    if header is None:
        raise ValueError("CSVヘッダー行（投球年...）が見つかりませんでした。")

    required_cols = [
        "hi_ax[g]",
        "hi_ay[g]",
        "hi_az[g]",
        "gyro_x[dps]",
        "gyro_y[dps]",
        "gyro_z[dps]",
    ]

    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(f"必要な列が見つかりません: {missing}")

    idx = {c: header.index(c) for c in required_cols}
    max_idx = max(idx.values())

    trial_dfs: Dict[int, pd.DataFrame] = {}

    for trial_no, rows in enumerate(trials_rows, start=1):
        extracted = []

        for row in rows:
            if len(row) <= max_idx:
                continue

            extracted.append({
                "hiacc_x": row[idx["hi_ax[g]"]],
                "hiacc_y": row[idx["hi_ay[g]"]],
                "hiacc_z": row[idx["hi_az[g]"]],
                "gyro_x": row[idx["gyro_x[dps]"]],
                "gyro_y": row[idx["gyro_y[dps]"]],
                "gyro_z": row[idx["gyro_z[dps]"]],
            })

        trial_df = pd.DataFrame(extracted)
        if trial_df.empty:
            continue

        trial_df = trial_df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

        # 既存アルゴリズムで最低限必要な長さ
        if len(trial_df) >= 131:
            trial_dfs[trial_no] = trial_df

    return trial_dfs


def estimate_torque_from_trial_df(
    trial_df: pd.DataFrame,
    mass: float,
    height: float,
    cb_csv_path: str = CB_FIXED_PATH,
    abc: int = 52,
    fps: float = 500.0,
) -> Dict[str, object]:
    base_columns = ["hiacc_x", "hiacc_y", "hiacc_z", "gyro_x", "gyro_y", "gyro_z"]
    all_columns = base_columns + ["hiacc_x_g", "hiacc_y_g", "hiacc_z_g"]

    missing = [c for c in base_columns if c not in trial_df.columns]
    if missing:
        raise ValueError(f"投球データに必要な列がありません: {missing}")

    df = trial_df[base_columns].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    if len(df) < 131:
        raise ValueError(f"投球データが短すぎます。少なくとも131行必要ですが、{len(df)}行です。")

    df = add_global_coords(df, fps=fps, cb_csv_path=cb_csv_path)

    corrected_df, flip_points = fix_gyro_sign_flips(df, threshold=2000.0)

    release_idx, release_val = detect_release_timing(corrected_df)

    if release_idx is None:
        filtered_df = corrected_df.copy()
        frames_after_release = None
    else:
        frames_after_release = count_frames_after_release(corrected_df, release_idx)
        filtered_df = drop_after_release(corrected_df, release_idx)

    unified_df = filtered_df[all_columns].copy()
    last82 = unified_df.tail(82).reset_index(drop=True)

    feat = make_features_from_last82(last82, mass=mass, height=height, abc=abc)
    peak_torque = calc_peak_torque_from_features(feat)

    return {
        "estimated_peak_torque": peak_torque,
        "release_idx": release_idx,
        "release_val": release_val,
        "frames_after_release": frames_after_release,
        "used_frames": len(last82),
        "flip_points": flip_points,
        "features": feat,
    }


def estimate_torques_from_day_csv(
    uploaded_file,
    mass: float,
    height: float,
    cb_csv_path: str = CB_FIXED_PATH,
    abc: int = 52,
    fps: float = 500.0,
) -> pd.DataFrame:
    trials = load_trials_from_day_csv(uploaded_file)

    if not trials:
        raise ValueError(
            "有効な投球データが見つかりませんでした。"
            " yy0以外の投球ブロックがあるか、各投球が131行以上あるか確認してください。"
        )

    results = []

    for trial_no, trial_df in trials.items():
        try:
            result = estimate_torque_from_trial_df(
                trial_df=trial_df,
                mass=mass,
                height=height,
                cb_csv_path=cb_csv_path,
                abc=abc,
                fps=fps,
            )

            results.append({
                "trial": trial_no,
                "torque": result["estimated_peak_torque"],
                "status": "ok",
                "error": "",
            })

        except Exception as e:
            results.append({
                "trial": trial_no,
                "torque": np.nan,
                "status": "error",
                "error": str(e),
            })

    return pd.DataFrame(results)


# =========================================================
# 4. Streamlit UI
# =========================================================
st.set_page_config(
    page_title="MA-Q Torque Estimator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Optional password auth
# =========================
if "APP_PASSWORD" in st.secrets:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 50% 20%, rgba(0,91,172,.22), transparent 30%),
                linear-gradient(135deg, #03111F 0%, #082B4C 55%, #061A2F 100%);
        }

        .block-container {
            max-width: 620px;
            padding-top: 12vh;
        }

        .login-card {
            max-width: 620px;
            margin: 0 auto;
            padding: 3rem;
            border-radius: 30px;
            background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(2,6,23,.96));
            border: 1px solid rgba(148,163,184,.25);
            box-shadow: 0 24px 70px rgba(0,0,0,.35);
            text-align: center;
        }

        .login-card h1 {
            color: #f8fafc;
            font-size: 3.8rem;
            font-weight: 850;
            line-height: 1.0;
            margin-bottom: 1rem;
        }

        .login-card p {
            color: #9db5d1;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.8rem;
        }

        .stTextInput {
            max-width: 500px;
            margin: 1.6rem auto 0 auto;
        }

        .stTextInput input {
            min-height: 58px;
            font-size: 1.35rem;
            font-weight: 650;
            text-align: center;
            border-radius: 16px;
        }

        .stTextInput input::placeholder {
            font-size: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="login-card">
            <h1>MA-Q Torque<br>Estimator</h1>
            <p>Password required</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    pwd = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
        label_visibility="collapsed"
    )

    if pwd:
        if pwd != st.secrets["APP_PASSWORD"]:
            st.error("Incorrect password")
            st.stop()
    else:
        st.stop()


# =========================
# Design CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0,174,239,0.16), transparent 28%),
            radial-gradient(circle at 85% 15%, rgba(0,91,172,0.18), transparent 30%),
            linear-gradient(135deg, #03111F 0%, #082B4C 48%, #061A2F 100%);
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0);
    }

    .block-container {
        padding-top: 2.3rem;
        max-width: 1120px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 30px;
        background: linear-gradient(135deg, rgba(3,17,31,.97), rgba(0,91,172,.34));
        border: 1px solid rgba(148,163,184,.22);
        box-shadow: 0 24px 80px rgba(0,0,0,.35);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
        text-align: left;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        right: -120px;
        top: -150px;
        background: radial-gradient(circle, rgba(0,91,172,.35), transparent 68%);
        border-radius: 50%;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .35rem .75rem;
        border-radius: 999px;
        background: rgba(0,91,172,.16);
        color: #dbeafe;
        border: 1px solid rgba(0,91,172,.42);
        font-size: .82rem;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin-bottom: .85rem;
    }

    .hero h1 {
        font-size: clamp(2.1rem, 5vw, 4.4rem);
        line-height: .95;
        margin: 0;
        letter-spacing: -0.055em;
        color: #f8fafc;
    }

    .hero .sub {
        margin-top: 1rem;
        max-width: 760px;
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    .panel {
        padding: 1.1rem 1.25rem 1.25rem 1.25rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, .82);
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 14px 45px rgba(0,0,0,.22);
    }

    .panel-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 2.6rem 2rem;
        border-radius: 28px;
        text-align: center;
        background: linear-gradient(145deg, #003B73, #005BAC);
        border: 1px solid rgba(111,195,255,.40);
        box-shadow:
            0 0 40px rgba(0,91,172,.45),
            0 20px 80px rgba(0,91,172,.35);
        margin-top: 1.8rem;
    }

    .result-label {
        color: #EAF4FF;
        text-transform: uppercase;
        letter-spacing: .06em;
        font-size: 1.45rem;
        font-weight: 850;
        margin-bottom: 1rem;
    }

    .result-value {
        font-size: clamp(3.2rem, 8vw, 6.4rem);
        font-weight: 900;
        letter-spacing: -0.06em;
        color: #ffffff;
        line-height: 1;
    }

    .result-unit {
        color: #dbeafe;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: .45rem;
    }

    .torque-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-top: 1.8rem;
    }

    .torque-card {
        padding: 1.35rem 1rem;
        border-radius: 22px;
        text-align: center;
        background: linear-gradient(145deg, #003B73, #005BAC);
        border: 1px solid rgba(111,195,255,.38);
        box-shadow: 0 16px 50px rgba(0,91,172,.24);
    }

    .torque-trial {
        color: #dbeafe;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: .65rem;
    }

    .torque-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.05em;
    }

    .torque-unit {
        color: #dbeafe;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: .3rem;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(0,91,172,.50);
        border-radius: 18px;
        padding: .35rem;
        background: rgba(2,6,23,.24);
    }

    .stButton > button {
        width: 100%;
        border-radius: 18px;
        padding: .9rem 1.1rem;
        font-weight: 850;
        letter-spacing: .02em;
        background: linear-gradient(90deg, #005BAC, #0078D7);
        border: none;
        color: white;
        box-shadow: 0 12px 30px rgba(0,91,172,.32);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 40px rgba(0,91,172,.28);
    }

    [data-testid="stWidgetLabel"] p {
        color: #D6ECFF !important;
        font-size: 1.08rem !important;
        font-weight: 800 !important;
    }

    .stNumberInput label,
    .stFileUploader label {
        color: #D6ECFF !important;
        font-weight: 800 !important;
    }

    .trial-result {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 0.7rem 0;
        letter-spacing: .03em;
    }

    @media (max-width: 760px) {
        .hero {
            padding: 1.5rem;
        }
        .result-label {
            font-size: 1.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">⚾ MA-Q SENSOR DATA ESTIMATOR</div>
        <h1>Elbow Varus Torque<br>Estimation System</h1>
        <p class="sub">
            MA-Qセンサの同一計測日CSVから複数の投球データを自動で抽出し、
            肘関節内反ピークトルクを推定します。
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


if not os.path.exists(CB_FIXED_PATH):
    st.error(
        f"固定CB.csvが見つかりません: `{CB_FIXED_PATH}`\n\n"
        "アプリフォルダ内に `data` フォルダを作成し、その中に `CB.csv` を置いてください。"
    )


_, center, _ = st.columns([1, 3, 1])

with center:
    st.markdown(
        """
        <div class="result-card">
            <div class="result-label">
                Estimated Peak Elbow Varus Torque
            </div>
        """,
        unsafe_allow_html=True
    )

    for _, row in result_df.iterrows():
        st.markdown(
            f"""
            <div class="trial-result">
                {int(row["trial"])}球目　{row["torque"]:.2f} N·m
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Result area
# =========================
if estimate_clicked:

    if maq_csv is None:
        st.error("MA-Q 計測日CSVをアップロードしてください。")
        st.stop()

    if mass is None or height is None:
        st.warning("身長と体重を入力してください。")
        st.stop()

    if not os.path.exists(CB_FIXED_PATH):
        st.error(f"固定CB.csvが見つかりません: {CB_FIXED_PATH}")
        st.stop()

    try:
        with st.spinner("Analyzing MA-Q data..."):
            result_df = estimate_torques_from_day_csv(
                uploaded_file=maq_csv,
                mass=mass,
                height=height,
                cb_csv_path=CB_FIXED_PATH,
                abc=52,
                fps=500.0,
            )

        ok_df = result_df[result_df["status"] == "ok"].copy()

        if ok_df.empty:
            st.error("推定できた投球がありませんでした。")
            st.stop()

        # 1球のみなら大きく表示
        if len(ok_df) == 1:
            torque = float(ok_df.iloc[0]["torque"])

            _, result_center, _ = st.columns([1, 4, 1])

            with result_center:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            Estimated Peak Elbow Varus Torque
                        </div>
                        <div class="result-value">
                            {torque:.2f}
                        </div>
                        <div class="result-unit">
                            N·m
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 複数球なら、各投球のトルクだけをカード表示
        else:
            cards_html = '<div class="torque-grid">'

            for _, row in ok_df.iterrows():
                cards_html += f"""
                <div class="torque-card">
                    <div class="torque-trial">{int(row["trial"])}球目</div>
                    <div class="torque-value">{float(row["torque"]):.2f}</div>
                    <div class="torque-unit">N·m</div>
                </div>
                """

            cards_html += "</div>"

            _, result_center, _ = st.columns([1, 5, 1])

            with result_center:
                st.markdown(cards_html, unsafe_allow_html=True)

    except Exception as e:
        st.error("推定中にエラーが発生しました。")
        st.exception(e)
