#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MA-Q csv + 身長 + 体重だけで肘内反ピークトルクを推定するStreamlitアプリ

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
import tempfile
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
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


def estimate_torque_from_uploaded_maq_csv(
    maq_csv_file,
    mass: float,
    height: float,
    cb_csv_path: str = CB_FIXED_PATH,
    abc: int = 52,
    fps: float = 500.0,
) -> Dict[str, object]:
    base_columns = ["hiacc_x", "hiacc_y", "hiacc_z", "gyro_x", "gyro_y", "gyro_z"]
    all_columns = base_columns + ["hiacc_x_g", "hiacc_y_g", "hiacc_z_g"]

    with tempfile.TemporaryDirectory() as tmpdir:
        maq_path = os.path.join(tmpdir, "maq.csv")

        with open(maq_path, "wb") as f:
            f.write(maq_csv_file.getbuffer())

        df = pd.read_csv(maq_path, header=None)

        if df.empty:
            raise ValueError("MA-Q csvが空です。")

        if df.shape[1] < 6:
            raise ValueError(f"MA-Q csvの列数が6列未満です: {df.shape[1]}列")

        df = df.iloc[:, -6:]
        df.columns = base_columns
        df = df.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

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



# =========================================================
# 3. Streamlit UI
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
# Streamlit Cloud の Secrets に APP_PASSWORD を設定すると有効化されます。
# 設定しない場合はパスワードなしで動きます。
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
            max-width: 520px;
            padding-top: 14vh;
        }

        .login-card {
            padding: 2.2rem;
            border-radius: 28px;
            background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(2,6,23,.96));
            border: 1px solid rgba(148,163,184,.25);
            box-shadow: 0 24px 70px rgba(0,0,0,.35);
            text-align: center;
        }

        .login-card h1 {
            color: #f8fafc;
            font-size: 2.3rem;
            line-height: 1.05;
            margin: 0 0 .8rem 0;
        }

        .login-card p {
            color: #9db5d1;
            margin-bottom: 1.5rem;
        }

        [data-testid="stTextInput"] {
            max-width: 360px;
            margin: 1rem auto 0 auto;
        }

        [data-testid="stTextInput"] input {
            border-radius: 14px;
            height: 48px;
            text-align: center;
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

    if pwd != st.secrets["APP_PASSWORD"]:
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
        padding-top: 2.5rem;
        max-width: 1120px;
    }

    .result-wrap {
        max-width: 960px;
        margin: 3.5rem auto 0 auto;
    }

    .hero {
        padding: 2.3rem 2.4rem;
        border-radius: 30px;
        background: linear-gradient(135deg, rgba(3,17,31,.97), rgba(0,91,172,.34));
        border: 1px solid rgba(148,163,184,.22);
        box-shadow: 0 24px 80px rgba(0,0,0,.35);
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
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
        font-size: 1.08rem;
        line-height: 1.75;
    }

    .panel {
        padding: 0.8rem 1.2rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, .82);
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 14px 45px rgba(0,0,0,.22);
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: .65rem;
    }

    .small-muted {
        color: #9db5d1;
        font-size: .92rem;
        line-height: 1.6;
    }

    .result-card {
        padding: 2.6rem 2rem;
        border-radius: 28px;
        text-align: center;
    
        background:
        linear-gradient(
            145deg,
            #003B73,
            #005BAC
        );
    
        border: 1px solid rgba(111,195,255,.40);
    
        box-shadow:
            0 0 40px rgba(0,91,172,.45),
            0 20px 80px rgba(0,91,172,.35);
    
        margin-top: 1rem;
    }

    .result-label {
        color: #dbeafe;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-size: .82rem;
        margin-bottom: .5rem;
    }

    .result-value {
        font-size: clamp(2.5rem, 7vw, 5rem);
        font-weight: 850;
        letter-spacing: -0.06em;
        color: #ffffff;
        line-height: 1;
    }

    .result-unit {
        color: #dbeafe;
        font-size: 1.2rem;
        margin-top: .3rem;
    }

    .info-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: .8rem;
        margin-top: 1rem;
    }

    .info-box {
        padding: .85rem;
        border-radius: 18px;
        background: rgba(2,6,23,.32);
        border: 1px solid rgba(148,163,184,.14);
    }

    .info-k {
        color: #9db5d1;
        font-size: .78rem;
    }

    .info-v {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .15rem;
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
        padding: .85rem 1.1rem;
        font-weight: 800;
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

    .footer {
        color: #64748b;
        text-align: center;
        margin-top: 2.5rem;
        font-size: .86rem;
    }

    /* =========================
    入力ラベルを白・青に変更
    ========================= */
    
    [data-testid="stWidgetLabel"] p {
    color: #D6ECFF !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}
    
    /* NumberInputラベル */
    .stNumberInput label {
    color: #D6ECFF !important;
    font-weight: 700 !important;
    }
    
    /* FileUploaderラベル */
    .stFileUploader label {
    color: #D6ECFF !important;
    font-weight: 700 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    }


    @media (max-width: 760px) {
        .info-row {
            grid-template-columns: 1fr 1fr;
        }
        .hero {
            padding: 1.5rem;
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
        <h1>Elbow Varus Torque<br>Prediction System</h1>
        <p class="sub">
            ミズノのMA-Qセンサから得られる投球データを用いて、
            肘内反ピークトルクを推定します。MA-Q試技CSV、身長、体重を入力してください。
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

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-title">1. Input data</div>', unsafe_allow_html=True)

    maq_csv = st.file_uploader("MA-Q 試技CSV", type=["csv"], help="MA-Qから出力された試技CSVをアップロードしてください。")

    col1, col2 = st.columns(2)
    with col1:
        mass = st.number_input("Body mass [kg]", min_value=20.0, max_value=150.0, value=82.2, step=0.1)
    with col2:
        height = st.number_input("Body height [m]", min_value=1.20, max_value=2.20, value=1.773, step=0.001, format="%.3f")

    with st.expander("Advanced settings"):
        abc = st.number_input("Part 1 frames", min_value=1, max_value=81, value=52, step=1)
        fps = st.number_input("Sampling frequency [Hz]", min_value=1.0, max_value=2000.0, value=500.0, step=1.0)

    estimate_clicked = st.button("Estimate torque", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">2. Model information</div>
            <div class="small-muted">
                This tool estimates peak elbow varus torque from MA-Q acceleration and gyroscope data.
                The fixed CB.csv is used internally for global coordinate definition.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="panel" style="margin-top:1rem;">
            <div class="panel-title">Fixed calibration file</div>
            <div class="small-muted">
                Status: {"Available" if os.path.exists(CB_FIXED_PATH) else "Not found"}<br>
                Path: <code>{CB_FIXED_PATH}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Result area
# =========================
if estimate_clicked:

    if maq_csv is None:
        st.error("MA-Q 試技CSVをアップロードしてください。")
    elif not os.path.exists(CB_FIXED_PATH):
        st.error(f"固定CB.csvが見つかりません: {CB_FIXED_PATH}")
    else:
        try:
            with st.spinner("Analyzing MA-Q data..."):
                result = estimate_torque_from_uploaded_maq_csv(
                    maq_csv_file=maq_csv,
                    mass=mass,
                    height=height,
                    cb_csv_path=CB_FIXED_PATH,
                    abc=int(abc),
                    fps=float(fps),
                )

            torque = result["estimated_peak_torque"]

            _, center, _ = st.columns([1, 6, 1])

            with center:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Estimated peak elbow varus torque</div>
                        <div class="result-value">{torque:.2f}</div>
                        <div class="result-unit">N･m</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="info-row">
                        <div class="info-box">
                            <div class="info-k">Release frame</div>
                            <div class="info-v">{result["release_idx"]}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-k">Used frames</div>
                            <div class="info-v">{result["used_frames"]}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-k">Frames after release</div>
                            <div class="info-v">{result["frames_after_release"]}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-k">Mass / Height</div>
                            <div class="info-v">{mass:.1f} kg / {height:.3f} m</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("Processing details"):
                    st.write({
                        "release_idx": result["release_idx"],
                        "release_val": result["release_val"],
                        "frames_after_release": result["frames_after_release"],
                        "used_frames": result["used_frames"],
                        "flip_points": result["flip_points"],
                    })

                with st.expander("Extracted features"):
                    feat_df = pd.DataFrame(
                        [{"feature": k, "value": v} for k, v in result["features"].items()]
                    )
                    st.dataframe(feat_df, use_container_width=True)

        except Exception as e:
            st.error("推定中にエラーが発生しました。")
            st.exception(e)
