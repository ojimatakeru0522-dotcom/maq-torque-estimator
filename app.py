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
    layout="centered"
)

st.title("MA-Q 肘内反ピークトルク推定")
st.write("MA-Qの試技CSV、身長、体重を入力すると推定ピークトルクを算出します。")

if not os.path.exists(CB_FIXED_PATH):
    st.error(
        f"固定CB.csvが見つかりません: `{CB_FIXED_PATH}`\n\n"
        "アプリフォルダ内に `data` フォルダを作成し、その中に `CB.csv` を置いてください。"
    )
else:
    st.info(f"固定CB.csvを使用中: `{CB_FIXED_PATH}`")

maq_csv = st.file_uploader("MA-Q 試技CSVをアップロード", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    mass = st.number_input("体重 mass [kg]", min_value=20.0, max_value=150.0, value=82.2, step=0.1)
with col2:
    height = st.number_input("身長 height [m]", min_value=1.20, max_value=2.20, value=1.773, step=0.001, format="%.3f")

with st.expander("詳細設定"):
    abc = st.number_input("分割フレーム数 part1", min_value=1, max_value=81, value=52, step=1)
    fps = st.number_input("サンプリング周波数 [Hz]", min_value=1.0, max_value=2000.0, value=500.0, step=1.0)

if st.button("トルクを推定する", type="primary"):
    if maq_csv is None:
        st.error("MA-Q 試技CSVをアップロードしてください。")
    elif not os.path.exists(CB_FIXED_PATH):
        st.error(f"固定CB.csvが見つかりません: {CB_FIXED_PATH}")
    else:
        try:
            result = estimate_torque_from_uploaded_maq_csv(
                maq_csv_file=maq_csv,
                mass=mass,
                height=height,
                cb_csv_path=CB_FIXED_PATH,
                abc=int(abc),
                fps=float(fps),
            )

            st.success("推定が完了しました。")
            st.metric("推定ピークトルク", f"{result['estimated_peak_torque']:.2f} N･m")

            st.subheader("処理情報")
            st.write({
                "release_idx": result["release_idx"],
                "release_val": result["release_val"],
                "frames_after_release": result["frames_after_release"],
                "used_frames": result["used_frames"],
                "flip_points": result["flip_points"],
            })

            with st.expander("作成された特徴量を見る"):
                feat_df = pd.DataFrame(
                    [{"feature": k, "value": v} for k, v in result["features"].items()]
                )
                st.dataframe(feat_df, use_container_width=True)

        except Exception as e:
            st.error("推定中にエラーが発生しました。")
            st.exception(e)
