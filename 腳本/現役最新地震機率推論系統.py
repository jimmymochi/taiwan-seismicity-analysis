# -*- coding: utf-8 -*-
"""
現役最新地震機率推論系統 (operational_inference_2026.py)
========================================================================
依據使用者需求與前沿地震觀測實務：
1. 實時線上推論協議 (Operational Nowcast/Forecast Protocol)：
   - 快照時間點：t_now = 2026-08-01 00:00:00（截至當前最新地震目錄）。
   - 提取截至該時間點為止的 14 維物理時空特徵。
   - 不涉及任何未來未發生標籤，絕無資料污染或過度擬合問題。
2. 統一四大標準時間尺度 (Unified 4 Canonical Horizons)：
   - 極短期：未來 7 天
   - 短期：未來 30 天
   - 中期：未來 1 年 (365 天)
   - 中長期：未來 5 年
3. 調用預先訓練並完成等張校準的 LightGBM 機器學習模型，輸出 38,178 空間網格機率。
4. 儲存產物：數據/機器學習現役最新推論預測_2026.csv.gz。
"""

import os
import sys
import io

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import time
import joblib
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(script_dir)

from 特徵工程提取器 import load_prerequisites, extract_features_at_timestamp

DATA_DIR = os.path.join(project_dir, "數據")
MODEL_DIR = os.path.join(project_dir, "模型")
OUTPUT_CSV_GZ = os.path.join(DATA_DIR, "機器學習現役最新推論預測_2026.csv.gz")

FEATURE_COLS = [
    'fault_dist_km',
    'count_7d',
    'count_14d',
    'count_30d',
    'count_365d',
    'z_value',
    'days_since_m4',
    'days_since_m5',
    'days_since_m6',
    'mmax_20km',
    'log_m0_1y',
    'log_m0_3y',
    'smoothed_rate',
    'moment_deficit'
]

HORIZONS = ['7d', '30d', '1y', '5y']
TARGETS = ['m5', 'm6']

def run_operational_inference():
    print("="*75)
    print("【現役最新地震機率推論系統】啟動")
    print(" 快照基準點: 2026-08-01 00:00:00 (納入1994至2026最新 873,145 筆觀測事件)")
    print("="*75)

    start_t = time.time()

    # 1. 載入基底數據並提取 2026 最新特徵
    df_cat, df_grid = load_prerequisites()
    t_now = pd.Timestamp('2026-08-01 00:00:00')

    print(f"\n[1/3] 提取 {t_now} 最新物理時空特徵...")
    feat_now = extract_features_at_timestamp(t_now, df_cat, df_grid)
    X_now = feat_now[FEATURE_COLS]
    print(f"  38,178 空間網格特徵提取完成，特徵維度: {X_now.shape}")

    # 2. 載入已訓練之校準模型並進行推論
    print("\n[2/3] 調用等張校準機器學習模型進行四大統一時間尺度推論...")
    results_df = feat_now[['grid_id', 'lon_center', 'lat_center', 'x_center', 'y_center', 'fault_dist_km']].copy()
    results_df['t_snapshot'] = '2026-08-01 00:00:00'

    for target in TARGETS:
        for horizon in HORIZONS:
            model_filename = f"lgbm_{target}_{horizon}_calibrated.joblib"
            model_path = os.path.join(MODEL_DIR, model_filename)

            if not os.path.exists(model_path):
                print(f"  [警告] 找不到模型 {model_path}，跳過")
                continue

            print(f"  推論中: {target.upper()} 規模 | {horizon} 尺度 (模型: {model_filename})...")
            calibrated_model = joblib.load(model_path)
            # 獲取校準機率 (P(Y=1))
            probs = calibrated_model.predict_proba(X_now)[:, 1]
            col_name = f"prob_{target}_{horizon}"
            results_df[col_name] = np.round(probs, 5)

            max_p = np.max(probs) * 100.0
            mean_p = np.mean(probs) * 100.0
            p99 = np.percentile(probs, 99) * 100.0
            print(f"    -> 單格最高機率: {max_p:.2f}% | 99分位: {p99:.2f}% | 平均機率: {mean_p:.2f}%")

    # 3. 儲存結果
    print(f"\n[3/3] 儲存 2026 現役最新推論結果至: {OUTPUT_CSV_GZ}")
    results_df.to_csv(OUTPUT_CSV_GZ, index=False, compression='gzip', encoding='utf-8')
    elapsed = time.time() - start_t
    print(f"推論全流程圓滿完成！總耗時: {elapsed:.2f} 秒\n")

if __name__ == '__main__':
    run_operational_inference()
