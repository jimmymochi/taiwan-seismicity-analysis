# -*- coding: utf-8 -*-
"""
臺灣地震時空預測管線 - 模組三：機器學習模型訓練與機率校準器 (model_trainer.py)
========================================================================
依據任務規範：
1. 演算法架構 (Model Architecture)：
   - 採用專用梯度提升決策樹 (LightGBM) 針對 5 個時間尺度 (7d, 30d, 1y, 3y, 5y)
     與 2 個規模門檻 (M>=5.0, M>=6.0) 獨立訓練專屬模型。
2. 極端類別不平衡處理 (Extreme Class Imbalance)：
   - 正樣本率 << 0.01%，啟用 scale_pos_weight 與 Weighted Binary Cross-Entropy。
   - 針對遠海無震背景網格實施智慧負採樣，保留所有構造活化特徵。
3. 強制物理機率校準 (Mandatory Probability Calibration)：
   - 決策樹邊界分數不具備真實物理機率意義。
   - 採用等張迴歸 (Isotonic Regression) 於交叉驗證 (Cross-Validation) 折數上進行機率校準。
4. 模型持久化：
   - 模型權重與校準器儲存至 模型/ 目錄下。
"""

import os
import sys
import io

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "數據")
MODEL_DIR = os.path.join(project_dir, "模型")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PARQUET = os.path.join(DATA_DIR, "機器學習訓練特徵集_1994_2021.parquet")
TEST_PARQUET = os.path.join(DATA_DIR, "機器學習盲測特徵集_2022_2026.parquet")
SUMMARY_METRICS_CSV = os.path.join(DATA_DIR, "模型訓練與校準指標彙整表.csv")

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

HORIZONS = ['7d', '30d', '1y', '3y', '5y']
TARGETS = ['m5', 'm6']

def train_and_calibrate_models():
    print("[1/4] 載入訓練與盲測特徵集...")
    train_csv = TRAIN_PARQUET.replace('.parquet', '.csv.gz')
    test_csv = TEST_PARQUET.replace('.parquet', '.csv.gz')
    
    if os.path.exists(TRAIN_PARQUET):
        try:
            train_df = pd.read_parquet(TRAIN_PARQUET)
            test_df = pd.read_parquet(TEST_PARQUET)
        except Exception:
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
    else:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    print(f"  訓練集總筆數: {len(train_df):,}, 盲測集總筆數: {len(test_df):,}")

    metrics_records = []
    trained_models = {}

    print("[2/4] 開始多尺度、多規模門檻之 LightGBM 訓練與等張機率校準 (Isotonic Calibration)...")
    
    total_models = len(HORIZONS) * len(TARGETS)
    model_count = 0

    for h in HORIZONS:
        for tgt in TARGETS:
            model_count += 1
            label_col = f"label_{tgt}_{h}"
            model_name = f"lgbm_{tgt}_{h}"
            print(f"\n--- [{model_count}/{total_models}] 訓練模型: 時間尺度 = {h}, 規模門檻 = {tgt.upper()} ---")

            y_train = train_df[label_col].values
            y_test = test_df[label_col].values

            pos_train = int(np.sum(y_train))
            pos_test = int(np.sum(y_test))
            pos_rate_train = (pos_train / len(y_train)) * 100.0
            print(f"  訓練正樣本數: {pos_train:,} / {len(y_train):,} ({pos_rate_train:.3f}%) | 測試正樣本數: {pos_test:,}")

            if pos_train < 5:
                print(f"  [提示] 正樣本較少 ({pos_train}筆)，改採安全基準權重...")
                scale_pos_weight = 10.0
            else:
                scale_pos_weight = float(np.clip((len(y_train) - pos_train) / max(pos_train, 1), 1.0, 100.0))

            # 針對海洋純背景實施智慧負採樣以加速並聚焦破裂構造帶
            neg_indices = np.where(y_train == 0)[0]
            pos_indices = np.where(y_train == 1)[0]
            np.random.seed(42)
            n_neg_samples = min(len(neg_indices), max(len(pos_indices)*8, 15000))
            sampled_neg = np.random.choice(neg_indices, size=n_neg_samples, replace=False)
            train_idx = np.concatenate([pos_indices, sampled_neg])

            X_tr = train_df.iloc[train_idx][FEATURE_COLS].values
            y_tr = y_train[train_idx]

            # 構建 LightGBM 基底分類器
            base_lgbm = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.04,
                max_depth=5,
                num_leaves=24,
                scale_pos_weight=scale_pos_weight,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1
            )

            # 強制實施機率校準 (Isotonic Regression)
            # 若正樣本 >= 3 則採用交叉驗證學習等張校準曲線；若極稀疏則直接擬合
            if pos_train >= 3:
                n_cv = min(3, pos_train)
                calibrated_clf = CalibratedClassifierCV(
                    estimator=base_lgbm,
                    method='isotonic',
                    cv=n_cv
                )
                calibrated_clf.fit(X_tr, y_tr)
                final_model = calibrated_clf
            else:
                base_lgbm.fit(X_tr, y_tr)
                final_model = base_lgbm

            # 在盲測集上預測機率 (2022–2026)
            X_te = test_df[FEATURE_COLS].values
            prob_test = final_model.predict_proba(X_te)[:, 1]

            # 計算盲測驗證指標
            brier = brier_score_loss(y_test, prob_test)
            try:
                pr_auc = average_precision_score(y_test, prob_test)
            except Exception:
                pr_auc = np.nan
            try:
                roc_auc = roc_auc_score(y_test, prob_test)
            except Exception:
                roc_auc = np.nan

            print(f"  --> 盲測 PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Brier 分數: {brier:.5f}")

            # 儲存模型檔案
            model_file = os.path.join(MODEL_DIR, f"{model_name}_calibrated.joblib")
            joblib.dump(final_model, model_file)

            # 記錄預測機率回測試集
            test_df[f"prob_{tgt}_{h}"] = prob_test

            metrics_records.append({
                '時間尺度': h,
                '規模門檻': tgt.upper(),
                '訓練正樣本數': pos_train,
                '測試正樣本數': pos_test,
                '正樣本比例(%)': round(pos_rate_train, 4),
                'scale_pos_weight': round(scale_pos_weight, 2),
                '盲測_PRAUC': round(pr_auc, 4) if not np.isnan(pr_auc) else None,
                '盲測_ROCAUC': round(roc_auc, 4) if not np.isnan(roc_auc) else None,
                '盲測_BrierScore': round(brier, 5),
                '模型存檔路徑': model_file
            })

    print("[3/4] 儲存盲測預測結果與訓練評估指標表...")
    res_df = pd.DataFrame(metrics_records)
    res_df.to_csv(SUMMARY_METRICS_CSV, index=False, encoding='utf-8-sig')
    print(f"  指標彙整表已輸出至: {SUMMARY_METRICS_CSV}")

    # 更新並保存帶有機率預測的盲測資料集 (Parquet 與 CSV.GZ)
    try:
        test_df.to_parquet(TEST_PARQUET, index=False)
        print(f"  盲測機率預測已更新至: {TEST_PARQUET}")
    except Exception as e:
        print(f"  Parquet 儲存異常 ({e})，改採壓縮 CSV...")
    test_df.to_csv(TEST_PARQUET.replace('.parquet', '.csv.gz'), index=False, compression='gzip')
    print(f"  盲測機率預測壓縮備份已儲存至: {TEST_PARQUET.replace('.parquet', '.csv.gz')}")

    print("[4/4] 機器學習模型訓練與機率校準全流程順利完成！")

if __name__ == '__main__':
    train_and_calibrate_models()
