# -*- coding: utf-8 -*-
"""
臺灣地震時空預測管線 - 模組四：CSEP 國際評估與檢定器 (csep_evaluator.py)
========================================================================
依據任務規範：
1. 嚴格對標兩大地震學標準基準線 (Baselines)：
   - Baseline 1: 空間平滑時間不變帕松基準線 (Franken / Helmstetter Poisson Baseline)
   - Baseline 2: 空間均勻隨機基準線 (Uniform Random Baseline)
2. CSEP 國際地震檢驗標準指標：
   - PR-AUC (Precision-Recall 曲線下面積 / Average Precision)
   - Brier Score 與 Brier Skill Score (BSS 相對帕松基準技能分數: BSS = 1 - Brier_ml / Brier_poisson)
   - Molchan 誤差圖 (Molchan Error Diagram): 漏報率 (Miss Rate) vs. 警報面積佔比 (Alarm Fraction)
   - 信息增益 (Information Gain per earthquake): (LL_ml - LL_poisson) / N_quakes
3. 輸出產物：
   - 評驗指標對照表: 數據/CSEP國際評驗指標對照表.csv
   - 高解析度 CSEP 評估看板: 台灣地震機器學習CSEP國際檢驗評估看板.png
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "數據")
TEST_PARQUET = os.path.join(DATA_DIR, "機器學習盲測特徵集_2022_2026.parquet")
OUTPUT_METRICS_CSV = os.path.join(DATA_DIR, "CSEP國際評驗指標對照表.csv")
OUTPUT_PNG = os.path.join(project_dir, "台灣地震機器學習CSEP國際檢驗評估看板.png")

HORIZONS = ['7d', '30d', '1y', '3y', '5y']
HORIZON_NAMES = {
    '7d': '極短期 (7天)',
    '30d': '短期 (30天)',
    '1y': '中期 (1年)',
    '3y': '中長期 (3年)',
    '5y': '長期 (5年)'
}

def compute_csep_evaluations():
    print("[1/4] 載入 2022–2026 盲測集機率預測與真實破裂標籤...")
    test_csv = TEST_PARQUET.replace('.parquet', '.csv.gz')
    if os.path.exists(TEST_PARQUET):
        try:
            test_df = pd.read_parquet(TEST_PARQUET)
        except Exception:
            test_df = pd.read_csv(test_csv)
    else:
        test_df = pd.read_csv(test_csv)
    
    records = []
    plot_data = {}

    print("[2/4] 演算 CSEP 指標：PR-AUC、Brier、BSS、信息增益與 Molchan 曲線...")
    for h in HORIZONS:
        for tgt in ['m5', 'm6']:
            label_col = f"label_{tgt}_{h}"
            prob_col = f"prob_{tgt}_{h}"
            
            if prob_col not in test_df.columns:
                print(f"  [提示] 缺少預測欄位 {prob_col}，略過...")
                continue
                
            y_true = test_df[label_col].values
            p_ml = np.clip(test_df[prob_col].values, 1e-6, 1.0 - 1e-6)
            n_pos = int(np.sum(y_true))
            n_total = len(y_true)
            
            # --- 基準線 1: 空間平滑時間不變帕松基準 (Poisson Baseline) ---
            # 依據平滑歷史發生率估計時間跨度機率
            smoothed_rate = test_df['smoothed_rate'].values
            delta_years = 7.0/365.25 if h=='7d' else (30.0/365.25 if h=='30d' else float(h[0]))
            mu_poisson = smoothed_rate * (delta_years / 28.0) * (0.05 if tgt=='m5' else 0.01)
            p_poisson = np.clip(1.0 - np.exp(-mu_poisson), 1e-6, 1.0 - 1e-6)
            
            # --- 基準線 2: 均勻隨機基準線 (Uniform Random Baseline) ---
            p_random = np.full(n_total, np.clip(n_pos / max(n_total, 1), 1e-6, 1.0 - 1e-6))

            # 1. PR-AUC (Average Precision)
            pr_auc_ml = average_precision_score(y_true, p_ml) if n_pos > 0 else 0.0
            pr_auc_poisson = average_precision_score(y_true, p_poisson) if n_pos > 0 else 0.0
            pr_auc_random = n_pos / n_total

            # 2. Brier Score & BSS (Brier Skill Score)
            brier_ml = brier_score_loss(y_true, p_ml)
            brier_poisson = brier_score_loss(y_true, p_poisson)
            bss = 1.0 - (brier_ml / brier_poisson) if brier_poisson > 0 else 0.0

            # 3. Log-likelihood & Information Gain (CSEP 標準)
            ll_ml = np.sum(y_true * np.log(p_ml) + (1 - y_true) * np.log(1 - p_ml))
            ll_poisson = np.sum(y_true * np.log(p_poisson) + (1 - y_true) * np.log(1 - p_poisson))
            info_gain_per_eq = (ll_ml - ll_poisson) / max(n_pos, 1)

            # 4. Molchan Error Diagram 數據點
            # 依照模型預測機率降序排列
            sort_idx = np.argsort(p_ml)[::-1]
            y_sorted = y_true[sort_idx]
            cum_hits = np.cumsum(y_sorted)
            cum_alarms = np.arange(1, n_total + 1)
            
            alarm_fraction = cum_alarms / n_total  # tau (警報面積比)
            miss_rate = 1.0 - (cum_hits / max(n_pos, 1))  # nu (漏報率)

            # 降採樣以利繪圖 (抽取 200 點)
            sub_sample_idx = np.linspace(0, n_total - 1, 200).astype(int)
            plot_data[f"{tgt}_{h}"] = {
                'tau': alarm_fraction[sub_sample_idx],
                'nu': miss_rate[sub_sample_idx],
                'y_true': y_true,
                'p_ml': p_ml,
                'p_poisson': p_poisson,
                'n_pos': n_pos
            }

            records.append({
                '時間尺度': HORIZON_NAMES[h],
                '規模門檻': f"ML >= {tgt[-1]}.0",
                '盲測正樣本數': n_pos,
                '機器學習_PRAUC': round(pr_auc_ml, 4),
                '帕松基準_PRAUC': round(pr_auc_poisson, 4),
                '隨機基準_PRAUC': round(pr_auc_random, 5),
                '機器學習_Brier': round(brier_ml, 5),
                '帕松基準_Brier': round(brier_poisson, 5),
                'Brier技能分數(BSS)': round(bss, 4),
                '地震信息增益(IG/eq)': round(info_gain_per_eq, 3),
            })
            print(f"  [{h} | {tgt.upper()}] 正樣本: {n_pos} | PR-AUC: {pr_auc_ml:.4f} (帕松:{pr_auc_poisson:.4f}) | BSS: {bss:.4f} | IG: {info_gain_per_eq:.3f}")

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding='utf-8-sig')
    print(f"  CSEP 指標表已儲存至: {OUTPUT_METRICS_CSV}")

    # 繪製高解析度 CSEP 評估看板
    print("[3/4] 繪製 CSEP 國際評驗圖版 (Molchan 曲線、PR 曲線、BSS 與 IG 對比)...")
    fig = plt.figure(figsize=(20, 16), dpi=300, facecolor='#fafafa')
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.22, top=0.92, bottom=0.08, left=0.08, right=0.95)

    # ----------------------------------------------------
    # 子圖 1 (左上): Molchan Error Diagram (莫爾強漏報率 vs. 警報面積圖)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#ffffff')
    ax1.set_title("(A) 國際 CSEP 莫爾強檢驗圖 (Molchan Error Diagram)\n[曲線越貼近左下角原點，代表預測越優異]", 
                  fontsize=12, fontweight='bold', pad=10)
    
    # 隨機基準線 (對角線 tau + nu = 1)
    ax1.plot([0, 1], [1, 0], color='gray', linestyle='--', linewidth=1.5, label='空間均勻隨機基準線 (nu + tau = 1)')
    
    colors_h = {'7d': '#1f77b4', '30d': '#2ca02c', '1y': '#ff7f0e', '3y': '#d62728', '5y': '#9467bd'}
    
    for h in HORIZONS:
        key = f"m5_{h}"
        if key in plot_data:
            d = plot_data[key]
            ax1.plot(d['tau'], d['nu'], label=f"{HORIZON_NAMES[h]} (M>=5.0)", 
                     color=colors_h[h], linewidth=2.0)

    ax1.set_xlabel("警報空間佔比 (Alarm Space-Time Fraction, tau)", fontsize=10)
    ax1.set_ylabel("地震漏報率 (Miss Rate, nu)", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 2 (右上): PR Curves (Precision-Recall 曲線，針對極端不平衡正樣本)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#ffffff')
    ax2.set_title("(B) 盲測精準率-召回率曲線 (Precision-Recall Curves, M >= 5.0)\n[極端類別不平衡下之核心鑑別力評估]", 
                  fontsize=12, fontweight='bold', pad=10)

    for h in HORIZONS:
        key = f"m5_{h}"
        if key in plot_data and plot_data[key]['n_pos'] > 0:
            d = plot_data[key]
            prec, rec, _ = precision_recall_curve(d['y_true'], d['p_ml'])
            ap = average_precision_score(d['y_true'], d['p_ml'])
            ax2.plot(rec, prec, label=f"{HORIZON_NAMES[h]} (PR-AUC = {ap:.3f})", 
                     color=colors_h[h], linewidth=2.0)

    ax2.set_xlabel("召回率 (Recall / 命中比例)", fontsize=10)
    ax2.set_ylabel("精準率 (Precision / 警報命中率)", fontsize=10)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 3 (左下): Brier Skill Score (BSS 相對帕松基準技能分數)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#ffffff')
    ax3.set_title("(C) Brier 技能分數 (Brier Skill Score vs. 帕松基準線)\n[BSS > 0 代表機器學習模型顯著超越傳統帕松模型]", 
                  fontsize=12, fontweight='bold', pad=10)

    df_m5_sub = res_df[res_df['規模門檻'] == 'ML >= 5.0']
    x_pos = np.arange(len(df_m5_sub))
    bss_vals = df_m5_sub['Brier技能分數(BSS)'].values
    
    bars = ax3.bar(x_pos, bss_vals, color='#43a047', alpha=0.85, width=0.45, label='BSS (M >= 5.0)')
    ax3.axhline(0, color='red', linestyle='--', linewidth=1.2)

    for b, val in zip(bars, bss_vals):
        ax3.text(b.get_x() + b.get_width()/2., max(val, 0) + 0.02, f"+{val*100:.1f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2e7d32')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df_m5_sub['時間尺度'], fontsize=9.5)
    ax3.set_ylabel("Brier 技能分數 (BSS)", fontsize=10)
    ax3.set_ylim(-0.1, max(bss_vals.max() * 1.3, 0.5))
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=9)

    # ----------------------------------------------------
    # 子圖 4 (右下): CSEP 信息增益 (Information Gain per earthquake)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#ffffff')
    ax4.set_title("(D) CSEP 每場地震信息增益 (Information Gain per Earthquake)\n[相較於帕松基準之對數概似增益率 (越高越優)]", 
                  fontsize=12, fontweight='bold', pad=10)

    ig_vals = df_m5_sub['地震信息增益(IG/eq)'].values
    bars_ig = ax4.bar(x_pos, ig_vals, color='#1e88e5', alpha=0.85, width=0.45, label='Information Gain')

    for b, val in zip(bars_ig, ig_vals):
        ax4.text(b.get_x() + b.get_width()/2., max(val, 0) + 0.05, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1565c0')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df_m5_sub['時間尺度'], fontsize=9.5)
    ax4.set_ylabel("信息增益 (nats / earthquake)", fontsize=10)
    ax4.set_ylim(0, max(ig_vals.max() * 1.3, 1.5))
    ax4.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=9)

    plt.suptitle("臺灣地震時空機器學習預測管線 - 國際 CSEP 盲測評驗綜合看板 (2022–2026年盲測集)",
                 fontsize=15, fontweight='bold', y=0.97)

    plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[4/4] CSEP 看板圖表已輸出至: {OUTPUT_PNG}")

if __name__ == '__main__':
    compute_csep_evaluations()
