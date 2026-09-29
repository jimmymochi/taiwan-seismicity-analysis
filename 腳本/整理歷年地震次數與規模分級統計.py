# -*- coding: utf-8 -*-
"""
整理臺灣歷年地震次數與規模分級統計
=====================================================
計算 1994–2026 年（32.58年、873,145筆事件）每年：
1. 總地震筆數
2. 最大地震規模 (Mmax) 與發生日期
3. 平均規模
4. 各規模區間筆數：
   - 微震 (M < 3.0)
   - 輕震 (3.0 <= M < 4.0)
   - 中震 (4.0 <= M < 5.0)
   - 強震 (5.0 <= M < 6.0)
   - 烈震 (6.0 <= M < 7.0)
   - 大震 (M >= 7.0)
輸出 CSV 統計表與兩子圖高解析度圖表。
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "地震目錄")
OUTPUT_CSV = os.path.join(project_dir, "數據", "台灣歷年地震次數與規模分級統計表_1994_2026.csv")
OUTPUT_PNG = os.path.join(project_dir, "台灣歷年地震次數與規模分級統計圖.png")

def main():
    print("[1/3] 讀取地震原始目錄進行年度統計...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "GDMS_*.csv")))
    dfs = [pd.read_csv(f, usecols=['date', 'time', 'ML', 'lat', 'lon', 'depth']) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['ML'] = pd.to_numeric(df['ML'], errors='coerce')
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    df = df.dropna(subset=['year', 'ML'])
    df['year'] = df['year'].astype(int)

    years = sorted(df['year'].unique())
    records = []
    for y in years:
        sub = df[df['year'] == y]
        max_idx = sub['ML'].idxmax()
        max_row = sub.loc[max_idx]
        records.append({
            '年度': y,
            '總地震筆數': len(sub),
            '最大規模': round(float(max_row['ML']), 2),
            '最大規模日期': str(max_row['date']),
            '平均規模': round(float(sub['ML'].mean()), 2),
            '微震(M<3.0)': int((sub['ML'] < 3.0).sum()),
            '輕震(3.0<=M<4.0)': int(((sub['ML'] >= 3.0) & (sub['ML'] < 4.0)).sum()),
            '中震(4.0<=M<5.0)': int(((sub['ML'] >= 4.0) & (sub['ML'] < 5.0)).sum()),
            '強震(5.0<=M<6.0)': int(((sub['ML'] >= 5.0) & (sub['ML'] < 6.0)).sum()),
            '烈震(6.0<=M<7.0)': int(((sub['ML'] >= 6.0) & (sub['ML'] < 7.0)).sum()),
            '大震(M>=7.0)': int((sub['ML'] >= 7.0).sum())
        })

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"[2/3] CSV 統計表已儲存至：{OUTPUT_CSV}")

    # 繪製圖表
    print("[3/3] 繪製歷年地震次數與規模變化圖...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), dpi=300, facecolor='#fafafa')
    plt.subplots_adjust(hspace=0.25, top=0.93, bottom=0.08, left=0.07, right=0.95)

    # 1. 歷年次數堆疊長條圖
    bar_years = res_df['年度'].values
    m_micro = res_df['微震(M<3.0)'].values
    m_minor = res_df['輕震(3.0<=M<4.0)'].values
    m_med = res_df['中震(4.0<=M<5.0)'].values
    m_strong = res_df['強震(5.0<=M<6.0)'].values
    m_major = res_df['烈震(6.0<=M<7.0)'].values
    m_great = res_df['大震(M>=7.0)'].values

    ax1.set_facecolor('#ffffff')
    ax1.set_title("(A) 臺灣歷年地震總筆數與規模分級堆疊柱狀圖 (1994–2026年，共 873,145 筆)", fontsize=13, fontweight='bold', pad=10)

    p1 = ax1.bar(bar_years, m_micro, label='微震 (M < 3.0)', color='#90caf9', alpha=0.85)
    p2 = ax1.bar(bar_years, m_minor, bottom=m_micro, label='輕震 (3.0 ≤ M < 4.0)', color='#a5d6a7', alpha=0.85)
    p3 = ax1.bar(bar_years, m_med, bottom=m_micro+m_minor, label='中震 (4.0 ≤ M < 5.0)', color='#ffb74d', alpha=0.9)
    p4 = ax1.bar(bar_years, m_strong, bottom=m_micro+m_minor+m_med, label='強震 (5.0 ≤ M < 6.0)', color='#ff7043', alpha=0.9)
    p5 = ax1.bar(bar_years, m_major, bottom=m_micro+m_minor+m_med+m_strong, label='烈震 (6.0 ≤ M < 7.0)', color='#e53935', alpha=0.95)
    p6 = ax1.bar(bar_years, m_great, bottom=m_micro+m_minor+m_med+m_strong+m_major, label='大震 (M ≥ 7.0)', color='#880e4f', alpha=1.0)

    # 標註每年的總數
    for y, total in zip(bar_years, res_df['總地震筆數']):
        ax1.text(y, total + 600, f"{total:,}", ha='center', va='bottom', fontsize=7, rotation=45, color='#333333')

    ax1.set_xlim(1993, 2027)
    ax1.set_xticks(bar_years)
    ax1.set_xticklabels(bar_years, rotation=45, fontsize=8.5)
    ax1.set_ylabel("年度地震次數 (次)", fontsize=10)
    ax1.set_ylim(0, 56000)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', ncol=6, fontsize=9, framealpha=0.9)

    # 2. 歷年最大地震規模與歷史大震標註
    ax2.set_facecolor('#ffffff')
    ax2.set_title("(B) 臺灣歷年最大地震規模 (Mmax) 與重大地震事件演變", fontsize=13, fontweight='bold', pad=10)

    max_mags = res_df['最大規模'].values
    ax2.plot(bar_years, max_mags, color='#d32f2f', marker='o', linewidth=2.0, markersize=6, label='該年最大地震規模 Mmax')
    ax2.axhline(6.0, color='#ff9800', linestyle='--', linewidth=1.2, label='規模 6.0 破壞強震門檻')
    ax2.axhline(7.0, color='#b71c1c', linestyle='-', linewidth=1.5, label='規模 7.0 重大災害大震門檻')

    # 標註代表性歷史大震
    events_to_mark = {
        1994: ("1994-05-24 M6.60\n花蓮外海", 15),
        1999: ("1999-09-21 M7.30\n921集集大地震", 15),
        2002: ("2002-03-31 M6.80\n331花蓮外海", 12),
        2006: ("2006-12-26 M6.99\n恆春外海雙震", 12),
        2009: ("2009-12-19 M6.92\n花蓮外海", 12),
        2016: ("2016-02-06 M6.60\n高雄美濃地震", 12),
        2018: ("2018-02-06 M6.26\n花蓮地震", 12),
        2022: ("2022-09-18 M6.83\n台東池上地震", 12),
        2024: ("2024-04-03 M7.19\n花蓮0403大地震", 15),
        2025: ("2025-12-27 M7.01\n外海強震", 12)
    }

    for y, (label_txt, offset_y) in events_to_mark.items():
        val = res_df[res_df['年度'] == y]['最大規模'].values[0]
        ax2.annotate(label_txt, xy=(y, val), xytext=(y, val + 0.25),
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#880e4f',
                     arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.0),
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffebee', edgecolor='#b71c1c', alpha=0.85, lw=0.6))

    ax2.set_xlim(1993, 2027)
    ax2.set_xticks(bar_years)
    ax2.set_xticklabels(bar_years, rotation=45, fontsize=8.5)
    ax2.set_ylabel("芮氏規模 ML", fontsize=10)
    ax2.set_ylim(5.0, 8.2)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.suptitle("交通部中央氣象署 1994–2026 年臺灣地震年度統計彙整 (總計 873,145 筆)", fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"圖表已輸出至：{OUTPUT_PNG}")

if __name__ == '__main__':
    main()
