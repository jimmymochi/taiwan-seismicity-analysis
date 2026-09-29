# -*- coding: utf-8 -*-
"""
臺灣歷年地震活動度與重大強震全景分析圖 (lieflat-charts 規範)
========================================================================
依據 lieflat-charts 視覺語法：
1. 拒絕雜亂五彩堆疊圖：將「全量監測活動度」與「顯著破壞性強震」脫鉤分層呈現，
   避免高達 90% 以上的微震擠壓強震視覺通道。
2. 上圖 (A)：全臺歷年地震總監測筆數（1994–2026）
   - 採用青瓷藍階 (Porcelain) 梯階柱狀圖 (Rung Bars)。
   - 標註 1999 年集集 (49,556 筆)、2024 年花蓮 0403 (34,408 筆+)。
   - 標註 2012 年氣象署觀測網密化升級（下探偵測極限）。
3. 下圖 (B)：破壞性與顯著強震歷年次數 (M>=5.0, M>=6.0, M>=7.0)
   - 採用帶圓珠之髮絲針狀圖 (Lollipop / Hairline Beaded Stems)。
   - 標記全臺 10 大歷史標竿震災。
   - 輔助右軸：累積地震能量/地震矩釋放曲線 (Energy / Moment Release)。
4. 排版與字體遵循 lieflat 規範：結論式標題、副標題四件套、來源底注。
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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

CSV_PATH = os.path.join(project_dir, "數據", "台灣歷年地震次數與規模分級統計表_1994_2026.csv")
CATALOG_PATH = os.path.join(project_dir, "數據", "全台灣地震彙整目錄_1994_2026.csv")
OUTPUT_PNG = os.path.join(project_dir, "看板圖表", "臺灣歷年地震活動度與重大強震全景分析圖.png")

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# lieflat porcelain / wire 精神色板
BG_COLOR = "#fbfaf8"       # 暖紙底色
CARD_BG = "#ffffff"        # 卡片底色
INK_DARK = "#1a2536"       # 墨色主文字
MUTED_TXT = "#5e6d82"      # 次級文字
FAINT_TXT = "#94a3b8"      # 標籤與刻度
GRID_LINE = "#e8edf2"      # 發絲網格線

# Porcelain 藍階與琥珀朱紅
BLUE_MAIN = "#2b5c8f"      # 主數據青瓷藍
BLUE_LIGHT = "#96b6d8"     # 淺藍
BLUE_ACCENT = "#183a60"    # 深邃海藍
AMBER_HERO = "#d97724"     # 琥珀橙 (M>=6.0)
CRIMSON_HERO = "#b91c1c"   # 胭脂深紅 (M>=7.0)

def main():
    print("[1/3] 讀取歷年統計表與地震目錄...")
    df_stat = pd.read_csv(CSV_PATH)
    years = df_stat['年度'].values
    totals = df_stat['總地震筆數'].values
    m5_counts = df_stat['強震(5.0<=M<6.0)'].values
    m6_counts = df_stat['烈震(6.0<=M<7.0)'].values
    m7_counts = df_stat['大震(M>=7.0)'].values
    m_max = df_stat['最大規模'].values
    m_max_dates = df_stat['最大規模日期'].values

    # 計算累積地震能量 (焦耳: log10(E) = 4.8 + 1.5*M)
    df_cat = pd.read_csv(CATALOG_PATH)
    df_cat['year'] = pd.to_datetime(df_cat['datetime']).dt.year
    df_cat['energy'] = 10.0 ** (4.8 + 1.5 * df_cat['ML'])
    yearly_energy = df_cat.groupby('year')['energy'].sum().reindex(years).fillna(0).values / 1e15 # 單位 10^15 J

    print("[2/3] 繪製 lieflat-charts 規範雙聯全景圖表...")
    fig = plt.figure(figsize=(19, 12.5), dpi=300, facecolor=BG_COLOR)
    
    # 頂部標頭區域 (Title + Subtitle)
    fig.text(0.06, 0.955, "全臺歷年地震觀測活動度與重大強震全景紀事 (1994–2026)", 
             fontsize=19, fontweight='bold', color=INK_DARK)
    fig.text(0.06, 0.932, "873,145 筆儀器監測紀錄 · 氣象署監測網演進與規模分層 · 破壞性大地震歷史對照 · 32.6 年完整序列", 
             fontsize=10.5, color=MUTED_TXT)

    # 建立兩大子圖 Grid
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.3], top=0.90, bottom=0.09, left=0.06, right=0.94, hspace=0.30)
    
    # ─────────────────────────────────────────────────────────────
    # 圖 (A): 歷年地震總監測筆數 (Rung Bars)
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(CARD_BG)
    for spine in ax1.spines.values():
        spine.set_color(GRID_LINE)
        spine.set_linewidth(1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 繪製細緻柱狀圖
    bars = ax1.bar(years, totals, width=0.62, color=BLUE_MAIN, alpha=0.88, zorder=3, edgecolor='none')

    # 高亮 1999 與 2024 兩大歷史餘震峰值
    for y, bar, tot in zip(years, bars, totals):
        if y == 1999:
            bar.set_color(AMBER_HERO)
            bar.set_alpha(0.95)
            ax1.annotate(f"1999 集集大地震餘震群\n總筆數: {tot:,} 次",
                         xy=(y, tot), xytext=(y - 2.8, tot + 5500),
                         arrowprops=dict(arrowstyle="->", color=AMBER_HERO, lw=1.5),
                         fontsize=9.5, fontweight='bold', color=AMBER_HERO,
                         bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec=AMBER_HERO, lw=1))
        elif y == 2024:
            bar.set_color(CRIMSON_HERO)
            bar.set_alpha(0.95)
            ax1.annotate(f"2024 花蓮0403強震餘震群\n總筆數: {tot:,} 次 (上半年)",
                         xy=(y, tot), xytext=(y - 3.2, tot + 6200),
                         arrowprops=dict(arrowstyle="->", color=CRIMSON_HERO, lw=1.5),
                         fontsize=9.5, fontweight='bold', color=CRIMSON_HERO,
                         bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=CRIMSON_HERO, lw=1))
        elif y == 2012:
            # 氣象署監測網升級
            ax1.axvline(2011.5, color='#94a3b8', linestyle='--', lw=1.2, zorder=2)
            ax1.text(2011.8, 42000, "2012 氣象署密集地震網升級\n(BATS + CWASN 全面聯網，偵測下限下探 Mc=1.8)",
                     fontsize=8.5, color='#475569', style='italic',
                     bbox=dict(boxstyle="square,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8))

    ax1.set_ylabel("年度監測地震總筆數 (次/年)", fontsize=10.5, fontweight='bold', color=INK_DARK, labelpad=8)
    ax1.set_xlim(1992.5, 2027.5)
    ax1.set_ylim(0, 58000)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) if y % 2 == 0 or y in [1999, 2024] else "" for y in years], fontsize=8.5, color=MUTED_TXT)
    ax1.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)

    # 圖 A 內嵌指標說明卡
    ax1.text(0.015, 0.90, "(A) 全臺年度儀器監測總地震筆數 (含極微震 M < 3.0)", transform=ax1.transAxes,
             fontsize=11.5, fontweight='bold', color=INK_DARK)

    # ─────────────────────────────────────────────────────────────
    # 圖 (B): 破壞性與顯著強震 (M>=5.0, M>=6.0, M>=7.0) 歷年 Lollipop 視圖 + 能量曲線
    # ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CARD_BG)
    for spine in ax2.spines.values():
        spine.set_color(GRID_LINE)
        spine.set_linewidth(1.0)
    ax2.spines['top'].set_visible(False)

    # 次軸: 能量釋放
    ax2_energy = ax2.twinx()
    ax2_energy.spines['top'].set_visible(False)
    ax2_energy.spines['left'].set_visible(False)
    ax2_energy.spines['bottom'].set_visible(False)
    ax2_energy.spines['right'].set_color('#cbd5e1')
    ax2_energy.plot(years, yearly_energy, color='#94a3b8', lw=1.6, linestyle='-', marker='o', markersize=3.5, label='年度累積釋放能量 (10^15 J)', zorder=2)
    ax2_energy.set_ylabel("年度地震釋放能量 (10^15 J)", fontsize=9.5, color='#64748b', labelpad=8)
    ax2_energy.tick_params(axis='y', labelcolor='#64748b', labelsize=8.5)
    ax2_energy.set_ylim(0, max(yearly_energy)*1.25)

    # 組合圖例 (不使用缺字元符號)
    ax2.text(0.015, 0.90, "(B) 破壞性與顯著強震年遇次數 (M >= 5.0 與 M >= 6.0) 及重大實震里程碑", 
             transform=ax2.transAxes, fontsize=11.5, fontweight='bold', color=INK_DARK)

    # 繪製 M>=5.0, M>=6.0, M>=7.0 Lollipop
    # 1. M5.0-5.9
    ax2.vlines(years - 0.15, 0, m5_counts, color=BLUE_LIGHT, lw=2.2, label='中強震 (5.0 <= M < 6.0)', zorder=3)
    ax2.scatter(years - 0.15, m5_counts, color=BLUE_MAIN, s=32, zorder=4)

    # 2. M6.0-6.9 (烈震)
    ax2.vlines(years + 0.15, 0, m6_counts, color='#fdba74', lw=2.8, label='烈震 (6.0 <= M < 7.0)', zorder=3)
    ax2.scatter(years + 0.15, m6_counts, color=AMBER_HERO, s=48, edgecolor='#ffffff', lw=0.8, zorder=4)

    # 3. M>=7.0 (大震)
    m7_mask = m7_counts > 0
    if np.any(m7_mask):
        ax2.scatter(years[m7_mask] + 0.15, m6_counts[m7_mask] + 8, color=CRIMSON_HERO, s=120, marker='*', 
                    label='強大震 (M >= 7.0)', zorder=5, edgecolor='#ffffff', lw=0.8)

    # 標記全臺重大歷史強震事件 (錯落排版，杜絕重疊)
    events = [
        (1999, 14, "1999 921集集 M7.3 (南投)\n烈震14次、強震74次", 28),
        (2002, 5, "2002 331 M6.8 (花蓮外海)", 22),
        (2006, 5, "2006 1226 M7.0 (恆春海域雙震)", 22),
        (2010, 2, "2010 0304 M6.4 (高雄甲仙)", 20),
        (2016, 3, "2016 0206 M6.6 (美濃/維冠)", 32),
        (2018, 2, "2018 0206 M6.2 (花蓮/統帥)", 18),
        (2022, 12, "2022 0918 M6.8 (台東池上/關山)", 30),
        (2024, 10, "2024 0403 M7.2 (花蓮和平外海)\n破裂長度逾70km", 45),
    ]

    for yr, yval, label, y_offset in events:
        ax2.annotate(label, xy=(yr + 0.15, yval),
                     xytext=(yr, yval + y_offset),
                     arrowprops=dict(arrowstyle="->", color='#334155', lw=1.0),
                     fontsize=8.2, color='#0f172a', fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cbd5e1", lw=0.8, alpha=0.95),
                     zorder=6)

    ax2.set_ylabel("強震與烈震發生次數 (次/年)", fontsize=10.5, fontweight='bold', color=INK_DARK, labelpad=8)
    ax2.set_xlabel("觀測年度 (西元)", fontsize=10.5, fontweight='bold', color=INK_DARK, labelpad=10)
    ax2.set_xlim(1992.5, 2027.5)
    ax2.set_ylim(0, 160)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, fontsize=8.5, color=MUTED_TXT, rotation=45)
    ax2.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)

    # 組合圖例
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax2_energy.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper right', frameon=True, 
               facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8.5, ncol=4)

    # 底部來源底注 (Footnote)
    fig.text(0.06, 0.025, 
             "DATA SOURCE: 中央氣象署 (CWA) 地球物理資料管理系統 (GDMS) · 涵蓋 1994/01/01 至 2026/07/30 共 32.58 年 873,145 筆儀器觀測資料 · 視覺規範符合 LIEFLAT-CHARTS MONO/PORCELAIN 標準",
             fontsize=8.2, color=FAINT_TXT)

    print(f"[3/3] 輸出發布級全景圖表至: {OUTPUT_PNG}")
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print("歷年全景分析圖繪製完成！\n")

if __name__ == '__main__':
    main()
