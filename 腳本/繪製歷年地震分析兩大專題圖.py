# -*- coding: utf-8 -*-
"""
繪製歷年地震分析兩大獨立專題圖 (lieflat-charts 規範)
========================================================================
依據使用者要求，將原先擠在同一張圖的資訊拆分為兩張獨立、高資訊量、發布級專題圖：

專題圖一：臺灣歷年地震總筆數與監測網演進分析圖
- 聚焦 1994–2026 全臺 873,145 筆觀測事件的活動度時序
- 解析 2012 氣象署 BATS + CWASN 寬頻監測網升級帶來的偵測極限下探 (Mc=1.8)
- 呈現微震 (M<3.0) 與有感震的歷年比例變化
- 標註 1999 集集、2024 花蓮等重大餘震群巔峰

專題圖二：臺灣歷年重大強震與能量釋放歷史紀事圖
- 專門解析破壞性強震 (M>=5.0, M>=6.0, M>=7.0)
- 標註 12 大歷史標竿震災之精確日期、震央位置、深度與災損影響
- 疊加歷年地震釋放能量 (10^15 J) 與累積地震矩曲線
- 提供平均回歸期、規模頻率分佈指標
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

CSV_PATH = os.path.join(project_dir, "數據", "台灣歷年地震次數與規模分級統計表_1994_2026.csv")
CATALOG_PATH = os.path.join(project_dir, "數據", "全台灣地震彙整目錄_1994_2026.csv")
CHARTS_DIR = os.path.join(project_dir, "看板圖表")

OUT_PNG1 = os.path.join(CHARTS_DIR, "臺灣歷年地震總筆數與監測網演進分析圖.png")
OUT_PNG2 = os.path.join(CHARTS_DIR, "臺灣歷年重大強震與能量釋放歷史紀事圖.png")

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# lieflat porcelain & editorial 色板
BG_COLOR = "#fbfaf8"       # 暖紙白底
CARD_BG = "#ffffff"        # 卡片底白
INK_DARK = "#0f172a"       # 墨色主字
MUTED_TXT = "#475569"      # 次級文字
FAINT_TXT = "#94a3b8"      # 標籤與刻度
GRID_LINE = "#e2e8f0"      # 發絲網格線

BLUE_MAIN = "#2b5c8f"      # 青瓷藍
BLUE_LIGHT = "#93c5fd"     # 淺藍 (微震)
BLUE_DARK = "#1e3a5f"      # 深藍
AMBER_HERO = "#d97724"     # 琥珀金 (M>=6.0)
CRIMSON_HERO = "#b91c1c"   # 胭脂朱紅 (M>=7.0)

def draw_chart1_total_seismicity(df_stat):
    """繪製專題圖一：歷年總地震筆數與監測網演進"""
    print("[1/2] 繪製專題圖一：臺灣歷年地震總筆數與監測網演進分析圖...")
    years = df_stat['年度'].values
    totals = df_stat['總地震筆數'].values
    micros = df_stat['微震(M<3.0)'].values
    felt = totals - micros  # M >= 3.0 有感地震
    micro_ratios = (micros / totals) * 100.0

    fig = plt.figure(figsize=(18, 11.2), dpi=300, facecolor=BG_COLOR)
    
    # 標頭 (Conclusion Title + Subtitle)
    fig.text(0.06, 0.958, "全臺歷年地震總監測筆數與觀測網密化演進 (1994–2026)", 
             fontsize=19, fontweight='bold', color=INK_DARK)
    fig.text(0.06, 0.934, "873,145 筆儀器監測紀錄 · 微震 (M < 3.0) 與有感地震分層結構 · 2012 氣象署觀測網密化里程碑 · 32.58 年時序", 
             fontsize=10.5, color=MUTED_TXT)

    # 主子圖
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 0.9], top=0.90, bottom=0.09, left=0.06, right=0.95, hspace=0.32)

    # 1. 主圖：總筆數堆疊條形圖 (微震 vs 有感震)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(CARD_BG)
    for sp in ax1.spines.values():
        sp.set_color(GRID_LINE)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    p1 = ax1.bar(years, micros, width=0.62, color='#94b8db', alpha=0.9, label='微震 (M < 3.0，儀器微震)', zorder=3)
    p2 = ax1.bar(years, felt, bottom=micros, width=0.62, color=BLUE_MAIN, alpha=0.95, label='有感與顯著地震 (M >= 3.0)', zorder=3)

    # 高亮 1999 與 2024
    for y, tot in zip(years, totals):
        if y == 1999:
            ax1.bar(y, tot, width=0.62, color=AMBER_HERO, alpha=0.95, zorder=4)
            ax1.annotate(f"1999 921集集大地震餘震群\n總筆數：{tot:,} 次 (單年最高)\n微震佔比：{micros[years==y][0]/tot*100:.1f}%",
                         xy=(y, tot), xytext=(y - 4.2, 55500),
                         arrowprops=dict(arrowstyle="->", color=AMBER_HERO, lw=1.5),
                         fontsize=8.8, fontweight='bold', color=AMBER_HERO,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#fffbeb", ec=AMBER_HERO, lw=1.2))
        elif y == 2024:
            ax1.bar(y, tot, width=0.62, color=CRIMSON_HERO, alpha=0.95, zorder=4)
            ax1.annotate(f"2024 花蓮0403強震餘震潮\n總筆數：{tot:,} 次 (僅計至7月)\nM5+強震達 144 次破紀錄",
                         xy=(y, tot), xytext=(y - 3.5, tot + 4800),
                         arrowprops=dict(arrowstyle="->", color=CRIMSON_HERO, lw=1.5),
                         fontsize=8.8, fontweight='bold', color=CRIMSON_HERO,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec=CRIMSON_HERO, lw=1.2))

    # 標記 2012 觀測網升級線
    ax1.axvline(2011.5, color='#64748b', linestyle='--', lw=1.5, zorder=2)
    ax1.annotate("2012年：氣象署密集地震網 (CWASN + BATS) 全面啟用\n觀測下限顯著下探 (完整度門檻由 Mc~2.2 降至 Mc~1.8)\n偵測微震能力自然倍增，年監測量常態性突破 3.5 萬次",
                 xy=(2011.5, 43000), xytext=(2012.2, 46500),
                 arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2),
                 fontsize=8.8, color='#1e293b', fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.4", fc="#f1f5f9", ec="#94a3b8", lw=1))

    # 柱頂數值標註 (偶數年或重要年份)
    for y, tot in zip(years, totals):
        if y % 2 == 0 or y in [1999, 2024]:
            ax1.text(y, tot + 600, f"{tot:,}", ha='center', va='bottom', fontsize=7.2, color='#475569', rotation=45)

    ax1.set_ylabel("年度地震監測總數 (次/年)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=8)
    ax1.set_xlim(1992.5, 2027.5)
    ax1.set_ylim(0, 63000)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) if y % 2 == 0 or y in [1999, 2024] else "" for y in years], fontsize=9, color=MUTED_TXT)
    ax1.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)
    ax1.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

    # 2. 下子圖：微震佔比演化曲線 (%)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CARD_BG)
    for sp in ax2.spines.values():
        sp.set_color(GRID_LINE)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.plot(years, micro_ratios, color=BLUE_MAIN, lw=2.0, marker='o', markersize=4, zorder=3)
    ax2.fill_between(years, micro_ratios, 80, color='#e0f2fe', alpha=0.5, zorder=2)
    ax2.axhline(np.mean(micro_ratios), color='#94a3b8', linestyle=':', lw=1.2, label=f"32年平均微震佔比 ({np.mean(micro_ratios):.1f}%)")

    # 標註微震佔比極高年
    ax2.text(2013, 98.2, "2013-2019 高微震監測期 (>95%)", fontsize=8.5, color='#0369a1', fontweight='bold')

    ax2.set_ylabel("微震比例 (%)", fontsize=10, fontweight='bold', color=INK_DARK, labelpad=8)
    ax2.set_xlabel("觀測年度 (西元)", fontsize=10.5, fontweight='bold', color=INK_DARK, labelpad=8)
    ax2.set_xlim(1992.5, 2027.5)
    ax2.set_ylim(75, 102)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, fontsize=8.5, color=MUTED_TXT, rotation=45)
    ax2.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)
    ax2.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5)

    # 底部底注
    fig.text(0.05, 0.02, 
             "DATA SOURCE: 交通部中央氣象署 (CWA) 地球物理資料管理系統 · 涵蓋 1994/01/01 至 2026/07/30 共 873,145 筆 · 視覺規範遵循 LIEFLAT-CHARTS MONO/PORCELAIN 標準",
             fontsize=8.5, color=FAINT_TXT)

    plt.savefig(OUT_PNG1, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  -> 專題圖一已輸出: {OUT_PNG1}")

def draw_chart2_damaging_earthquakes(df_stat, df_cat):
    """繪製專題圖二：重大強震與能量釋放歷史紀事"""
    print("[2/2] 繪製專題圖二：臺灣歷年重大強震與能量釋放歷史紀事圖...")
    years = df_stat['年度'].values
    m5_counts = df_stat['強震(5.0<=M<6.0)'].values
    m6_counts = df_stat['烈震(6.0<=M<7.0)'].values
    m7_counts = df_stat['大震(M>=7.0)'].values

    # 計算累積能量
    df_cat['year'] = pd.to_datetime(df_cat['datetime']).dt.year
    df_cat['energy'] = 10.0 ** (4.8 + 1.5 * df_cat['ML'])
    yearly_energy = df_cat.groupby('year')['energy'].sum().reindex(years).fillna(0).values / 1e15 # 10^15 J

    fig = plt.figure(figsize=(19, 12), dpi=300, facecolor=BG_COLOR)
    
    # 標頭
    fig.text(0.06, 0.958, "臺灣歷年顯著強震、破壞性烈震與能量釋放歷史全景紀事 (1994–2026)", 
             fontsize=19, fontweight='bold', color=INK_DARK)
    fig.text(0.06, 0.934, "M >= 5.0 中強震 (1,217次) · M >= 6.0 烈震 (126次) · M >= 7.0 大震 (4次) · 12 大歷史標竿震災精確標註 · 地震能量釋放對照", 
             fontsize=10.5, color=MUTED_TXT)

    ax = fig.add_axes([0.06, 0.11, 0.88, 0.80])
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_LINE)
    ax.spines['top'].set_visible(False)

    # 次軸：年度累積能量釋放 (10^15 J)
    ax_energy = ax.twinx()
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['left'].set_visible(False)
    ax_energy.spines['bottom'].set_visible(False)
    ax_energy.spines['right'].set_color('#94a3b8')
    
    # 能量填充階梯曲線
    ax_energy.plot(years, yearly_energy, color='#cbd5e1', lw=1.8, linestyle='-', marker='s', markersize=3.5, label='年度累積釋放能量 (10^15 J)', zorder=2)
    ax_energy.fill_between(years, 0, yearly_energy, color='#f1f5f9', alpha=0.6, zorder=1)
    ax_energy.set_ylabel("年度地震釋放總能量 (10^15 焦耳)", fontsize=10.5, color='#64748b', labelpad=10)
    ax_energy.tick_params(axis='y', labelcolor='#64748b', labelsize=9)
    ax_energy.set_ylim(0, max(yearly_energy) * 1.35)

    # 主軸：強震 Lollipop (針狀圓珠圖)
    # 1. M5.0-5.9 中強震
    ax.vlines(years - 0.18, 0, m5_counts, color=BLUE_LIGHT, lw=2.2, label='中強震 (5.0 <= M < 6.0)', zorder=3)
    ax.scatter(years - 0.18, m5_counts, color=BLUE_MAIN, s=36, zorder=4)

    # 2. M6.0-6.9 烈震
    ax.vlines(years + 0.18, 0, m6_counts, color='#fdba74', lw=2.8, label='烈震 (6.0 <= M < 7.0)', zorder=3)
    ax.scatter(years + 0.18, m6_counts, color=AMBER_HERO, s=54, edgecolor='#ffffff', lw=0.8, zorder=4)

    # 3. M>=7.0 歷史大震星標
    m7_mask = m7_counts > 0
    if np.any(m7_mask):
        ax.scatter(years[m7_mask] + 0.18, m6_counts[m7_mask] + 8, color=CRIMSON_HERO, s=150, marker='*', 
                   label='歷史大震 (M >= 7.0)', zorder=5, edgecolor='#ffffff', lw=1.0)

    # 12 大歷史標竿地震完整紀事標記 (精確微調位置，確保不遮擋且不超出圖表邊界)
    major_events = [
        (1994, 29, "1994/05/24 花蓮外海 M6.6\n深度 5.3km，全臺有感", 28, -0.6),
        (1996, 18, "1996/09/05 蘭嶼海域 M7.07\n東部隱沒帶深層大震", 58, 0.0),
        (1998, 15, "1998/07/17 嘉義瑞里 M6.2\n梅山斷層帶延伸，傷亡5人", 26, 0.0),
        (1999, 74, "1999/09/21 南投集集 M7.3\n車籠埔斷層破裂 100km，烈震14次、強震74次\n死亡2,415人，臺灣百年最巨震災", 36, -0.8),
        (2002, 26, "2002/03/31 花蓮外海331 M6.8\n臺北101起重機震落，造成5死", 45, -0.5),
        (2003, 20, "2003/12/10 臺東成功 M6.48\n池上斷層同震潛變蠕變觸發", 24, 0.8),
        (2006, 15, "2006/12/26 恆春海域雙震 M6.99\n震毀國際海纜，癱瘓東亞跨國通訊", 52, 0.0),
        (2010, 28, "2010/03/04 高雄甲仙 M6.42\n潮州斷層構造，高鐵首次出軌", 28, 0.0),
        (2016, 24, "2016/02/06 高雄美濃 M6.60\n場址效應，臺南維冠大樓倒塌117死", 45, -0.6),
        (2018, 30, "2018/02/06 花蓮外海/米崙 M6.26\n米崙斷層破裂，統帥飯店倒塌17死", 24, 0.8),
        (2022, 61, "2022/09/18 臺東池上 M6.83\n中央山脈斷層系統，高寮大橋震斷\n烈震高達12次", 35, -1.0),
        (2024, 144, "2024/04/03 花蓮和平外海 M7.19 (M7.2)\n破裂長度逾 70km，強震高達144次破歷史紀錄\n太魯閣巨石崩落，全臺規模最顯著餘震序列", 16, -1.8),
    ]

    for yr, base_val, text, y_off, x_off in major_events:
        anchor_y = m5_counts[years == yr][0] if yr in [1994, 1999, 2022, 2024] else m6_counts[years == yr][0]
        ax.annotate(text, xy=(yr + 0.18, anchor_y),
                    xytext=(yr + x_off, anchor_y + y_off),
                    arrowprops=dict(arrowstyle="->", color='#334155', lw=1.1),
                    fontsize=8.2, color='#0f172a', fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cbd5e1", lw=0.9, alpha=0.96),
                    zorder=6)

    ax.set_ylabel("顯著強震與烈震發生次數 (次/年)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=8)
    ax.set_xlabel("觀測年度 (西元)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=10)
    ax.set_xlim(1992.5, 2027.5)
    ax.set_ylim(0, 195)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=9, color=MUTED_TXT, rotation=45)
    ax.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)

    # 組合圖例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_energy.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.22, 0.98), frameon=True, 
              facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9.2, ncol=4)

    # 底部底注
    fig.text(0.06, 0.02, 
             "DATA SOURCE: 中央氣象署 (CWA) 地球物理資料管理系統 · 能量換算依據芮氏規模公式 log10(E) = 4.8 + 1.5*M · 視覺規範遵循 LIEFLAT-CHARTS MONO/PORCELAIN 標準",
             fontsize=8.5, color=FAINT_TXT)

    plt.savefig(OUT_PNG2, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  -> 專題圖二已輸出: {OUT_PNG2}")

def main():
    print("="*75)
    print("【繪製歷年地震分析兩大獨立專題圖】啟動")
    print("="*75)
    df_stat = pd.read_csv(CSV_PATH)
    df_cat = pd.read_csv(CATALOG_PATH)

    draw_chart1_total_seismicity(df_stat)
    draw_chart2_damaging_earthquakes(df_stat, df_cat)
    print("\n兩大獨立專題圖繪製圓滿完成！\n")

if __name__ == '__main__':
    main()
