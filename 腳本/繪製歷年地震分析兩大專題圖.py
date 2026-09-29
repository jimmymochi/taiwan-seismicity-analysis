# -*- coding: utf-8 -*-
"""
繪製歷年地震分析兩大獨立專題圖 (重構版：杜絕文字重疊與箭頭歧義)
========================================================================
依據使用者回饋改善：
1. 專題圖一：全臺歷年地震總監測筆數與觀測網密化演進
   - 解決痛點：移除覆蓋在柱狀圖上的文字方框，改為在上方無數據空白區設置「兩大觀測時代導覽帶」。
   - 取消密集旋轉數值，僅在顯著標竿年份（1994, 1999, 2012, 2018, 2024等）清晰標註整齊數值。
   - 確保上下子圖邊界充足，避免文字與圖例、底注碰撞。

2. 專題圖二：臺灣歷年重大強震與能量釋放歷史紀事
   - 解決痛點：原圖將12場地震文字方塊與複雜數據疊加，箭頭指向不明確且互相遮擋。
   - 創新重構：採用「上下雙軌同步聯動架構」：
     * 上軌（35%）：【12 大歷史標竿震災時空里程碑軌道】，12 場地震以奇偶交錯卡片排列，
       垂直指針精準指向時間軸上的確切日期刻度，零重疊、零歧義！
     * 下軌（65%）：【歷年強震次數與累積釋放能量對照】，雙軸清晰呈現中強震(M5+)、烈震(M6+)、
       歷史大震(M7+)星標與年度累積能量曲線，並以垂直導引光帶與上軌完全聯動。
"""

import os
import sys
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

# 字型配置
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# lieflat porcelain 標準色彩系統
BG_COLOR = "#fbfaf8"       # 暖紙底色
CARD_BG = "#ffffff"        # 潔白卡片底
INK_DARK = "#0f172a"       # 墨黑主標
SLATE_DARK = "#1e293b"     # 炭黑內文
MUTED_TXT = "#475569"      # 次級灰
FAINT_TXT = "#94a3b8"      # 淺灰底注
GRID_LINE = "#e2e8f0"      # 髮絲網格線

BLUE_MAIN = "#1d4ed8"      # 飽和皇家藍 (有感地震)
BLUE_LIGHT = "#93c5fd"     # 冰藍 (微震)
AMBER_HERO = "#d97706"     # 琥珀金 (921與烈震)
CRIMSON_HERO = "#dc2626"   # 胭脂朱紅 (0403與M7+)

def draw_chart1_total_seismicity(df_stat):
    """繪製專題圖一：歷年總地震筆數與監測網演進 (解決遮擋與破圖問題)"""
    print("[1/2] 繪製專題圖一：臺灣歷年地震總筆數與監測網演進分析圖 (精緻重構版)...")
    years = df_stat['年度'].values
    totals = df_stat['總地震筆數'].values
    micros = df_stat['微震(M<3.0)'].values
    felt = totals - micros
    micro_ratios = (micros / totals) * 100.0

    fig = plt.figure(figsize=(19, 12), dpi=300, facecolor=BG_COLOR)
    
    # 標頭 (Conclusion Title + Subtitle)
    fig.text(0.06, 0.960, "全臺歷年地震總監測筆數與觀測網密化演進 (1994–2026)", 
             fontsize=19, fontweight='bold', color=INK_DARK)
    fig.text(0.06, 0.936, "873,145 筆儀器監測紀錄 · 微震 (M < 3.0) 與有感地震分層結構 · 2012 氣象署觀測網密化里程碑 · 32.58 年時序", 
             fontsize=10.5, color=MUTED_TXT)

    # 上下子圖比例配置 (上圖 70%，下圖 30%)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], top=0.91, bottom=0.08, left=0.06, right=0.95, hspace=0.34)

    # -------------------------------------------------------------
    # 1. 上子圖：年度地震總數堆疊柱狀圖
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(CARD_BG)
    for sp in ax1.spines.values():
        sp.set_color(GRID_LINE)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 設置背景觀測網時代色帶 (完全不遮擋柱狀圖)
    ax1.axvspan(1992.5, 2011.5, color='#f8fafc', alpha=0.6, zorder=0)
    ax1.axvspan(2011.5, 2027.5, color='#f0f9ff', alpha=0.5, zorder=0)

    # 時代分界垂直虛線
    ax1.axvline(2011.5, color='#3b82f6', linestyle='--', lw=1.6, zorder=2)

    # 時代導覽看板（放置在上方 y=56000~64000 的無數據空白區，杜絕任何遮擋）
    era1_text = "【前期基礎數位觀測期 (1994–2011)】\n- 全臺測站約 70 處 · 監測極限門檻 Mc ~ 2.2\n- 年均監測量約 1.8 萬次 · 規模 2.0 以下微震常未收錄"
    ax1.text(2002.0, 57000, era1_text, ha='center', va='bottom', fontsize=8.6, color='#334155',
             bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#cbd5e1", lw=1.1, alpha=0.95), zorder=5)

    era2_text = "【密集寬頻即時整合觀測期 (2012–2026)】\n- 氣象署 CWASN + 中研院 BATS 全面併網 (超過 150 測站)\n- 偵測完整度下探至 Mc ~ 1.8 · 微震捕捉能力翻倍 · 年均量常態突破 3.5 萬次"
    ax1.text(2019.5, 57000, era2_text, ha='center', va='bottom', fontsize=8.6, color='#1e3a8a',
             bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#93c5fd", lw=1.2, alpha=0.95), zorder=5)

    # 堆疊柱狀圖：微震 (淺冰藍) + 有感地震 (飽和深藍)
    ax1.bar(years, micros, width=0.60, color=BLUE_LIGHT, alpha=0.92, label='微震 (M < 3.0，人體無感微震)', zorder=3)
    ax1.bar(years, felt, bottom=micros, width=0.60, color=BLUE_MAIN, alpha=0.95, label='有感與顯著地震 (M >= 3.0)', zorder=3)

    # 特殊高亮年份：1999 (921 集集) 與 2024 (0403 花蓮)
    for y, tot in zip(years, totals):
        if y == 1999:
            ax1.bar(y, tot, width=0.60, color=AMBER_HERO, alpha=0.98, zorder=4)
            # 乾淨卡片，指針精確指向柱頂
            ax1.annotate("1999 921集集大地震餘震群\n總數：49,556 次 (單年最高紀錄)\n微震佔比：88.4%",
                         xy=(y, tot), xytext=(y - 2.5, 53000),
                         arrowprops=dict(arrowstyle="->", color=AMBER_HERO, lw=1.5),
                         fontsize=8.5, fontweight='bold', color='#78350f', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="#fffbeb", ec=AMBER_HERO, lw=1.2), zorder=6)
        elif y == 2024:
            ax1.bar(y, tot, width=0.60, color=CRIMSON_HERO, alpha=0.98, zorder=4)
            ax1.annotate("2024 花蓮0403強震餘震潮\n總數：34,408 次 (僅計至7月)\nM5+強震高達 144 次破紀錄",
                         xy=(y, tot), xytext=(y + 0.2, 42000),
                         arrowprops=dict(arrowstyle="->", color=CRIMSON_HERO, lw=1.5),
                         fontsize=8.5, fontweight='bold', color='#7f1d1d', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="#fef2f2", ec=CRIMSON_HERO, lw=1.2), zorder=6)

    # 僅在關鍵里程碑年份柱頂標註清晰數字（杜絕密集旋轉數字互相重疊）
    milestone_years = {1994: 17506, 1996: 16436, 1998: 14456, 2002: 27697, 2010: 22632, 
                       2012: 30543, 2014: 35799, 2016: 47976, 2018: 49706, 2022: 13292, 2026: 20304}
    for myr, mval in milestone_years.items():
        ax1.text(myr, mval + 700, f"{mval:,}", ha='center', va='bottom', fontsize=7.8, color='#475569')

    ax1.set_ylabel("年度地震監測總數 (次/年)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=8)
    ax1.set_xlim(1992.5, 2027.5)
    ax1.set_ylim(0, 66000)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) if (y % 2 == 0 or y in [1999, 2024]) else "" for y in years], fontsize=9, color=MUTED_TXT)
    ax1.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)
    ax1.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)

    # -------------------------------------------------------------
    # 2. 下子圖：微震佔比演化曲線 (%)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CARD_BG)
    for sp in ax2.spines.values():
        sp.set_color(GRID_LINE)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.plot(years, micro_ratios, color=BLUE_MAIN, lw=2.2, marker='o', markersize=4.5, zorder=3)
    ax2.fill_between(years, micro_ratios, 70, color='#e0f2fe', alpha=0.45, zorder=2)
    mean_ratio = np.mean(micro_ratios)
    ax2.axhline(mean_ratio, color='#64748b', linestyle=':', lw=1.4, label=f"32 年歷史平均微震佔比 ({mean_ratio:.1f}%)")

    # 標註微震佔比異常期
    ax2.text(2015.5, 98.2, "2013–2019 高微震監測期 (微震比例常態 >96%)", ha='center', fontsize=8.8, color='#0284c7', fontweight='bold')
    ax2.annotate("2022 池上連震帶動顯著有感震增加", xy=(2022, micro_ratios[years==2022][0]), xytext=(2020.2, 80),
                 arrowprops=dict(arrowstyle="->", color='#0284c7', lw=1.1),
                 fontsize=8.0, color='#0f172a', bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#bae6fd", lw=0.8))

    ax2.set_ylabel("微震佔比 (%)", fontsize=10.5, fontweight='bold', color=INK_DARK, labelpad=8)
    ax2.set_xlabel("觀測年度 (西元)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=8)
    ax2.set_xlim(1992.5, 2027.5)
    ax2.set_ylim(70, 102)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, fontsize=8.5, color=MUTED_TXT, rotation=45)
    ax2.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)
    ax2.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.5)

    # 底部底注 (保留充裕留白)
    fig.text(0.06, 0.016, 
             "DATA SOURCE: 交通部中央氣象署 (CWA) 地球物理資料管理系統 · 涵蓋 1994/01/01 至 2026/07/30 共 873,145 筆 · 視覺規範遵循 LIEFLAT-CHARTS MONO/PORCELAIN 標準",
             fontsize=8.5, color=FAINT_TXT)

    plt.savefig(OUT_PNG1, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  -> 專題圖一已輸出: {OUT_PNG1}")

def draw_chart2_damaging_earthquakes(df_stat, df_cat):
    """繪製專題圖二：重大強震與能量釋放歷史紀事 (上下雙軌同步聯動，杜絕箭頭歧義)"""
    print("[2/2] 繪製專題圖二：臺灣歷年重大強震與能量釋放歷史紀事圖 (上下雙軌聯動架構)...")
    years = df_stat['年度'].values
    m5_counts = df_stat['強震(5.0<=M<6.0)'].values
    m6_counts = df_stat['烈震(6.0<=M<7.0)'].values
    m7_counts = df_stat['大震(M>=7.0)'].values

    # 計算累積能量 (焦耳)
    df_cat['year'] = pd.to_datetime(df_cat['datetime']).dt.year
    df_cat['energy'] = 10.0 ** (4.8 + 1.5 * df_cat['ML'])
    yearly_energy = df_cat.groupby('year')['energy'].sum().reindex(years).fillna(0).values / 1e15 # 10^15 J

    fig = plt.figure(figsize=(20, 13.5), dpi=300, facecolor=BG_COLOR)
    
    # 標頭 (Conclusion Title + Subtitle)
    fig.text(0.05, 0.965, "臺灣歷年顯著強震、破壞性烈震與能量釋放歷史全景紀事 (1994–2026)", 
             fontsize=19, fontweight='bold', color=INK_DARK)
    fig.text(0.05, 0.942, "上軌：12 大歷史標竿震災精準時序里程碑 · 下軌：歷年中強震頻率與累積能量釋放曲線 · 雙軌時序完全同步", 
             fontsize=10.5, color=MUTED_TXT)

    # 劃分上下雙軌：上軌里程碑 36%，下軌量化數據 64%，適度緊湊間距
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.2], top=0.92, bottom=0.07, left=0.05, right=0.94, hspace=0.18)

    # =========================================================================
    # 上軌：【12 大歷史標竿地震時空大事紀時間軸 (Milestone Ribbon)】
    # =========================================================================
    ax_time = fig.add_subplot(gs[0])
    ax_time.set_facecolor(CARD_BG)
    for sp in ax_time.spines.values():
        sp.set_visible(False)
    ax_time.set_xticks([])
    ax_time.set_yticks([])
    ax_time.set_xlim(1992.5, 2027.5)
    ax_time.set_ylim(-2.4, 2.5)

    # 中央時間基線
    ax_time.axhline(0, color='#94a3b8', lw=2.2, zorder=2)
    ax_time.text(1992.8, 0, "歷史時序軸", va='center', ha='right', fontsize=9.5, fontweight='bold', color='#64748b')

    # 12 大地震事件清單 (精確小數年份、標題、規模/深度、影響、上下排布 row: +1=上排, -1=下排)
    events = [
        {"x": 1994.40, "row": 1, "title": "1994/05/24 花蓮外海", "sub": "M 6.6 · 深度 5.3km", "desc": "深度極淺，全臺強烈搖晃", "m": 6.6},
        {"x": 1996.68, "row": -1, "title": "1996/09/05 蘭嶼海域", "sub": "M 7.1 · 深度 14.8km", "desc": "東部隱沒帶歷史深層大震", "m": 7.1},
        {"x": 1998.54, "row": 1, "title": "1998/07/17 嘉義瑞里", "sub": "M 6.2 · 深度 2.8km", "desc": "梅山斷層帶延伸，傷亡5人", "m": 6.2},
        {"x": 1999.72, "row": -1, "title": "1999/09/21 南投集集 (921)", "sub": "M 7.3 · 深度 8.0km", "desc": "車籠埔斷層破裂100km · 2,415死", "m": 7.3, "hero": True},
        {"x": 2002.25, "row": 1, "title": "2002/03/31 花蓮外海 (331)", "sub": "M 6.8 · 深度 13.8km", "desc": "臺北101起重機震落，造成5死", "m": 6.8},
        {"x": 2003.94, "row": -1, "title": "2003/12/10 臺東成功", "sub": "M 6.5 · 深度 17.7km", "desc": "池上斷層同震潛變蠕變觸發", "m": 6.5},
        {"x": 2006.98, "row": 1, "title": "2006/12/26 恆春海域雙震", "sub": "M 7.0 · 深度 44.1km", "desc": "震毀國際海纜，癱瘓東亞跨國通訊", "m": 7.0},
        {"x": 2010.17, "row": -1, "title": "2010/03/04 高雄甲仙", "sub": "M 6.4 · 深度 22.6km", "desc": "潮州斷層構造，高鐵首次出軌", "m": 6.4},
        {"x": 2016.10, "row": 1, "title": "2016/02/06 高雄美濃", "sub": "M 6.6 · 深度 14.6km", "desc": "場址效應，臺南維冠倒塌117死", "m": 6.6},
        {"x": 2018.10, "row": -1, "title": "2018/02/06 花蓮米崙", "sub": "M 6.3 · 深度 6.3km", "desc": "米崙斷層破裂，統帥飯店倒塌17死", "m": 6.3},
        {"x": 2022.71, "row": 1, "title": "2022/09/18 臺東池上", "sub": "M 6.8 · 深度 7.0km", "desc": "中央山脈斷層系統 · 高寮大橋斷裂", "m": 6.8},
        {"x": 2024.26, "row": -1, "title": "2024/04/03 花蓮和平外海", "sub": "M 7.2 · 深度 15.5km", "desc": "破裂逾70km · 強震144次破紀錄", "m": 7.2, "hero": True},
    ]

    for ev in events:
        x = ev["x"]
        row = ev["row"]
        is_hero = ev.get("hero", False)
        
        # 標記在時間基線上的圓點指針
        dot_color = CRIMSON_HERO if ev["m"] >= 7.0 else AMBER_HERO
        dot_size = 75 if ev["m"] >= 7.0 else 45
        ax_time.scatter(x, 0, color=dot_color, s=dot_size, zorder=5, edgecolor='#ffffff', lw=1.2)
        
        # 垂直連接線 (指針)
        target_y = 1.15 if row == 1 else -1.15
        ax_time.plot([x, x], [0, target_y], color=dot_color if is_hero else '#94a3b8', lw=1.3, linestyle='-', zorder=3)

        # 卡片框背景色與邊框
        if is_hero:
            box_bg = "#fef2f2" if ev["m"] >= 7.2 else "#fffbeb"
            box_ec = CRIMSON_HERO if ev["m"] >= 7.2 else AMBER_HERO
            box_lw = 1.4
            title_color = "#991b1b" if ev["m"] >= 7.2 else "#92400e"
        else:
            box_bg = "#ffffff"
            box_ec = "#cbd5e1"
            box_lw = 1.0
            title_color = "#0f172a"

        # 卡片文字內容
        card_text = f"{ev['title']}\n{ev['sub']}\n{ev['desc']}"
        va_align = 'bottom' if row == 1 else 'top'
        
        ax_time.text(x, target_y, card_text, ha='center', va=va_align, fontsize=8.0,
                     color=title_color, linespacing=1.25,
                     bbox=dict(boxstyle="round,pad=0.38", fc=box_bg, ec=box_ec, lw=box_lw, alpha=0.98),
                     zorder=6)

    # =========================================================================
    # 下軌：【歷年強震頻率與累積能量釋放量化對照】
    # =========================================================================
    ax_quant = fig.add_subplot(gs[1])
    ax_quant.set_facecolor(CARD_BG)
    for sp in ax_quant.spines.values():
        sp.set_color(GRID_LINE)
    ax_quant.spines['top'].set_visible(False)

    # 垂直導引光帶（將 1999 與 2024 由上軌直通下軌，視覺聯動極其明確）
    ax_quant.axvspan(1998.7, 2000.3, color='#fffbeb', alpha=0.6, zorder=0)
    ax_quant.axvspan(2023.7, 2024.9, color='#fef2f2', alpha=0.6, zorder=0)

    # 右軸：年度累積能量釋放階梯/曲線 (10^15 J)
    ax_energy = ax_quant.twinx()
    ax_energy.spines['top'].set_visible(False)
    ax_energy.spines['left'].set_visible(False)
    ax_energy.spines['bottom'].set_visible(False)
    ax_energy.spines['right'].set_color('#94a3b8')

    ax_energy.plot(years, yearly_energy, color='#64748b', lw=2.0, linestyle='-', marker='s', markersize=4, 
                   label='年度累積釋放能量 (10^15 焦耳)', zorder=2)
    ax_energy.fill_between(years, 0, yearly_energy, color='#f1f5f9', alpha=0.55, zorder=1)
    ax_energy.set_ylabel("年度地震釋放總能量 (10^15 焦耳)", fontsize=10.5, color='#475569', labelpad=10)
    ax_energy.tick_params(axis='y', labelcolor='#475569', labelsize=9)
    ax_energy.set_ylim(0, max(yearly_energy) * 1.30)

    # 左軸：並排柱狀圖 (Grouped Bar Chart)
    width = 0.36
    b1 = ax_quant.bar(years - width/2, m5_counts, width=width, color='#60a5fa', alpha=0.9, 
                      label='中強震 (5.0 <= M < 6.0)', zorder=3)
    b2 = ax_quant.bar(years + width/2, m6_counts, width=width, color='#f59e0b', alpha=0.95, 
                      label='烈震 (6.0 <= M < 7.0)', zorder=3)

    # 在 M>=7.0 年份頂端標註大紅星標與規模數字 (帶白底徽章防遮擋)
    m7_indices = np.where(m7_counts > 0)[0]
    m7_labels = {1996: "M7.1", 1999: "M7.3", 2024: "M7.2", 2025: "M7.0"}
    for idx in m7_indices:
        yr = years[idx]
        max_bar = max(m5_counts[idx], m6_counts[idx])
        y_star = max_bar + 8
        y_text = max_bar + 14
        if yr == 1999:
            y_star = 92
            y_text = 100
        ax_quant.scatter(yr, y_star, color=CRIMSON_HERO, s=160, marker='*', zorder=6, edgecolor='#ffffff', lw=0.9)
        lbl = m7_labels.get(yr, "M>=7.0")
        ax_quant.text(yr, y_text, lbl, ha='center', va='bottom', fontsize=8.0, fontweight='bold', color=CRIMSON_HERO,
                      bbox=dict(boxstyle="round,pad=0.22", fc="#ffffff", ec=CRIMSON_HERO, lw=0.8, alpha=0.95), zorder=7)

    # 在 1999 (74次中強震/14次烈震) 與 2024 (144次中強震) 柱頂標註明確數字
    ax_quant.text(1999 - width/2, m5_counts[years==1999][0] + 2, "74", ha='center', fontsize=8, color='#1e40af', fontweight='bold')
    ax_quant.text(1999 + width/2, m6_counts[years==1999][0] + 2, "14", ha='center', fontsize=8, color='#b45309', fontweight='bold')
    ax_quant.text(2024 - width/2, m5_counts[years==2024][0] + 2, "144 次", ha='center', fontsize=8.5, color='#1e40af', fontweight='bold')

    ax_quant.set_ylabel("顯著強震與烈震發生次數 (次/年)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=8)
    ax_quant.set_xlabel("觀測年度 (西元)", fontsize=11, fontweight='bold', color=INK_DARK, labelpad=10)
    ax_quant.set_xlim(1992.5, 2027.5)
    ax_quant.set_ylim(0, 168)
    ax_quant.set_xticks(years)
    ax_quant.set_xticklabels(years, fontsize=9, color=MUTED_TXT, rotation=45)
    ax_quant.grid(axis='y', color=GRID_LINE, linestyle='-', linewidth=0.8, zorder=1)

    # 組合圖例
    handles1, labels1 = ax_quant.get_legend_handles_labels()
    handles2, labels2 = ax_energy.get_legend_handles_labels()
    # 增加歷史大震圖例項
    dummy_star = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=CRIMSON_HERO, markersize=12, label='歷史大震 (M >= 7.0)')
    all_handles = [handles1[0], handles1[1], dummy_star, handles2[0]]
    all_labels = [labels1[0], labels1[1], '歷史大震 (M >= 7.0)', labels2[0]]
    ax_quant.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, 
                    facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.2, ncol=4)

    # 底部底注
    fig.text(0.05, 0.016, 
             "DATA SOURCE: 中央氣象署 (CWA) 地球物理資料管理系統 · 能量公式 log10(E) = 4.8 + 1.5*M · 視覺規範遵循 LIEFLAT-CHARTS MONO/PORCELAIN 標準",
             fontsize=8.5, color=FAINT_TXT)

    plt.savefig(OUT_PNG2, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  -> 專題圖二已輸出: {OUT_PNG2}")

def main():
    print("="*75)
    print("【重構歷年地震分析兩大獨立專題圖】啟動")
    print("="*75)
    df_stat = pd.read_csv(CSV_PATH)
    df_cat = pd.read_csv(CATALOG_PATH)

    draw_chart1_total_seismicity(df_stat)
    draw_chart2_damaging_earthquakes(df_stat, df_cat)
    print("\n兩大獨立專題圖重構完成！\n")

if __name__ == '__main__':
    main()
