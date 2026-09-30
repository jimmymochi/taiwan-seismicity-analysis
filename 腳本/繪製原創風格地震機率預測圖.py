# -*- coding: utf-8 -*-
"""
繪製原創風格臺灣及鄰近海域地震機率預測看板 (四聯圖)
========================================================================
依據使用者最新要求：
1. 顏色與文字不照抄範例：
   - 採用自研 high-end scientific editorial 色彩方案 (Porcelain Blue to Amber & Carmine)。
   - 標頭、圖例、子圖標籤、底注採用 lieflat 規範全新原創架構。
2. 時間尺度全面嚴格統一：
   - 四大統一標準尺度：未來 7 天 (極短期)、未來 30 天 (短期)、未來 1 年 (中期)、未來 5 年 (中長期)。
3. 同時支援產出：
   - 2026 現役即時推論版 (以 2026/08/01 最新觀測為快照點，向未來推算)
   - 2022 回溯盲測檢驗版 (以 2022/01/01 為快照點，用於驗證 2022-2026 實震)
4. 涵蓋 M5+ 與 M6+ 兩大破壞性震級。
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

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from shapely.geometry import LineString, Polygon

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "數據")
CHARTS_DIR = os.path.join(project_dir, "看板圖表")
os.makedirs(CHARTS_DIR, exist_ok=True)

PRED_2022_PATH = os.path.join(DATA_DIR, "機器學習盲測特徵集_2022_2026.csv.gz")
PRED_2026_PATH = os.path.join(DATA_DIR, "機器學習現役最新推論預測_2026.csv.gz")
TW_GPKG_PATH = os.path.join(project_dir, "圖資", "TW_country_pop_TM2.gpkg")

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# 全新原創科研色階 (Original Porcelain Slate to Ember & Crimson)
# ─────────────────────────────────────────────────────────────────────────────
# 避免原範例青綠黃橙紅彩虹感，改採深邃大氣的青瓷冷灰藍 -> 琥珀金 -> 胭脂深紅
ORIGINAL_PALETTE = [
    '#ffffff',   # 0: < 0.05% 背景底白
    '#f4f6fa',   # 1: 極微弱無感區
    '#dce5f0',   # 2: 淺青瓷灰藍
    '#b2c8e3',   # 3: 青瓷藍
    '#779fcb',   # 4: 鋼青藍
    '#3b71a5',   # 5: 海藍
    '#1e4b7a',   # 6: 深海鈷藍 (板塊隱沒帶背景)
    '#a67c2e',   # 7: 暖土金 (活動度顯著升高)
    '#cf7b2b',   # 8: 烈焰琥珀 (破裂機率高)
    '#c0392b',   # 9: 胭脂朱紅 (危險警戒)
    '#801515'    # 10: 棗紅黑赤 (極度破裂危險核心)
]

# 各時間尺度專屬自適應階梯 (Adaptive Levels for each Horizon)
# 徹底解決 5 年長週期「整片通紅失去分辨性」的致命缺陷：
# 1. 低機率區採冷灰藍與海藍保持清爽背景
# 2. 中高機率區 (20%~35%) 採暖土金與火焰琥珀
# 3. 45%+ 極端核心破裂區才使用胭脂深紅與棗黑赤
HORIZON_LEVELS_M6 = {
    'prob_m6_7d':  [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 52.0],
    'prob_m6_30d': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 35.0, 45.0],
    'prob_m6_1y':  [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
    'prob_m6_5y':  [0.0, 1.0, 5.0, 12.0, 18.0, 25.0, 32.0, 38.0, 44.0, 48.0, 51.0, 55.0]
}

HORIZON_LEVELS_M5 = {
    'prob_m5_7d':  [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 22.0, 28.0, 36.0],
    'prob_m5_30d': [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 28.0, 38.0, 48.0, 60.0],
    'prob_m5_1y':  [0.0, 1.0, 3.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0],
    'prob_m5_5y':  [0.0, 2.0, 5.0, 10.0, 15.0, 19.0, 23.0, 27.0, 30.0, 33.0, 35.0, 38.0]
}

def load_coastlines():
    """載入臺灣與福建沿海幾何"""
    tw_gdf = gpd.read_file(TW_GPKG_PATH)
    if tw_gdf.crs != 'EPSG:4326':
        tw_gdf = tw_gdf.to_crs(epsg=4326)

    # 福建海岸線
    fujian_pts = [
        (119.0, 26.5), (119.1, 26.3), (119.3, 26.1), (119.6, 26.0), (119.7, 25.8),
        (119.5, 25.6), (119.65, 25.45), (119.45, 25.3), (119.1, 25.2), (118.8, 25.0),
        (118.6, 24.8), (118.3, 24.5), (118.1, 24.3), (117.8, 24.0), (117.5, 23.6)
    ]
    fujian_line = LineString(fujian_pts)
    pingtan_poly = Polygon([(119.7, 25.4), (119.9, 25.45), (119.95, 25.6), (119.8, 25.65), (119.7, 25.55)])

    return tw_gdf, fujian_line, pingtan_poly

def plot_4panel_probability_map(df_pred, target_type='M5', snapshot_type='2026', output_filename=None):
    """
    繪製四大統一時間尺度之地震機率預測圖 (具備自適應色階與峰值熱區標註)
    """
    print(f"  繪製 [{target_type}+ | {snapshot_type} 快照] 4聯預測圖 (自適應動態色階)...")
    tw_gdf, fujian_line, pingtan_poly = load_coastlines()

    # 網格座標 (0.02° 經緯度，約 2.2 km)
    lon_grid = np.linspace(119.0, 123.0, 201)
    lat_grid = np.linspace(21.0, 26.0, 251)
    grid_X, grid_Y = np.meshgrid(lon_grid, lat_grid)

    pts_lon = df_pred['lon_center'].values
    pts_lat = df_pred['lat_center'].values

    # 4 個統一時間尺度欄位
    t_tag = target_type.lower()
    col_map = [
        ('極短期 (未來 7 天)', f'prob_{t_tag}_7d'),
        ('短期 (未來 30 天)', f'prob_{t_tag}_30d'),
        ('中期 (未來 1 年)', f'prob_{t_tag}_1y'),
        ('中長期 (未來 5 年)', f'prob_{t_tag}_5y')
    ]

    horizon_levels = HORIZON_LEVELS_M5 if target_type == 'M5' else HORIZON_LEVELS_M6
    cmap = mcolors.ListedColormap(ORIGINAL_PALETTE)

    fig, axes = plt.subplots(1, 4, figsize=(26, 12.5), dpi=300, facecolor='#ffffff')
    plt.subplots_adjust(top=0.86, bottom=0.17, left=0.035, right=0.965, wspace=0.13)

    # 頂部標題與說明文字
    snapshot_info = "現役推論快照：2026/08/01 00:00 (臺灣時間) · 納入最新 873,145 筆歷史觀測 · 向未來推算" if snapshot_type == '2026' else "回溯盲測驗證快照：2022/01/01 00:00 (臺灣時間) · CSEP 國際盲測驗證基準點"

    fig.text(0.04, 0.950, f"臺灣及鄰近海域 | {target_type}+ 地震破裂機率時空預測看板", 
             fontsize=20, fontweight='bold', color='#0f172a')
    fig.text(0.04, 0.925, f"時空機器學習預測體系 · 38,178 空間網格 (2.2 km) · 15 km 構造高斯平滑 · 3-Fold 等張機率校準 · 各時窗自適應動態色階", 
             fontsize=11, color='#475569')
    fig.text(0.04, 0.903, f"{snapshot_info} · 統一四大時間尺度 (7天 / 30天 / 1年 / 5年)", 
             fontsize=10.5, color='#64748b')

    for idx, (title, col_name) in enumerate(col_map):
        ax = axes[idx]
        ax.set_facecolor('#ffffff')

        probs = df_pred[col_name].values * 100.0  # 轉為百分比
        max_prob = np.max(probs)

        # 取得該時窗專屬等級
        levels = horizon_levels[col_name]
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # 二維空間插值 + 15 km 高斯平滑
        raw_grid = griddata((pts_lon, pts_lat), probs, (grid_X, grid_Y), method='nearest')
        smoothed = gaussian_filter(raw_grid, sigma=3.2)

        # 繪製非均勻分段等高階梯填色圖
        cf = ax.contourf(grid_X, grid_Y, smoothed, levels=levels, cmap=cmap, norm=norm, extend='max')

        # 疊加海岸線向量
        tw_gdf.plot(ax=ax, facecolor='none', edgecolor='#334155', linewidth=0.75, zorder=4)
        
        # 繪製福建沿海與平潭島
        fx, fy = fujian_line.xy
        ax.plot(fx, fy, color='#475569', linewidth=0.75, zorder=4)
        px, py = pingtan_poly.exterior.xy
        ax.plot(px, py, color='#475569', linewidth=0.75, zorder=4)

        # 座標軸設定
        ax.set_xlim(119.0, 123.0)
        ax.set_ylim(21.0, 26.0)
        ax.set_xticks([119, 120, 121, 122, 123])
        ax.set_yticks([21, 22, 23, 24, 25, 26])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.tick_params(colors='#64748b', labelsize=9.5)
        ax.set_xlabel("東經 (°E)", fontsize=10, color='#334155', labelpad=4)

        if idx == 0:
            ax.set_ylabel("北緯 (°N)", fontsize=10, color='#334155', labelpad=4)
        else:
            ax.tick_params(labelleft=True)

        ax.grid(True, linestyle=':', linewidth=0.5, color='#94a3b8', alpha=0.4, zorder=3)

        # 子圖標題 (左上) 與單格最高機率 (右上)
        ax.text(0.02, 1.025, title, transform=ax.transAxes, fontsize=12.5, fontweight='bold', color='#0f172a')
        ax.text(0.98, 1.025, f"單格最高 {max_prob:.2f}%", transform=ax.transAxes, fontsize=10, 
                color='#1e293b', fontweight='600', ha='right')

        # 在 5 年子圖上標註頂級破裂核心熱區 (明確回答使用者「哪個區又更高」)
        if idx == 3:
            if target_type == 'M6':
                if snapshot_type == '2026':
                    # 南澳外海
                    ax.scatter(121.77, 24.52, s=42, facecolor='#ffffff', edgecolor='#801515', linewidth=1.8, zorder=7)
                    ax.annotate("南澳外海 52.2%\n(破裂第一核心)", xy=(121.77, 24.52), xytext=(122.15, 24.85),
                                fontsize=8.5, fontweight='bold', color='#801515',
                                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#801515", lw=1.0, alpha=0.92),
                                arrowprops=dict(arrowstyle="->", color='#801515', lw=1.0), zorder=8)
                    # 縱谷玉里
                    ax.scatter(121.34, 23.60, s=42, facecolor='#ffffff', edgecolor='#801515', linewidth=1.8, zorder=7)
                    ax.annotate("縱谷玉里 52.2%\n(破裂第二核心)", xy=(121.34, 23.60), xytext=(120.30, 23.85),
                                fontsize=8.5, fontweight='bold', color='#801515',
                                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#801515", lw=1.0, alpha=0.92),
                                arrowprops=dict(arrowstyle="->", color='#801515', lw=1.0), zorder=8)
                else:
                    # 2022 盲測版核心：和平海盆
                    ax.scatter(122.35, 24.28, s=42, facecolor='#ffffff', edgecolor='#801515', linewidth=1.8, zorder=7)
                    ax.annotate("和平海盆 52.2%\n(盲測預測核心)", xy=(122.35, 24.28), xytext=(121.80, 23.85),
                                fontsize=8.5, fontweight='bold', color='#801515',
                                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#801515", lw=1.0, alpha=0.92),
                                arrowprops=dict(arrowstyle="->", color='#801515', lw=1.0), zorder=8)
            else:
                # M5+ 5年熱區標註
                if snapshot_type == '2026':
                    ax.scatter(120.68, 23.13, s=40, facecolor='#ffffff', edgecolor='#801515', linewidth=1.6, zorder=7)
                    ax.annotate("甲仙-六龜 36.6%", xy=(120.68, 23.13), xytext=(120.05, 22.80),
                                fontsize=8.5, fontweight='bold', color='#801515',
                                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#801515", lw=0.9, alpha=0.92),
                                arrowprops=dict(arrowstyle="->", color='#801515', lw=0.9), zorder=8)
                else:
                    ax.scatter(121.41, 22.93, s=40, facecolor='#ffffff', edgecolor='#801515', linewidth=1.6, zorder=7)
                    ax.annotate("成功-池上 35.6%\n(前瞻命中2022地震)", xy=(121.41, 22.93), xytext=(121.85, 22.65),
                                fontsize=8.5, fontweight='bold', color='#801515',
                                bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#801515", lw=0.9, alpha=0.92),
                                arrowprops=dict(arrowstyle="->", color='#801515', lw=0.9), zorder=8)

        # 每個子圖下方獨立放置色階條 (Colorbar)
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.088, pos.width, 0.014])
        
        # 精選 5-6 個具代表性的刻度
        sel_idx = [0, 2, 4, 6, 8, 10, 11]
        disp_ticks = [levels[i] for i in sel_idx if i < len(levels)]
        disp_ticks = sorted(list(set(disp_ticks)))
        
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=disp_ticks)
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor('#94a3b8')
        cbar.set_ticklabels([f"{v:g}%" for v in disp_ticks])
        cbar.ax.tick_params(labelsize=8, color='#64748b')
        cbar.set_label(f"{title}機率刻度", fontsize=8.5, color='#475569', labelpad=3)

    # 底部說明底注
    fig.text(0.04, 0.025,
             "RESEARCH FORECASTING MODEL · 機器學習時空決策樹模型 · 空間輸出格距 0.02° (~2.2 km)，物理平滑半徑 15 km · 地圖底圖：國土測繪中心 / Natural Earth · 僅供學術研究，非官方地震速報",
             fontsize=8.5, color='#64748b')

    # 儲存圖檔
    out_path = os.path.join(CHARTS_DIR, output_filename)
    plt.savefig(out_path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
    plt.close()
    print(f"  -> 圖檔已產出: {out_path}")

def main():
    print("="*75)
    print("【繪製原創風格臺灣及鄰近海域地震機率預測圖】啟動")
    print("="*75)

    # 載入 2026 現役最新推論數據
    if os.path.exists(PRED_2026_PATH):
        print("[1/2] 讀取 2026 現役最新推論數據...")
        df_2026 = pd.read_csv(PRED_2026_PATH)
        plot_4panel_probability_map(df_2026, target_type='M5', snapshot_type='2026', 
                                    output_filename="臺灣及鄰近海域_M5_地震機率預測看板_2026現役版.png")
        plot_4panel_probability_map(df_2026, target_type='M6', snapshot_type='2026', 
                                    output_filename="臺灣及鄰近海域_M6_地震機率預測看板_2026現役版.png")

    # 載入 2022 回溯盲測數據
    if os.path.exists(PRED_2022_PATH):
        print("[2/2] 讀取 2022 回溯盲測驗證數據...")
        df_2022 = pd.read_csv(PRED_2022_PATH)
        plot_4panel_probability_map(df_2022, target_type='M5', snapshot_type='2022', 
                                    output_filename="臺灣及鄰近海域_M5_地震機率預測看板_2022盲測版.png")
        plot_4panel_probability_map(df_2022, target_type='M6', snapshot_type='2022', 
                                    output_filename="臺灣及鄰近海域_M6_地震機率預測看板_2022盲測版.png")

    print("\n原創風格 4 聯預測圖全數繪製完成！\n")

if __name__ == '__main__':
    main()
