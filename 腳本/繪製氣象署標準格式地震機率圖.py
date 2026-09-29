# -*- coding: utf-8 -*-
"""
臺灣地震時空預測管線 - 氣象署標準格式四聯/五聯地震機率圖繪製器
========================================================================
依據使用者提供之「臺灣及鄰近海域 | M5+ 地震機率・研究原型」參考格式：
1. 寬版現代資訊圖表排版 (Data Journalism / Scientific Report Layout)
2. 頂部深藍主標題 + 灰色雙行資料快照與校準狀態描述
3. 橫向 4 個並列子圖（未來 7 天、30 天、1 年、5 年），右上角標註單格最高機率百分比
4. 經緯度座標標籤 (119°E-123°E, 21°N-26°N)，細緻灰黑海岸線與福建沿岸
5. 15 km 高斯平滑曲面 + 非均勻機率離散色階等高面填充
6. 底部置中非等寬分段機率色條與官方研究原型免責宣告
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
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Polygon

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "數據")
MAP_DIR = os.path.join(project_dir, "圖資")
OUTPUT_DIR = os.path.join(project_dir, "看板圖表")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CSV = os.path.join(DATA_DIR, "機器學習盲測特徵集_2022_2026.csv.gz")
BOUNDARY_GPKG = os.path.join(MAP_DIR, "TW_country_pop_TM2.gpkg")

# 提取參考圖中的精確 10 階色彩 (從淺灰冰藍至磚橘、深紅)
COLOR_PALETTE = [
    '#ffffff',   # 0 ~ 0.05% (幾乎無色/背景)
    '#f2f5f8',   # 0.05% ~ 0.1% (極淺冰藍)
    '#ddeaf4',   # 0.1% ~ 0.2% (柔和天藍)
    '#c4dfec',   # 0.2% ~ 0.5% (粉彩蔚藍)
    '#9aceda',   # 0.5% ~ 1.0% (淺湖水綠)
    '#69b7c1',   # 1.0% ~ 2.0% (中度青綠)
    '#36a39e',   # 2.0% ~ 5.0% (深海綠)
    '#74ba75',   # 5.0% ~ 10.0% (草綠/春綠)
    '#d4cd55',   # 10.0% ~ 18.0% (暖黃芥末)
    '#efad46',   # 18.0% ~ 25.0% (溫暖杏橘)
    '#d76642',   # 25.0% ~ 35.0% (磚紅/珊瑚橘紅)
    '#b83e2e'    # > 35.0% (深赭紅)
]

LEVELS_M5 = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 25.0, 35.0, 45.0]
LEVELS_M6 = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]

def load_coastline_and_data():
    print("[1/3] 載入盲測預測數據與海岸線幾何向量...")
    df = pd.read_csv(TEST_CSV)
    
    # 臺灣行政與本島海岸線向量 (轉為 WGS84)
    tw = gpd.read_file(BOUNDARY_GPKG).to_crs(epsg=4326)
    tw_union = tw.geometry.union_all()

    # 福建沿海與平潭島簡化外廓 (精準符合參考圖左上方陸地幾何)
    fujian_line = LineString([
        (119.0, 26.0), (119.08, 25.95), (119.22, 25.92), (119.45, 25.86), 
        (119.65, 25.72), (119.78, 25.50), (119.70, 25.35), (119.52, 25.26), 
        (119.35, 25.18), (119.12, 25.08), (119.0, 25.02)
    ])
    pingtan_poly = Polygon([
        (119.72, 25.68), (119.86, 25.70), (119.92, 25.55), 
        (119.80, 25.42), (119.70, 25.48)
    ])

    return df, tw_union, fujian_line, pingtan_poly

def interpolate_and_smooth(points, values, lon_grid, lat_grid, sigma=3.5):
    """
    將 2.2km 網格數據插值至 0.02° 正則網格並進行 15km 高斯平滑
    """
    grid_val = griddata(points, values, (lon_grid, lat_grid), method='linear', fill_value=0.0)
    grid_smooth = gaussian_filter(grid_val, sigma=sigma)
    return np.clip(grid_smooth, 0.0, None)

def render_reference_style_map(df, tw_union, fujian_line, pingtan_poly, target='m5', num_panels=4):
    target_str = "M5+" if target == 'm5' else "M6+"
    levels = LEVELS_M5 if target == 'm5' else LEVELS_M6
    cmap = ListedColormap(COLOR_PALETTE[:len(levels)-1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if num_panels == 4:
        horizons = ['7d', '30d', '1y', '5y']
        horizon_titles = ['未來 7 天', '未來 30 天', '未來 365 天', '未來 5 年']
        out_name = f"臺灣及鄰近海域_{target.upper()}_地震機率研究模型_4聯圖.png"
        figsize = (28, 13)
    else:
        horizons = ['7d', '30d', '1y', '3y', '5y']
        horizon_titles = ['未來 7 天', '未來 30 天', '未來 1 年', '未來 3 年', '未來 5 年']
        out_name = f"臺灣及鄰近海域_{target.upper()}_地震機率研究模型_5聯圖.png"
        figsize = (32, 13)

    out_path_sub = os.path.join(OUTPUT_DIR, out_name)
    out_path_root = os.path.join(project_dir, out_name)

    print(f"[2/3] 繪製 {target_str} {num_panels}聯版參考風格圖: {out_name} ...")

    # 建立畫布
    fig = plt.figure(figsize=figsize, dpi=300, facecolor='#ffffff')
    
    # 頂部文字空間留白
    top_margin = 0.82
    bottom_margin = 0.16
    left_margin = 0.04
    right_margin = 0.96

    # 頂部標題區域 (完全還原參考圖層次與文字佈局)
    fig.text(left_margin, 0.94, f"臺灣及鄰近海域 | {target_str} 地震機率・機器學習時空模型", 
             fontsize=24, fontweight='bold', color='#1e293b', ha='left', va='top')
    
    fig.text(left_margin, 0.895, 
             "資料快照：2022/01/01 00:00（臺灣時間） | 0.02° × 0.02°（約 2.2 km）· 50,000 格 | 所有深度 | 38,178 空間網格", 
             fontsize=12, color='#475569', ha='left', va='top')
    
    fig.text(left_margin, 0.865, 
             f"歷史目錄 873,145 次地震；完整性門檻 Mc=1.8。已實施 3-Fold 等張機率校準 (Isotonic Calibration)。各圖共用分段機率色階。", 
             fontsize=11.5, color='#64748b', ha='left', va='top')

    # 子圖網格
    gs = fig.add_gridspec(1, num_panels, left=left_margin, right=right_margin, 
                          bottom=bottom_margin, top=top_margin, wspace=0.14)

    lon_lin = np.linspace(119.0, 123.0, 201)
    lat_lin = np.linspace(21.0, 26.0, 251)
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
    points = df[['lon_center', 'lat_center']].values

    for idx, (h, h_title) in enumerate(zip(horizons, horizon_titles)):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#ffffff')

        col_name = f"prob_{target}_{h}"
        raw_vals = df[col_name].values * 100.0  # 轉為百分比
        max_val = raw_vals.max()

        # 0.02° 正則插值與 15km 高斯平滑
        smoothed_field = interpolate_and_smooth(points, raw_vals, lon_grid=grid_lon, lat_grid=grid_lat, sigma=3.2)

        # 1. 填色等高面 (contourf)
        cf = ax.contourf(grid_lon, grid_lat, smoothed_field, levels=levels, cmap=cmap, norm=norm, extend='max', antialiased=True)
        
        # 2. 微細等高線 (contour lines)，增添柔和地圖層次感
        ax.contour(grid_lon, grid_lat, smoothed_field, levels=levels[1:], colors='#94a3b8', linewidths=0.35, alpha=0.45)

        # 3. 繪製臺灣與鄰近海岸線
        gpd.GeoSeries([tw_union]).boundary.plot(ax=ax, color='#334155', linewidth=0.85, zorder=4)
        gpd.GeoSeries([fujian_line]).plot(ax=ax, color='#334155', linewidth=0.85, zorder=4)
        gpd.GeoSeries([pingtan_poly]).boundary.plot(ax=ax, color='#334155', linewidth=0.85, zorder=4)

        # 座標軸刻度與外觀
        ax.set_xlim(119.0, 123.0)
        ax.set_ylim(21.0, 26.0)
        ax.set_xticks([119, 120, 121, 122, 123])
        ax.set_yticks([21, 22, 23, 24, 25, 26])
        ax.set_xlabel("東經（°）", fontsize=11, color='#1e293b', labelpad=6)
        
        if idx == 0:
            ax.set_ylabel("北緯（°）", fontsize=11, color='#1e293b', labelpad=6)
            ax.tick_params(labelsize=10, colors='#334155')
        else:
            ax.set_ylabel("")
            ax.tick_params(labelsize=10, colors='#334155')

        ax.grid(True, linestyle=':', color='#cbd5e1', linewidth=0.6, alpha=0.55)
        
        # 子圖標題 (左上) 與 單格最高數值標籤 (右上)
        ax.set_title(h_title, loc='left', fontsize=14, fontweight='bold', color='#1e293b', pad=8)
        ax.set_title(f"單格最高 {max_val:.3f}%", loc='right', fontsize=10.5, color='#64748b', pad=8)

        # 外框顏色
        for spine in ax.spines.values():
            spine.set_color('#94a3b8')
            spine.set_linewidth(0.8)

    # 底部色條 (置中、分段、各階寬度不等)
    cbar_ax = fig.add_axes([0.28, 0.08, 0.44, 0.022])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal',
                      boundaries=levels, spacing='uniform')
    
    # 格式化色標標籤刻度
    def fmt_tick(v):
        if v == 0:
            return "0"
        elif v < 1:
            return f"{v:g}"
        else:
            return f"{int(v)}" if v.is_integer() else f"{v:g}"

    cb.set_ticks(levels)
    cb.set_ticklabels([fmt_tick(v) for v in levels], fontsize=10, color='#334155')
    cb.outline.set_edgecolor('#94a3b8')
    cb.outline.set_linewidth(0.8)

    cbar_ax.set_title(f"單格至少一次 {target_str} 以上的模型機率（%）；各色階寬度不等", 
                      fontsize=11.5, color='#334155', pad=7)

    # 底部註解免責文字
    fig.text(left_margin, 0.025, 
             "離線研究原型，非官方預報；0.02° 是輸出格距，平滑尺度為 15 km。資料：歷年目錄＋中央氣象署＋經濟部地質調查所；底圖：Natural Earth / 內政部國土測繪圖資。", 
             fontsize=10, color='#64748b', ha='left', va='bottom')

    # 儲存高解析度圖檔 (同時存於 看板圖表/ 與 專案根目錄)
    plt.savefig(out_path_sub, dpi=300, facecolor='#ffffff')
    plt.savefig(out_path_root, dpi=300, facecolor='#ffffff')
    plt.close()
    print(f"  圖檔已產出: {out_path_sub} 及 {out_path_root}")

def main():
    df, tw_union, fujian_line, pingtan_poly = load_coastline_and_data()
    # 繪製 M5+ 4聯版 (完全對標參考圖)
    render_reference_style_map(df, tw_union, fujian_line, pingtan_poly, target='m5', num_panels=4)
    # 繪製 M5+ 5聯版 (涵蓋全部 5 大尺度)
    render_reference_style_map(df, tw_union, fujian_line, pingtan_poly, target='m5', num_panels=5)
    # 繪製 M6+ 4聯版 (強震版)
    render_reference_style_map(df, tw_union, fujian_line, pingtan_poly, target='m6', num_panels=4)
    print("[3/3] 全部參考風格圖版繪製全數圓滿完成！")

if __name__ == '__main__':
    main()
