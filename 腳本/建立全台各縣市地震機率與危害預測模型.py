# -*- coding: utf-8 -*-
"""
台灣各縣市在地化未來地震機率與危害度預測模型
=====================================================
針對「全台灣範圍過大、缺乏局部實用性」之核心痛點，
本腳本將中央氣象署 1994–2026 年（32.58 年、873,145 筆紀錄）地震目錄，
透過地理空間聯結（Spatial Join）精確對應至臺灣各縣市行政轄區及其感震生活圈（含 20km 緩衝區）。
量化計算各縣市在未來不同時間跨度（1年、5年、10年、30年）發生中強震（ML >= 5.0）
與災害型強震（ML >= 6.0）之破裂發生機率、平均回歸週期與極淺層震源佔比。
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

data_dirs = [os.path.join(project_dir, "地震目錄"), r"D:\JIMMY CHEN\達意專題\地震目錄"]
data_dir = next((d for d in data_dirs if os.path.exists(d)), data_dirs[0])

gpkg_paths = [
    os.path.join(project_dir, "圖資", "TW_country_pop_TM2.gpkg"),
    r"D:\JIMMY CHEN\地理資訊系統參考資料-20260707T054908Z-3-001\地理資訊系統參考資料\TW_country_pop_TM2.gpkg"
]
gpkg_path = next((p for p in gpkg_paths if os.path.exists(p)), gpkg_paths[0])

OUTPUT_DIR = project_dir
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "台灣各縣市未來地震機率與危害預測圖.png")
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR, "數據")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_CSV_DIR, "台灣各縣市未來地震機率與危害度分析表.csv")

def load_data_and_shapefile():
    print("[1/5] 讀取全台歷史地震目錄 (17 個 CSV 檔案)...")
    files = sorted(glob.glob(os.path.join(data_dir, "GDMS_*.csv")))
    dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df['ML'] = pd.to_numeric(df['ML'], errors='coerce')
    df = df.dropna(subset=['datetime', 'lat', 'lon', 'depth', 'ML']).sort_values('datetime').reset_index(drop=True)
    
    t_start = df['datetime'].min()
    t_end = df['datetime'].max()
    duration_years = (t_end - t_start).total_seconds() / (365.25 * 86400.0)
    print(f"資料涵蓋期：{t_start.strftime('%Y-%m-%d')} 至 {t_end.strftime('%Y-%m-%d')} ({duration_years:.2f} 年)，有效筆數：{len(df):,}")

    print("[2/5] 載入臺灣縣市圖資並建立 20km 感震生活圈緩衝區...")
    gdf_tw = gpd.read_file(gpkg_path)
    # 確保座標系為 TM2 (EPSG:3826) 進行米制緩衝區運算
    if gdf_tw.crs is None or gdf_tw.crs.to_epsg() != 3826:
        gdf_tw = gdf_tw.to_crs(epsg=3826)
    
    # 建立 20km (20,000 公尺) 緩衝區（涵蓋近海震源與跨界活斷層）
    gdf_buf_3826 = gdf_tw.copy()
    gdf_buf_3826['geometry'] = gdf_buf_3826.geometry.buffer(20000)
    
    # 轉回 WGS84 (EPSG:4326) 供與經緯度空間聯結
    gdf_tw_wgs84 = gdf_tw.to_crs(epsg=4326)
    gdf_buf_wgs84 = gdf_buf_3826.to_crs(epsg=4326)
    
    return df, duration_years, gdf_tw_wgs84, gdf_buf_wgs84

def calculate_county_statistics(df, duration_years, gdf_tw_wgs84, gdf_buf_wgs84):
    print("[3/5] 執行地理空間聯結 (Spatial Join) 計算各縣市地震活動與機率...")
    # 建立地震點位之 GeoDataFrame (過濾 ML >= 3.0 以提高效率)
    df_m3 = df[df['ML'] >= 3.0].copy()
    geometry = [Point(xy) for xy in zip(df_m3['lon'], df_m3['lat'])]
    gdf_pts = gpd.GeoDataFrame(df_m3, geometry=geometry, crs='EPSG:4326')
    
    # 空間聯結：感震生活圈 (20km 緩衝區)
    joined_buf = gpd.sjoin(gdf_pts, gdf_buf_wgs84[['C_Name', 'geometry']], how='inner', predicate='within')
    
    # 空間聯結：嚴格轄區陸域內
    joined_strict = gpd.sjoin(gdf_pts, gdf_tw_wgs84[['C_Name', 'geometry']], how='inner', predicate='within')

    # 時間跨度定義 (年)
    time_spans = {
        "未來1年": 1.0,
        "未來5年": 5.0,
        "未來10年": 10.0,
        "未來30年": 30.0
    }

    county_results = []
    
    # 針對每個縣市統計
    for county_name in gdf_tw_wgs84['C_Name'].unique():
        # 感震生活圈數據
        c_buf = joined_buf[joined_buf['C_Name'] == county_name]
        c_strict = joined_strict[joined_strict['C_Name'] == county_name]
        
        n_m3 = len(c_buf)
        n_m5 = (c_buf['ML'] >= 5.0).sum()
        n_m6 = (c_buf['ML'] >= 6.0).sum()
        
        # 轄區陸域內中強震數
        n_m5_strict = (c_strict['ML'] >= 5.0).sum() if len(c_strict) > 0 else 0
        n_m6_strict = (c_strict['ML'] >= 6.0).sum() if len(c_strict) > 0 else 0
        
        # 震源深度特徵 (針對 ML >= 4.0 計算中位數與極淺層 <15km 佔比)
        c_m4 = c_buf[c_buf['ML'] >= 4.0]
        if len(c_m4) > 0:
            median_depth = c_m4['depth'].median()
            shallow_ratio = ((c_m4['depth'] < 15.0).sum() / len(c_m4)) * 100.0
        else:
            median_depth = np.nan
            shallow_ratio = 0.0

        # 計算年均發生率與回歸週期 (以感震生活圈為準)
        lam_m5 = n_m5 / duration_years
        lam_m6 = n_m6 / duration_years
        
        ret_m5 = (1.0 / lam_m5) if lam_m5 > 0 else np.nan
        ret_m6 = (1.0 / lam_m6) if lam_m6 > 0 else np.nan
        
        res = {
            "縣市名稱": county_name,
            "生活圈累積地震數(ML>=3.0)": n_m3,
            "生活圈中強震數(ML>=5.0)": int(n_m5),
            "生活圈強震數(ML>=6.0)": int(n_m6),
            "轄區陸域內中強震數(ML>=5.0)": int(n_m5_strict),
            "轄區陸域內強震數(ML>=6.0)": int(n_m6_strict),
            "平均震源深度(km)": round(median_depth, 1) if not np.isnan(median_depth) else None,
            "極淺層震源比例(<15km, %)": round(shallow_ratio, 1),
            "年均發生率_M5(次/年)": round(lam_m5, 4),
            "平均回歸週期_M5(年)": round(ret_m5, 2) if not np.isnan(ret_m5) else None,
            "年均發生率_M6(次/年)": round(lam_m6, 4),
            "平均回歸週期_M6(年)": round(ret_m6, 2) if not np.isnan(ret_m6) else None,
        }
        
        # 破裂機率計算
        for label, t_val in time_spans.items():
            p5 = (1.0 - math.exp(-lam_m5 * t_val)) * 100.0 if lam_m5 > 0 else 0.0
            p6 = (1.0 - math.exp(-lam_m6 * t_val)) * 100.0 if lam_m6 > 0 else 0.0
            res[f"{label}_M5機率(%)"] = round(p5, 2)
            res[f"{label}_M6機率(%)"] = round(p6, 2)
            
        county_results.append(res)
        
    res_df = pd.DataFrame(county_results)
    # 依未來 10 年 ML>=5.0 機率降序排列
    res_df = res_df.sort_values(by="未來10年_M5機率(%)", ascending=False).reset_index(drop=True)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"各縣市危害度分析表已輸出至：{OUTPUT_CSV}")
    return res_df

def plot_county_hazard_maps(gdf_tw_wgs84, res_df, df):
    print("[4/5] 繪製各縣市在地化危害地圖與排行圖表...")
    # 將統計數據合併回 GeoDataFrame
    gdf_plot = gdf_tw_wgs84.merge(res_df, left_on='C_Name', right_on='縣市名稱', how='left')
    
    # 排除外島（金門、連江、澎湖）以凸顯本島各縣市局部對比，但外島仍保留在 CSV 中
    mainland_names = [
        '基隆市', '臺北市', '新北市', '桃園市', '新竹市', '新竹縣', '苗栗縣', 
        '臺中市', '彰化縣', '南投縣', '雲林縣', '嘉義市', '嘉義縣', 
        '臺南市', '高雄市', '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣'
    ]
    gdf_mainland = gdf_plot[gdf_plot['C_Name'].isin(mainland_names)].copy()
    df_mainland_res = res_df[res_df['縣市名稱'].isin(mainland_names)].copy()

    fig = plt.figure(figsize=(22, 18), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.2)
    
    # ----------------------------------------------------
    # 子圖 1 (左上): 各縣市未來 10 年內發生 ML >= 5.0 中強震機率地圖
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(A) 全台各縣市生活圈「未來 10 年內發生 ML ≥ 5.0 中強震」機率地圖", fontsize=13, fontweight='bold', pad=10)
    
    # 以機率著色
    gdf_mainland.plot(
        column='未來10年_M5機率(%)',
        ax=ax1,
        cmap='YlOrRd',
        edgecolor='#333333',
        linewidth=0.8,
        legend=True,
        legend_kwds={'label': "未來 10 年發生 ML ≥ 5.0 機率 (%)", 'orientation': "horizontal", 'shrink': 0.7, 'pad': 0.05},
        vmin=0, vmax=100
    )
    
    # 在每個縣市幾何中心標註縣市名稱與機率
    for idx, row in gdf_mainland.iterrows():
        rep_pt = row.geometry.representative_point()
        c_name = row['C_Name']
        p_val = row['未來10年_M5機率(%)']
        
        # 微調特定重疊縣市文字位置
        off_x, off_y = 0, 0
        if c_name == '臺北市': off_x, off_y = 0.08, 0.05
        elif c_name == '嘉義市': off_x, off_y = -0.15, -0.05
        elif c_name == '新竹市': off_x, off_y = -0.1, -0.05
        elif c_name == '基隆市': off_x, off_y = 0.12, 0.02
        
        ax1.annotate(
            f"{c_name}\n{p_val:.0f}%",
            xy=(rep_pt.x + off_x, rep_pt.y + off_y),
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='#111111' if p_val < 70 else '#ffffff',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black' if p_val >= 70 else 'white', alpha=0.5, edgecolor='none')
        )
        
    ax1.set_xlim(119.8, 122.3)
    ax1.set_ylim(21.8, 25.4)
    ax1.set_xlabel("經度 (°E)", fontsize=10)
    ax1.set_ylabel("緯度 (°N)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ----------------------------------------------------
    # 子圖 2 (右上): 各縣市未來 30 年內發生 ML >= 6.0 災害型強震機率地圖
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(B) 全台各縣市生活圈「未來 30 年內發生 ML ≥ 6.0 災害型強震」機率地圖", fontsize=13, fontweight='bold', pad=10)
    
    gdf_mainland.plot(
        column='未來30年_M6機率(%)',
        ax=ax2,
        cmap='magma_r',
        edgecolor='#333333',
        linewidth=0.8,
        legend=True,
        legend_kwds={'label': "未來 30 年發生 ML ≥ 6.0 機率 (%)", 'orientation': "horizontal", 'shrink': 0.7, 'pad': 0.05},
        vmin=0, vmax=100
    )
    
    # 標繪歷史 ML >= 6.0 震央
    m6_all = df[df['ML'] >= 6.0]
    ax2.scatter(m6_all['lon'], m6_all['lat'], s=m6_all['ML']**2 * 0.7,
                facecolors='none', edgecolors='blue', linewidths=0.9, alpha=0.6, zorder=5,
                label=f'歷史 ML≥6.0 震央 (共{len(m6_all)}次)')
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.85)

    for idx, row in gdf_mainland.iterrows():
        rep_pt = row.geometry.representative_point()
        c_name = row['C_Name']
        p_val = row['未來30年_M6機率(%)']
        off_x, off_y = 0, 0
        if c_name == '臺北市': off_x, off_y = 0.08, 0.05
        elif c_name == '嘉義市': off_x, off_y = -0.15, -0.05
        elif c_name == '新竹市': off_x, off_y = -0.1, -0.05
        elif c_name == '基隆市': off_x, off_y = 0.12, 0.02
        
        ax2.annotate(
            f"{c_name}\n{p_val:.0f}%",
            xy=(rep_pt.x + off_x, rep_pt.y + off_y),
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            color='#111111' if p_val < 60 else '#ffffff',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black' if p_val >= 60 else 'white', alpha=0.5, edgecolor='none')
        )

    ax2.set_xlim(119.8, 122.3)
    ax2.set_ylim(21.8, 25.4)
    ax2.set_xlabel("經度 (°E)", fontsize=10)
    ax2.set_ylabel("緯度 (°N)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # ----------------------------------------------------
    # 子圖 3 (左下): 全台 19 縣市「未來 10 年發生 ML >= 5.0 機率」降序排行
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("(C) 全台 19 縣市未來 10 年內發生 ML ≥ 5.0 中強震機率排行榜與平均回歸週期", fontsize=13, fontweight='bold', pad=10)
    
    sorted_df = df_mainland_res.sort_values(by="未來10年_M5機率(%)", ascending=True)
    y_pos = np.arange(len(sorted_df))
    probs = sorted_df['未來10年_M5機率(%)'].values
    names = sorted_df['縣市名稱'].values
    ret_periods = sorted_df['平均回歸週期_M5(年)'].values
    shallow_ratios = sorted_df['極淺層震源比例(<15km, %)'].values
    
    # 顏色區分風險等級
    bar_colors = []
    for p in probs:
        if p >= 90: bar_colors.append('#d62728')   # 極高危險
        elif p >= 60: bar_colors.append('#ff7f0e') # 高危險
        elif p >= 30: bar_colors.append('#2ca02c') # 中危險
        else: bar_colors.append('#1f77b4')         # 低危險
        
    bars = ax3.barh(y_pos, probs, color=bar_colors, alpha=0.85, height=0.65)
    
    for i, (p, ret, sh) in enumerate(zip(probs, ret_periods, shallow_ratios)):
        ret_txt = f"{ret:.1f}年一遇" if not np.isnan(ret) else "極罕見"
        sh_txt = f"極淺層{sh:.0f}%"
        ax3.text(p + 1.5, i, f"{p:.1f}% ({ret_txt}, {sh_txt})",
                 va='center', fontsize=8, color='#222222', fontweight='bold')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel("未來 10 年內發生機率 (%)", fontsize=10)
    ax3.set_xlim(0, 115)
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # 子圖 4 (右下): 焦點縣市未來 1~30 年累積強震機率演化曲線 (ML >= 6.0)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("(D) 焦點縣市未來 1 至 30 年發生 ML ≥ 6.0 災害型強震之累積機率曲線", fontsize=13, fontweight='bold', pad=10)
    
    t_span = np.linspace(0.1, 30, 200)
    focus_counties = [
        ('花蓮縣', '#d62728', '-', 2.5),
        ('臺東縣', '#e377c2', '-', 2.0),
        ('南投縣', '#2ca02c', '-', 2.0),
        ('宜蘭縣', '#ff7f0e', '-', 2.0),
        ('嘉義縣', '#9467bd', '--', 2.2),
        ('臺南市', '#8c564b', '--', 2.2),
        ('高雄市', '#bcbd22', '--', 1.8),
        ('臺中市', '#17becf', '-.', 1.8),
        ('新北市', '#7f7f7f', ':', 1.8),
        ('臺北市', '#1f77b4', ':', 1.8),
    ]
    
    for c_name, col, ls, lw in focus_counties:
        c_row = res_df[res_df['縣市名稱'] == c_name]
        if len(c_row) > 0:
            lam = c_row['年均發生率_M6(次/年)'].values[0]
            if lam > 0:
                p_curve = (1.0 - np.exp(-lam * t_span)) * 100.0
                ax4.plot(t_span, p_curve, label=f"{c_name} (λ={lam:.2f}/年, 回歸期{1/lam:.1f}年)",
                         color=col, linestyle=ls, linewidth=lw)
            else:
                ax4.plot(t_span, np.zeros_like(t_span), label=f"{c_name} (觀測期無M6震央)",
                         color=col, linestyle=ls, linewidth=lw, alpha=0.5)

    ax4.set_xlabel("未來時間跨度 Δt (年)", fontsize=10)
    ax4.set_ylabel("發生至少一次 ML ≥ 6.0 強震機率 (%)", fontsize=10)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(-2, 105)
    ax4.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(1.1, 15, "1年", fontsize=8, color='gray')
    ax4.axvline(10.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(10.1, 15, "10年", fontsize=8, color='gray')
    ax4.axvline(30.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(28.2, 15, "30年(TEM)", fontsize=8, color='gray')
    
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    plt.suptitle("台灣各縣市在地化地震危害度與未來破裂機率預測模型\n(基於中央氣象署 1994–2026 年目錄，縣市 20km 感震生活圈空間聯結與帕松極值過程)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[5/5] 圖表輸出完成：{OUTPUT_IMG}")

def main():
    df, duration_years, gdf_tw_wgs84, gdf_buf_wgs84 = load_data_and_shapefile()
    res_df = calculate_county_statistics(df, duration_years, gdf_tw_wgs84, gdf_buf_wgs84)
    plot_county_hazard_maps(gdf_tw_wgs84, res_df, df)
    print("各縣市地震危害度與機率預測全流程順利完成！")

if __name__ == '__main__':
    main()
