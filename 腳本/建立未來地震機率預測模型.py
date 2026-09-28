# -*- coding: utf-8 -*-
"""
台灣各區域未來地震機率與規模預測模型
=====================================================
基於 1994–2026 年（32.6 年、873,145 筆紀錄）地震目錄，
結合 Gutenberg-Richter (G-R) 規模分佈與帕松隨機過程（Poisson Process），
量化推估台灣五大主要構造分區及空間網格在未來各時間跨度（1個月、1年、5年、10年、30年）
發生不同規模門檻（ML 5.0, 5.5, 6.0, 6.5, 7.0）之破裂發生機率與平均回歸週期。
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import geopandas as gpd
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
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "台灣各分區未來地震機率與規模預測模型.png")
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR, "數據")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_CSV_DIR, "台灣各區域地震危害度與未來機率預測表.csv")

def load_data():
    print("[1/5] 讀取全台地震歷史目錄 (17 個 CSV 檔案)...")
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
    print(f"資料時間涵蓋：{t_start.strftime('%Y-%m-%d')} 至 {t_end.strftime('%Y-%m-%d')}，總計 {duration_years:.2f} 年，有效筆數：{len(df):,}")
    return df, duration_years

def define_regions():
    # 定義五大主要構造區與全台範圍
    regions = {
        "全台灣整體 (Taiwan Overall)": {
            "lon": (119.2, 123.5),
            "lat": (21.5, 25.5),
            "color": "#1f77b4"
        },
        "東北部琉球隱沒帶 (Northeast Subduction)": {
            "lon": (121.5, 123.5),
            "lat": (23.8, 25.5),
            "color": "#d62728"
        },
        "花東縱谷與碰撞帶 (Hualien-Taitung Suture)": {
            "lon": (121.0, 121.8),
            "lat": (22.3, 23.8),
            "color": "#ff7f0e"
        },
        "嘉南前陸褶皺逆衝帶 (Chia-Nan Thrust Belt)": {
            "lon": (120.0, 120.8),
            "lat": (22.8, 23.8),
            "color": "#9467bd"
        },
        "中部車籠埔造山帶 (Central Chelungpu Zone)": {
            "lon": (120.4, 121.3),
            "lat": (23.8, 24.5),
            "color": "#2ca02c"
        },
        "大台北盆地構造區 (Taipei Basin & North)": {
            "lon": (121.1, 121.9),
            "lat": (24.8, 25.3),
            "color": "#8c564b"
        }
    }
    return regions

def calculate_probabilities(df, duration_years, regions):
    print("[2/5] 計算各區域不同規模與時間跨度之地震機率...")
    mag_thresholds = [5.0, 5.5, 6.0, 6.5, 7.0]
    time_horizons = {
        "未來1個月": 1.0 / 12.0,
        "未來1年": 1.0,
        "未來5年": 5.0,
        "未來10年": 10.0,
        "未來30年": 30.0
    }
    
    records = []
    
    for r_name, r_info in regions.items():
        min_lon, max_lon = r_info["lon"]
        min_lat, max_lat = r_info["lat"]
        
        rdf = df[
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon) &
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat)
        ]
        
        for m in mag_thresholds:
            n_events = (rdf['ML'] >= m).sum()
            # 年均發生率 lambda (events per year)
            lambda_rate = n_events / duration_years
            return_period = (1.0 / lambda_rate) if lambda_rate > 0 else np.nan
            
            row = {
                "區域": r_name,
                "規模門檻(>=ML)": m,
                "觀測期內次數": int(n_events),
                "年平均發生率(次/年)": round(lambda_rate, 4),
                "平均回歸週期(年)": round(return_period, 2) if not np.isnan(return_period) else None
            }
            
            for t_label, t_val in time_horizons.items():
                if lambda_rate > 0:
                    prob = 1.0 - math.exp(-lambda_rate * t_val)
                else:
                    prob = 0.0
                row[f"{t_label}機率(%)"] = round(prob * 100.0, 2)
            
            records.append(row)
            
    res_df = pd.DataFrame(records)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"機率預測表已儲存至：{OUTPUT_CSV}")
    return res_df

def compute_spatial_grid(df, duration_years, grid_step=0.2):
    print("[3/5] 計算空間網格化危害度機率 (0.2度)...")
    lon_bins = np.arange(119.5, 122.8, grid_step)
    lat_bins = np.arange(21.8, 25.5, grid_step)
    
    grid_data = []
    for lon in lon_bins:
        for lat in lat_bins:
            sub = df[
                (df['lon'] >= lon) & (df['lon'] < lon + grid_step) &
                (df['lat'] >= lat) & (df['lat'] < lat + grid_step)
            ]
            n_m5 = (sub['ML'] >= 5.0).sum()
            n_m6 = (sub['ML'] >= 6.0).sum()
            
            rate_m5 = n_m5 / duration_years
            rate_m6 = n_m6 / duration_years
            
            # 10年內 >= 5.0 機率
            p_10yr_m5 = (1.0 - math.exp(-rate_m5 * 10.0)) * 100.0 if rate_m5 > 0 else 0.0
            # 30年內 >= 6.0 機率
            p_30yr_m6 = (1.0 - math.exp(-rate_m6 * 30.0)) * 100.0 if rate_m6 > 0 else 0.0
            
            grid_data.append({
                "lon_center": lon + grid_step / 2.0,
                "lat_center": lat + grid_step / 2.0,
                "lon_min": lon,
                "lat_min": lat,
                "rate_m5": rate_m5,
                "p_10yr_m5": p_10yr_m5,
                "p_30yr_m6": p_30yr_m6
            })
            
    return pd.DataFrame(grid_data), lon_bins, lat_bins

def plot_forecast_figures(df, duration_years, regions, res_df, grid_df, lon_bins, lat_bins):
    print("[4/5] 繪製綜合預測模型圖表...")
    # 載入縣市疆界
    gdf = None
    if os.path.exists(gpkg_path):
        try:
            gdf = gpd.read_file(gpkg_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        except Exception as e:
            print(f"讀取圖資警訊：{e}")
            gdf = None

    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    
    # ----------------------------------------------------
    # 子圖 1: 空間機率分佈 - 未來 10 年內發生 ML >= 5.0 機率
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(A) 空間網格地震危害度：未來 10 年內發生 ML ≥ 5.0 地震之機率分佈", fontsize=13, fontweight='bold', pad=10)
    
    # 繪製縣市界線
    if gdf is not None:
        gdf.plot(ax=ax1, facecolor='#f0f0f0', edgecolor='#888888', linewidth=0.7, zorder=2, alpha=0.6)
    
    # 繪製網格機率散點/等高熱圖
    scatter1 = ax1.scatter(
        grid_df['lon_center'], grid_df['lat_center'],
        c=grid_df['p_10yr_m5'], s=140, marker='s',
        cmap='YlOrRd', vmin=0, vmax=100, alpha=0.85, zorder=3, edgecolors='none'
    )
    cb1 = fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("未來 10 年內 ML ≥ 5.0 破裂機率 (%)", fontsize=10)
    
    # 標註各構造分區邊框
    for r_name, r_info in list(regions.items())[1:]:
        min_lon, max_lon = r_info["lon"]
        min_lat, max_lat = r_info["lat"]
        rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                             fill=False, edgecolor=r_info["color"], linestyle='--', linewidth=1.5, zorder=4)
        ax1.add_patch(rect)
        ax1.text(min_lon + 0.05, max_lat - 0.12, r_name.split(' (')[0], 
                 fontsize=8, fontweight='bold', color=r_info["color"],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'), zorder=5)

    ax1.set_xlim(119.5, 122.8)
    ax1.set_ylim(21.8, 25.5)
    ax1.set_xlabel("經度 (Longitude °E)", fontsize=10)
    ax1.set_ylabel("緯度 (Latitude °N)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='gray')
    
    # ----------------------------------------------------
    # 子圖 2: 空間機率分佈 - 未來 30 年內發生 ML >= 6.0 強震機率
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(B) 災害型強震預警：未來 30 年內發生 ML ≥ 6.0 強震之機率分佈 (TEM標準週期)", fontsize=13, fontweight='bold', pad=10)
    
    if gdf is not None:
        gdf.plot(ax=ax2, facecolor='#f0f0f0', edgecolor='#888888', linewidth=0.7, zorder=2, alpha=0.6)
        
    scatter2 = ax2.scatter(
        grid_df['lon_center'], grid_df['lat_center'],
        c=grid_df['p_30yr_m6'], s=140, marker='s',
        cmap='magma_r', vmin=0, vmax=100, alpha=0.85, zorder=3, edgecolors='none'
    )
    cb2 = fig.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("未來 30 年內 ML ≥ 6.0 破裂機率 (%)", fontsize=10)

    # 標繪歷史 ML >= 6.0 地震震央
    m6_df = df[df['ML'] >= 6.0]
    ax2.scatter(m6_df['lon'], m6_df['lat'], s=m6_df['ML']**2 * 0.8,
                facecolors='none', edgecolors='blue', linewidths=0.8, alpha=0.5, zorder=4,
                label=f'歷史 ML≥6.0 事件 ({len(m6_df)}次)')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.85)

    ax2.set_xlim(119.5, 122.8)
    ax2.set_ylim(21.8, 25.5)
    ax2.set_xlabel("經度 (Longitude °E)", fontsize=10)
    ax2.set_ylabel("緯度 (Latitude °N)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='gray')

    # ----------------------------------------------------
    # 子圖 3: 各構造區年均地震發生率與平均回歸週期 (ML >= 5.0, 5.5, 6.0)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("(C) 各構造分區不同規模門檻之年均發生率 λ (次/年)", fontsize=13, fontweight='bold', pad=10)
    
    reg_names = [r.split(' (')[0] for r in regions.keys() if "全台" not in r]
    m_list = [5.0, 5.5, 6.0]
    bar_width = 0.25
    x = np.arange(len(reg_names))
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    for idx, m_val in enumerate(m_list):
        sub_res = res_df[res_df['規模門檻(>=ML)'] == m_val]
        rates = []
        for r_full in regions.keys():
            if "全台" in r_full:
                continue
            r_row = sub_res[sub_res['區域'] == r_full]
            rates.append(r_row['年平均發生率(次/年)'].values[0] if len(r_row) > 0 else 0)
            
        rects = ax3.bar(x + idx * bar_width, rates, width=bar_width, label=f'ML ≥ {m_val}', color=colors[idx], alpha=0.85)
        # 標註數值
        for r, val in zip(rects, rates):
            if val > 0.05:
                ax3.text(r.get_x() + r.get_width()/2., r.get_height() * 1.1, f'{val:.2f}',
                         ha='center', va='bottom', fontsize=8, rotation=0)

    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(reg_names, fontsize=9, rotation=15)
    ax3.set_ylabel("年平均發生率 λ (次/年, 對數刻度)", fontsize=10)
    ax3.set_yscale('log')
    ax3.set_ylim(0.01, 50)
    ax3.grid(True, which='both', linestyle=':', alpha=0.5)
    ax3.legend(title="規模門檻", fontsize=9)

    # ----------------------------------------------------
    # 子圖 4: 各構造區在不同時間跨度下發生 ML >= 6.0 強震之累積破裂機率
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("(D) 各區域未來發生 ML ≥ 6.0 災害型強震之累積機率曲線 P(Δt)", fontsize=13, fontweight='bold', pad=10)
    
    t_span = np.linspace(0.1, 30, 200) # 0.1 到 30 年
    sub_m6 = res_df[res_df['規模門檻(>=ML)'] == 6.0]
    
    for r_full, r_info in regions.items():
        r_row = sub_m6[sub_m6['區域'] == r_full]
        if len(r_row) > 0:
            lam = r_row['年平均發生率(次/年)'].values[0]
            short_name = r_full.split(' (')[0]
            if lam > 0:
                p_curve = (1.0 - np.exp(-lam * t_span)) * 100.0
                ls = '-' if "全台" in r_full or "東北" in r_full else ('--' if "花東" in r_full else '-.')
                lw = 2.5 if "全台" in r_full else 1.8
                ax4.plot(t_span, p_curve, label=f"{short_name} (λ={lam:.2f}/yr)",
                         color=r_info["color"], linestyle=ls, linewidth=lw)

    ax4.set_xlabel("未來預測時間跨度 Δt (年)", fontsize=10)
    ax4.set_ylabel("發生至少一次 ML ≥ 6.0 強震機率 (%)", fontsize=10)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 105)
    ax4.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(1.1, 15, "1年", fontsize=8, color='gray')
    ax4.axvline(10.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(10.1, 15, "10年", fontsize=8, color='gray')
    ax4.axvline(30.0, color='gray', linestyle=':', alpha=0.7)
    ax4.text(28.5, 15, "30年", fontsize=8, color='gray')
    
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    plt.suptitle("台灣各區域地震活動性分析與未來機率預測模型\n(基於中央氣象署 1994–2026 年目錄，G-R 定律與帕松極值過程)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[5/5] 圖表輸出完成：{OUTPUT_IMG}")

def main():
    df, duration_years = load_data()
    regions = define_regions()
    res_df = calculate_probabilities(df, duration_years, regions)
    grid_df, lon_bins, lat_bins = compute_spatial_grid(df, duration_years, grid_step=0.2)
    plot_forecast_figures(df, duration_years, regions, res_df, grid_df, lon_bins, lat_bins)
    print("模型計算與出圖全數順利完成！")

if __name__ == '__main__':
    main()
