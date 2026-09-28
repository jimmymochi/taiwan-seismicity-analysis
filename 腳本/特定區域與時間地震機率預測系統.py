# -*- coding: utf-8 -*-
"""
臺灣特定時空區域地震機率預測核心引擎
=====================================================
本系統基於交通部中央氣象署 1994–2026 年（32.58 年、873,145 筆紀錄）歷史地震大數據，
提供對「任意指定空間區域（經緯度範圍或中心半徑）」與「任意未來時間跨度（天/月/年）」之
地震破裂發生機率、規模分佈（G-R定律）與預期發生次數之量化預測演算。

核心數學原理：
1. Gutenberg-Richter 規模頻率分佈：log10 N(>=M) = a - b*M
2. Aki (1965) 最大概似估計（MLE）計算本地構造 b 值
3. 帕松隨機點過程（Poisson Point Process）：
   - 預期事件數 mu = lambda(M) * Delta_t
   - 至少發生一次規模 >= M 地震機率 P(N>=1) = 1 - exp(-mu)
   - 恰好發生 k 次機率 P(N=k) = (mu^k * e^-mu) / k!
4. 近期活動度偏離度指數（Seismic Activity Anomaly Index）：
   - 評估該區域當前處於「平靜蓄能期」或「活躍釋放期」
"""

import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "地震目錄")
CATALOG_CSV = os.path.join(project_dir, "數據", "全台灣地震彙整目錄_1994_2026.csv")
OUTPUT_DIR = project_dir
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "台灣特定區域未來地震機率預測綜合圖.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "數據", "特定區域地震機率預測試算表.csv")

def load_catalog():
    if os.path.exists(CATALOG_CSV):
        print(f"[1/4] 從彙整 CSV 載入地震目錄: {CATALOG_CSV}")
        df = pd.read_csv(CATALOG_CSV)
        df['datetime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df['lat'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['lon'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['depth'] = pd.to_numeric(df['Depth'], errors='coerce')
        df['ML'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    else:
        print("[1/4] 讀取原始 GDMS 地震檔案目錄...")
        files = sorted(glob.glob(os.path.join(DATA_DIR, "GDMS_*.csv")))
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
    print(f"資料時間跨度：{t_start.strftime('%Y-%m-%d')} 至 {t_end.strftime('%Y-%m-%d')} ({duration_years:.2f} 年)，有效事件數：{len(df):,}")
    return df, duration_years

def haversine_distance(lon1, lat1, lon2, lat2):
    """計算球面大圓距離 (公里)"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def filter_spatial_region(df, lon_min=None, lon_max=None, lat_min=None, lat_max=None, center_lon=None, center_lat=None, radius_km=None):
    """
    根據使用者給定條件過濾特定空間區域
    - 方式一：經緯度矩形範圍 (Bounding Box)
    - 方式二：圓形半徑 (Center Coordinate + Radius in km)
    """
    if center_lon is not None and center_lat is not None and radius_km is not None:
        dist = haversine_distance(center_lon, center_lat, df['lon'].values, df['lat'].values)
        mask = dist <= radius_km
        region_desc = f"中心點 ({center_lon:.2f}°E, {center_lat:.2f}°N) 半徑 {radius_km:.0f} km 圓形區域"
    elif all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]):
        mask = (df['lon'] >= lon_min) & (df['lon'] <= lon_max) & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
        region_desc = f"經度 [{lon_min:.2f}, {lon_max:.2f}] °E，緯度 [{lat_min:.2f}, {lat_max:.2f}] °N 矩形區域"
    else:
        raise ValueError("必須提供完整的矩形經緯度範圍 (lon_min/max, lat_min/max) 或圓形中心半徑 (center_lon/lat, radius_km)")
        
    sub_df = df[mask].copy().reset_index(drop=True)
    return sub_df, region_desc

def fit_gutenberg_richter(mags, duration_years, mc=3.0, d_m=0.1):
    """
    計算本地 Gutenberg-Richter 參數 (a 值, b 值)
    採用 Aki (1965) 最大概似估計 MLE: b = log10(e) / (mean(M) - (Mc - dM/2))
    """
    valid_m = mags[mags >= mc]
    if len(valid_m) < 10:
        return np.nan, np.nan, np.nan, len(valid_m)
    
    mean_m = valid_m.mean()
    b_val = math.log10(math.e) / (mean_m - (mc - d_m / 2.0))
    # a 值歸一化至年發生率 (次/年)
    n_annual = len(valid_m) / duration_years
    a_val = math.log10(n_annual) + b_val * mc
    return a_val, b_val, mean_m, len(valid_m)

def calculate_time_probability(lambda_annual, delta_days):
    """
    計算在特定未來天數 delta_days 內，發生地震的帕松破裂機率與期望事件數
    """
    delta_t_year = delta_days / 365.25
    expected_events = lambda_annual * delta_t_year
    if expected_events > 0:
        prob_at_least_one = 1.0 - math.exp(-expected_events)
    else:
        prob_at_least_one = 0.0
    return prob_at_least_one, expected_events

def run_specific_region_prediction(df, duration_years, region_params, target_days_list, mag_targets=[4.0, 5.0, 5.5, 6.0, 6.5, 7.0]):
    """
    對指定特定區域，推估其在特定未來時間跨度下的發生機率
    """
    sub_df, region_desc = filter_spatial_region(df, **region_params)
    n_total = len(sub_df)
    
    # 計算 G-R 參數
    a_val, b_val, mean_m, n_mc = fit_gutenberg_richter(sub_df['ML'], duration_years, mc=3.0)
    
    # 評估近期活度偏離度 (以最近 365 天與長期年平均對比)
    latest_time = sub_df['datetime'].max() if len(sub_df) > 0 else df['datetime'].max()
    past_1yr = sub_df[sub_df['datetime'] >= (latest_time - pd.Timedelta(days=365))]
    n_recent_m4 = (past_1yr['ML'] >= 4.0).sum()
    n_longterm_m4_annual = (sub_df['ML'] >= 4.0).sum() / duration_years if duration_years > 0 else 0
    activity_ratio = (n_recent_m4 / n_longterm_m4_annual) if n_longterm_m4_annual > 0 else 1.0
    
    results = []
    for m in mag_targets:
        # 計算歷史實測年平均發生率 (次/年)
        n_obs = (sub_df['ML'] >= m).sum()
        lam_obs = n_obs / duration_years
        
        # 若觀測次數極少，可結合 G-R 理論值平滑外推
        if not np.isnan(a_val) and not np.isnan(b_val):
            lam_gr = 10.0 ** (a_val - b_val * m)
        else:
            lam_gr = lam_obs
            
        ret_period = (1.0 / lam_obs) if lam_obs > 0 else ((1.0 / lam_gr) if lam_gr > 0 else np.nan)
        
        row = {
            "區域描述": region_desc,
            "規模門檻(>=ML)": m,
            "32年歷史累積次數": int(n_obs),
            "年平均發生率(次/年)": round(lam_obs, 4),
            "理論G-R年率(次/年)": round(lam_gr, 4) if not np.isnan(lam_gr) else None,
            "平均回歸週期(年)": round(ret_period, 2) if not np.isnan(ret_period) else None,
        }
        
        for days in target_days_list:
            label = f"{days}天內" if days < 365 else f"{days//365}年內"
            prob, exp_n = calculate_time_probability(lam_obs, days)
            row[f"{label}破裂機率(%)"] = round(prob * 100.0, 2)
            row[f"{label}預期次數"] = round(exp_n, 3)
            
        results.append(row)
        
    return pd.DataFrame(results), sub_df, region_desc, a_val, b_val, activity_ratio

def plot_custom_predictions(preset_cases_results, output_path):
    """
    繪製特定區域與特定時間的預測結果對比綜合圖版 (4 子圖)
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300, facecolor='#fafafa')
    plt.subplots_adjust(hspace=0.28, wspace=0.22, top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    # ----------------------------------------------------
    # 子圖 1 (左上): 四大特定區域未來 30 天內發生各規模之短期破裂機率
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#ffffff')
    ax1.set_title("(A) 短期特定時間預測：未來 30 天內 (1個月) 發生不同規模地震之機率 P(Δt=30天)", 
                  fontsize=12, fontweight='bold', pad=10)
    
    mags = [4.0, 5.0, 5.5, 6.0, 6.5]
    x = np.arange(len(mags))
    bar_width = 0.2
    
    colors = ['#d62728', '#9467bd', '#2ca02c', '#1f77b4']
    for idx, (r_name, r_df) in enumerate(preset_cases_results.items()):
        sub = r_df[r_df['規模門檻(>=ML)'].isin(mags)]
        probs = sub['30天內破裂機率(%)'].values
        rects = ax1.bar(x + idx * bar_width, probs, width=bar_width, label=r_name, color=colors[idx], alpha=0.85)
        for r, p in zip(rects, probs):
            if p >= 5.0:
                ax1.text(r.get_x() + r.get_width()/2., r.get_height() + 1.0, f"{p:.0f}%",
                         ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                
    ax1.set_xticks(x + bar_width * 1.5)
    ax1.set_xticklabels([f"ML ≥ {m}" for m in mags], fontsize=9.5)
    ax1.set_ylabel("未來 30 天內破裂機率 (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 2 (右上): 四大特定區域未來 1 年內發生各規模之破裂機率
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#ffffff')
    ax2.set_title("(B) 中期特定時間預測：未來 1 年內 (365天) 發生不同規模地震之機率 P(Δt=1年)", 
                  fontsize=12, fontweight='bold', pad=10)
    
    for idx, (r_name, r_df) in enumerate(preset_cases_results.items()):
        sub = r_df[r_df['規模門檻(>=ML)'].isin(mags)]
        probs = sub['1年內破裂機率(%)'].values
        rects = ax2.bar(x + idx * bar_width, probs, width=bar_width, label=r_name, color=colors[idx], alpha=0.85)
        for r, p in zip(rects, probs):
            if p >= 5.0:
                ax2.text(r.get_x() + r.get_width()/2., r.get_height() + 1.0, f"{p:.0f}%",
                         ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax2.set_xticks(x + bar_width * 1.5)
    ax2.set_xticklabels([f"ML ≥ {m}" for m in mags], fontsize=9.5)
    ax2.set_ylabel("未來 1 年內破裂機率 (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 3 (左下): 嘉南梅山斷層區（特定局部區域）連續時間跨度機率演化曲線
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#ffffff')
    ax3.set_title("(C) 特定區域深度演化：嘉南梅山-中埔斷層區 (半徑25km) 隨未來時間之累積機率曲線", 
                  fontsize=12, fontweight='bold', pad=10)
    
    t_days = np.linspace(1, 365 * 15, 300) # 1 天至 15 年
    t_years = t_days / 365.25
    
    chiayi_df = preset_cases_results['嘉南梅山-中埔盲斷層區']
    target_m_list = [4.5, 5.0, 5.5, 6.0, 6.5]
    curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, m_val in enumerate(target_m_list):
        row = chiayi_df[chiayi_df['規模門檻(>=ML)'] == m_val]
        if len(row) > 0:
            lam = row['年平均發生率(次/年)'].values[0]
            if lam > 0:
                p_curve = (1.0 - np.exp(-lam * t_years)) * 100.0
                ax3.plot(t_years, p_curve, label=f"ML ≥ {m_val} (λ={lam:.3f}/年, 回歸期{1/lam:.1f}年)",
                         color=curve_colors[idx], linewidth=2.0)

    ax3.set_xlabel("未來特定時間跨度 Δt (年)", fontsize=10)
    ax3.set_ylabel("累積破裂機率 (%)", fontsize=10)
    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 105)
    ax3.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax3.text(1.1, 15, "1年", fontsize=8.5, color='gray')
    ax3.axvline(5.0, color='gray', linestyle=':', alpha=0.7)
    ax3.text(5.1, 15, "5年", fontsize=8.5, color='gray')
    ax3.axvline(10.0, color='gray', linestyle=':', alpha=0.7)
    ax3.text(10.1, 15, "10年", fontsize=8.5, color='gray')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 4 (右下): 四大特定區域 Gutenberg-Richter 規模-頻率擬合線性對比
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#ffffff')
    ax4.set_title("(D) 本地構造規模分佈線性規律：Gutenberg-Richter 擬合線與 b 值", 
                  fontsize=12, fontweight='bold', pad=10)
    
    m_bins = np.arange(3.0, 7.5, 0.2)
    for idx, (r_name, r_df) in enumerate(preset_cases_results.items()):
        # 取得該區域的 a 與 b
        lam_m5 = r_df[r_df['規模門檻(>=ML)'] == 5.0]['年平均發生率(次/年)'].values[0]
        lam_m6 = r_df[r_df['規模門檻(>=ML)'] == 6.0]['年平均發生率(次/年)'].values[0]
        if lam_m5 > 0 and lam_m6 > 0:
            b_est = -(math.log10(lam_m6) - math.log10(lam_m5)) / (6.0 - 5.0)
            a_est = math.log10(lam_m5) + b_est * 5.0
            y_fit = a_est - b_est * m_bins
            ax4.plot(m_bins, y_fit, label=f"{r_name} (b={b_est:.2f})", color=colors[idx], linewidth=2.0)
            
    ax4.set_xlabel("芮氏規模 ML", fontsize=10)
    ax4.set_ylabel("log10 累積年發生率 log10(N/yr)", fontsize=10)
    ax4.set_xlim(3.0, 7.2)
    ax4.set_ylim(-3, 3)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    plt.suptitle("臺灣特定時空區域地震機率預測模型演算結果\n(指定空間半徑與邊界、自選未來 7天/30天/1年/5年/10年 時間跨度之量化機率演算)",
                 fontsize=15, fontweight='bold', y=0.97)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[3/4] 預測綜合圖表已輸出：{output_path}")

def main():
    df, duration_years = load_catalog()
    
    # 定義四個代表性「特定空間區域」範例
    preset_regions = {
        "花蓮近海破裂帶": {
            "lon_min": 121.45, "lon_max": 121.85,
            "lat_min": 23.75, "lat_max": 24.25
        },
        "嘉南梅山-中埔盲斷層區": {
            "center_lon": 120.55, "center_lat": 23.48, "radius_km": 25.0
        },
        "雙北山腳斷層陷落區": {
            "lon_min": 121.35, "lon_max": 121.65,
            "lat_min": 24.95, "lat_max": 25.25
        },
        "恆春外海板塊邊界區": {
            "center_lon": 120.70, "center_lat": 21.80, "radius_km": 40.0
        }
    }
    
    target_days = [7, 30, 90, 365, 365 * 5, 365 * 10]
    
    all_dfs = []
    preset_cases_results = {}
    
    print("[2/4] 執行特定空間與特定時間機率演算...")
    for r_name, params in preset_regions.items():
        res_df, sub_df, desc, a, b, anomaly = run_specific_region_prediction(
            df, duration_years, params, target_days
        )
        res_df['自訂區名'] = r_name
        all_dfs.append(res_df)
        preset_cases_results[r_name] = res_df
        print(f"\n--- 【{r_name}】預測結果 ---")
        print(f"範圍描述: {desc}")
        print(f"32年事件數: {len(sub_df):,} 筆 | G-R b值: {b:.2f} | 近期活動度比: {anomaly:.2f}")
        for idx, row in res_df[res_df['規模門檻(>=ML)'].isin([5.0, 6.0])].iterrows():
            m = row['規模門檻(>=ML)']
            print(f"  ML >= {m}: 年率 {row['年平均發生率(次/年)']}/年 | 30天機率: {row['30天內破裂機率(%)']}% | 1年機率: {row['1年內破裂機率(%)']}% | 5年機率: {row['5年內破裂機率(%)']}%")
            
    final_csv_df = pd.concat(all_dfs, ignore_index=True)
    final_csv_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[4/4] 試算表已輸出至：{OUTPUT_CSV}")
    
    plot_custom_predictions(preset_cases_results, OUTPUT_IMG)

if __name__ == '__main__':
    main()
