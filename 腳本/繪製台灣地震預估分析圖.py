import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

# 設置中文字體相容性
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 路徑設定
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dirs = [os.path.join(script_dir, "..", "地震目錄"), r"D:\JIMMY CHEN\達意專題\地震目錄"]
data_dir = next((d for d in data_dirs if os.path.exists(d)), data_dirs[0])

gpkg_paths = [
    os.path.join(script_dir, "..", "圖資", "TW_country_pop_TM2.gpkg"),
    r"D:\CIL資料夾\地震地下水\嘉南166地下水測站圖資\臺灣縣市邊界_嘉南與鄰近.geojson"
]
tw_map_path = next((p for p in gpkg_paths if os.path.exists(p)), gpkg_paths[0])

# 1. 讀取地震目錄資料 (87.3 萬筆)
files = sorted(glob.glob(os.path.join(data_dir, "GDMS_*.csv")))
print(f"載入目錄檔案共 {len(files)} 個 (來源: {data_dir})...")

dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'], errors='coerce')
df_all = df_all.dropna(subset=['datetime', 'lat', 'lon', 'depth', 'ML']).sort_values('datetime').reset_index(drop=True)

total_events = len(df_all)
start_dt = df_all['datetime'].min()
end_dt = df_all['datetime'].max()
time_span_years = (end_dt - start_dt).total_seconds() / (365.25 * 86400)
print(f"總事件數: {total_events}, 時間涵蓋: {start_dt.date()} ~ {end_dt.date()} ({time_span_years:.2f} 年)")

# 2. 建立 2x2 畫布
fig, axs = plt.subplots(2, 2, figsize=(19, 16.5), dpi=300)
plt.subplots_adjust(hspace=0.28, wspace=0.24)

# -------------------------------------------------------------------------
# 子圖 1: Gutenberg-Richter 定律 (規模 vs 年發生次數)
# -------------------------------------------------------------------------
ax1 = axs[0, 0]
m_bins = np.arange(2.0, 7.6, 0.1)
cum_counts = np.array([(df_all['ML'] >= m).sum() for m in m_bins])
annual_rate = cum_counts / time_span_years

fit_mask = (m_bins >= 3.0) & (m_bins <= 6.5)
m_fit = m_bins[fit_mask]
log_r_fit = np.log10(annual_rate[fit_mask])
poly = np.polyfit(m_fit, log_r_fit, 1)
b_val = -poly[0]
a_yr = poly[1]

m_line = np.linspace(2.0, 7.5, 100)
r_line = 10 ** (a_yr - b_val * m_line)

ax1.scatter(m_bins, annual_rate, color='#0d47a1', s=28, alpha=0.85, edgecolors='none', label='實際觀測年發生率 (次/年)')
ax1.plot(m_line, r_line, color='#c62828', lw=2.4, linestyle='--', 
         label=f'G-R 線性擬合線: log10(N) = {a_yr:.2f} - {b_val:.2f} * ML\n(b 值 = {b_val:.2f}，代表規模每增 1 級，次數約減 8.7 倍)')

ax1.set_yscale('log')
ax1.set_xlim(2.0, 7.5)
ax1.set_ylim(0.01, 100000)
ax1.set_xlabel('地震規模 (ML)', fontsize=13, fontweight='bold')
ax1.set_ylabel('年累積發生次數 N(M >= ML) [次/年]', fontsize=13, fontweight='bold')
ax1.set_title('【規模與頻率的線性規律】古騰堡－芮克特定律 (G-R Law)', fontsize=14, fontweight='bold', pad=12)
ax1.grid(True, which="both", ls=":", alpha=0.6)
ax1.legend(loc='upper right', fontsize=10.5, frameon=True, facecolor='white', framealpha=0.92)

key_m = [5.0, 6.0, 6.5, 7.0]
for km in key_m:
    rate = 10 ** (a_yr - b_val * km)
    t_days = 365.25 / rate
    text_label = f"M >= {km}: 約 {rate:.1f} 次/年\n(平均每 {t_days:.0f} 天一次)" if t_days < 365 else f"M >= {km}: 約 {rate:.2f} 次/年\n(平均每 {t_days/365.25:.1f} 年一次)"
    ax1.annotate(text_label, xy=(km, rate), xytext=(km+0.16, rate*2.0),
                 arrowprops=dict(arrowstyle="->", color='#333333', lw=1.3),
                 fontsize=9.5, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff9c4", ec="#fbc02d", lw=1.2, alpha=0.95))

# -------------------------------------------------------------------------
# 子圖 2: 累積 Benioff 應變釋放與板塊等速線性輸入
# -------------------------------------------------------------------------
ax2 = axs[0, 1]
df_all['benioff'] = 10.0 ** (2.4 + 0.75 * df_all['ML'])
cum_s = df_all['benioff'].cumsum() / 1e8

t_start_num = df_all['datetime'].iloc[0].timestamp()
t_end_num = df_all['datetime'].iloc[-1].timestamp()
t_all_num = df_all['datetime'].apply(lambda x: x.timestamp())

total_strain = cum_s.iloc[-1]
linear_slope = total_strain / (t_end_num - t_start_num)
linear_trend = linear_slope * (t_all_num - t_start_num)

plot_step = max(1, len(df_all) // 4000)
sample_idx = np.arange(0, len(df_all), plot_step)
if sample_idx[-1] != len(df_all)-1: sample_idx = np.append(sample_idx, len(df_all)-1)

sample_dt = df_all['datetime'].iloc[sample_idx]
sample_cum_s = cum_s.iloc[sample_idx]
sample_trend = linear_trend.iloc[sample_idx]

mask_deficit = (sample_dt >= pd.to_datetime('2008-01-01')) & (sample_dt <= pd.to_datetime('2022-09-01'))
ax2.fill_between(sample_dt[mask_deficit], sample_cum_s[mask_deficit], sample_trend[mask_deficit],
                 color='#ffcc80', alpha=0.45, label='2008–2022 應變虧損蓄積期 (Seismic Deficit)')

ax2.plot(sample_dt, sample_cum_s, color='#01579b', lw=2.4, label='實際累積應變釋放 (階梯跳躍)')
ax2.plot(sample_dt, sample_trend, color='#e65100', lw=2.0, linestyle='--', label='板塊構造恆定能量輸入 (等速線性累積假說)')

major_events = [
    ('1999-09-20 17:47', 7.3, '1999 921集集地震 (ML 7.3)'),
    ('2006-12-26 12:34', 7.0, '2006 恆春外海雙震 (ML 7.0)'),
    ('2022-09-18 06:44', 6.8, '2022 台東池上地震 (ML 6.8)'),
    ('2024-04-02 23:58', 7.2, '2024 0403花蓮地震 (ML 7.2)'),
]
for dt_str, ml, label_name in major_events:
    target_dt = pd.to_datetime(dt_str)
    idx = (df_all['datetime'] - target_dt).abs().idxmin()
    pos_x = df_all['datetime'].iloc[idx]
    pos_y = cum_s.iloc[idx]
    ax2.scatter(pos_x, pos_y, color='#b71c1c', s=65, zorder=5)
    offset_y = 10 if '2024' not in label_name else 6
    offset_x = -pd.Timedelta(days=1300) if '1999' in label_name or '2022' in label_name else pd.Timedelta(days=100)
    ax2.annotate(label_name, xy=(pos_x, pos_y), xytext=(pos_x + offset_x, pos_y + offset_y),
                 arrowprops=dict(arrowstyle="->", color='#b71c1c', lw=1.3),
                 fontsize=9, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.25", fc="#ffebee", ec="#ef5350", lw=1.1, alpha=0.92))

ax2.set_xlabel('時間年份 (1994 ~ 2026)', fontsize=13, fontweight='bold')
ax2.set_ylabel('累積 Benioff 應變釋放 (x 10^8)', fontsize=13, fontweight='bold')
ax2.set_title('【時間與能量的線性累積】應變釋放階梯 vs. 板塊穩定輸入線', fontsize=14, fontweight='bold', pad=12)
ax2.grid(True, ls=":", alpha=0.6)
ax2.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.92)

# -------------------------------------------------------------------------
# 子圖 3: 空間分佈 (未來在哪裡？)
# -------------------------------------------------------------------------
ax3 = axs[1, 0]
try:
    gdf_tw = gpd.read_file(tw_map_path)
    if gdf_tw.crs and gdf_tw.crs.to_epsg() != 4326:
        gdf_tw = gdf_tw.to_crs(epsg=4326)
    gdf_tw.boundary.plot(ax=ax3, color='#424242', linewidth=0.85, alpha=0.8, zorder=3)
except Exception as e:
    print(f"載入縣市邊界失敗: {e}")

bg_sample = df_all.sample(n=min(60000, len(df_all)), random_state=42)
h = ax3.hexbin(bg_sample['lon'], bg_sample['lat'], gridsize=85, cmap='YlOrRd', mincnt=1, alpha=0.55, bins='log', zorder=1)
cb = fig.colorbar(h, ax=ax3, fraction=0.046, pad=0.03)
cb.set_label('微小震活動密集度 (Log count)', fontsize=10)

sig_df = df_all[df_all['ML'] >= 5.8]
sc = ax3.scatter(sig_df['lon'], sig_df['lat'], s=(sig_df['ML']-4.5)**3.2 * 5.5, 
                 c=sig_df['depth'], cmap='viridis', edgecolors='black', linewidth=0.65, alpha=0.88, zorder=4)
cb_depth = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.09)
cb_depth.set_label('震源深度 (km)', fontsize=10)

ax3.set_xlim(119.2, 122.8)
ax3.set_ylim(21.4, 25.8)
ax3.set_xlabel('經度 (°E)', fontsize=13, fontweight='bold')
ax3.set_ylabel('緯度 (°N)', fontsize=13, fontweight='bold')
ax3.set_title('【在哪裡？空間構造分佈】台灣主要震源構造帶與顯著地震', fontsize=14, fontweight='bold', pad=12)

ax3.text(121.75, 24.35, 'A. 琉球海溝/隱沒帶\n   (花蓮-宜蘭海域/沖繩海槽)\n   全台地震最密集、釋放頻率最高', 
         fontsize=9, fontweight='bold', color='#b71c1c', 
         bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#b71c1c", lw=1.2, alpha=0.88))
ax3.text(121.45, 23.05, 'B. 花東縱谷縫合帶\n   (板塊邊界走滑逆衝系統)', 
         fontsize=8.5, fontweight='bold', color='#e65100', 
         bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#e65100", lw=1.1, alpha=0.88))
ax3.text(119.45, 23.85, 'C. 西部麓山帶/盲斷層\n   (921破裂帶/嘉南斷層帶)\n   平常震少，長期蓄積後風險極高', 
         fontsize=8.5, fontweight='bold', color='#1b5e20', 
         bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#1b5e20", lw=1.1, alpha=0.88))
ax3.text(119.65, 21.75, 'D. 恆春外海/海域\n   (馬尼拉隱沒帶北端)', 
         fontsize=8.5, fontweight='bold', color='#0d47a1', 
         bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#0d47a1", lw=1.1, alpha=0.88))
ax3.grid(True, ls=":", alpha=0.5)

# -------------------------------------------------------------------------
# 子圖 4: 重現週期預估模型 (規模 vs 平均間隔時間)
# -------------------------------------------------------------------------
ax4 = axs[1, 1]
m_pred = np.linspace(3.0, 7.5, 100)
rate_pred = 10 ** (a_yr - b_val * m_pred)
return_days_pred = 365.25 / rate_pred

m_obs = np.arange(3.5, 7.3, 0.2)
obs_return_days = []
for m in m_obs:
    cnt = (df_all['ML'] >= m).sum()
    obs_return_days.append(time_span_years * 365.25 / cnt if cnt > 0 else np.nan)

ax4.plot(m_pred, return_days_pred, color='#2e7d32', lw=2.5, label='G-R 線性模型預估重現期 (T = 10^(bM - a))')
ax4.scatter(m_obs, obs_return_days, color='#d81b60', s=38, zorder=5, label='1994–2026 實際歷史目錄觀測間隔')

ax4.set_yscale('log')
ax4.set_xlim(3.0, 7.5)
ax4.set_ylim(0.5, 15000)
ax4.set_xlabel('地震規模 (ML)', fontsize=13, fontweight='bold')
ax4.set_ylabel('平均重現間隔時間 (天數, 對數尺度)', fontsize=13, fontweight='bold')
ax4.set_title('【大概多久發生一次？】規模與平均重現週期預估', fontsize=14, fontweight='bold', pad=12)
ax4.grid(True, which="both", ls=":", alpha=0.6)
ax4.legend(loc='upper left', fontsize=10.5, frameon=True, facecolor='white', framealpha=0.92)

time_markers = [
    (1, '1 天', '#9e9e9e'),
    (7, '1 週 (7天)', '#9e9e9e'),
    (30.4, '1 個月 (約30天)', '#78909c'),
    (182.6, '半年 (約180天)', '#78909c'),
    (365.25, '1 年 (365天)', '#5c6bc0'),
    (1826.25, '5 年', '#5c6bc0'),
    (3652.5, '10 年', '#3949ab')
]
for d_val, d_label, col in time_markers:
    ax4.axhline(d_val, color=col, linestyle=':', alpha=0.65, lw=1.1)
    ax4.text(7.35, d_val*1.08, d_label, fontsize=8.5, color=col, va='bottom', ha='right', fontweight='bold')

summary_text = (
    "【全台及周圍海域 統計預估對照】\n"
    "• M >= 4.5: 平均每  3.8 天發生一次\n"
    "• M >= 5.0: 平均每 13.5 天 (約半個月)\n"
    "• M >= 5.5: 平均每 40.0 天 (約 1.3 個月)\n"
    "• M >= 6.0: 平均每  118 天 (約 4 個月)\n"
    "• M >= 6.5: 平均每  348 天 (約 1 年)\n"
    "• M >= 7.0: 平均約 2.8 年 (全台海陸域)\n\n"
    "★ 地震學核心本質：\n"
    "1.「線性」體現在：規模對數與發生次數(G-R)\n"
    "   以及板塊恆定運動速度(每年約 7-8 公分)。\n"
    "2. 地震釋放則是階梯狀非線性的；因此無法像\n"
    "   定時鬧鐘精確預報下週幾點幾分破裂。"
)
ax4.text(0.04, 0.26, summary_text, transform=ax4.transAxes, fontsize=9.2,
         verticalalignment='bottom', 
         bbox=dict(boxstyle="round,pad=0.55", fc="#f1f8e9", ec="#43a047", lw=1.4, alpha=0.95))

out_path = os.path.join(script_dir, "..", "台灣地震活動度與線性預估模型解析.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"已儲存高畫質圖檔至: {out_path}")
