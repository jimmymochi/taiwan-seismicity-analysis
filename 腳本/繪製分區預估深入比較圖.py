import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 路徑設定
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dirs = [os.path.join(script_dir, "..", "地震目錄"), r"D:\JIMMY CHEN\達意專題\地震目錄"]
data_dir = next((d for d in data_dirs if os.path.exists(d)), data_dirs[0])

# 1. 讀取地震資料
files = sorted(glob.glob(os.path.join(data_dir, "GDMS_*.csv")))
print(f"載入地震目錄: {len(files)} 個檔案 (來源: {data_dir})...")
dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'], errors='coerce')
df_all = df_all.dropna(subset=['datetime', 'lat', 'lon', 'depth', 'ML']).sort_values('datetime').reset_index(drop=True)

total_years = (df_all['datetime'].max() - df_all['datetime'].min()).total_seconds() / (365.25 * 86400)

# 定義五大分區
regions = [
    {
        'id': 'R1',
        'name': '東北部海域 (宜蘭-花蓮-沖繩海槽)',
        'lat': (23.8, 25.5), 'lon': (121.3, 122.8),
        'color': '#d32f2f'
    },
    {
        'id': 'R2',
        'name': '花東縱谷與台東外海 (板塊縫合帶)',
        'lat': (22.2, 23.8), 'lon': (121.0, 122.0),
        'color': '#f57c00'
    },
    {
        'id': 'R3',
        'name': '嘉南與中南部陸域 (西部前緣逆衝帶)',
        'lat': (22.8, 23.8), 'lon': (120.0, 120.9),
        'color': '#2e7d32'
    },
    {
        'id': 'R4',
        'name': '中部地區 (921集集破裂帶/彰投中)',
        'lat': (23.8, 24.6), 'lon': (120.4, 121.2),
        'color': '#0288d1'
    },
    {
        'id': 'R5',
        'name': '大台北與北部陸域 (都會盆地構造)',
        'lat': (24.7, 25.4), 'lon': (121.2, 121.9),
        'color': '#7b1fa2'
    }
]

# 2. 建立 2x2 畫布
fig, axs = plt.subplots(2, 2, figsize=(19, 16), dpi=300)
plt.subplots_adjust(hspace=0.28, wspace=0.24)

# ----------------------------------------------------
# 子圖 1 (左上): 五大分區 G-R 線性擬合對比
# ----------------------------------------------------
ax1 = axs[0, 0]
m_range = np.linspace(2.5, 7.0, 100)
m_bins = np.arange(2.5, 7.1, 0.1)

params = {}
for reg in regions:
    sub = df_all[(df_all['lat'] >= reg['lat'][0]) & (df_all['lat'] <= reg['lat'][1]) &
                 (df_all['lon'] >= reg['lon'][0]) & (df_all['lon'] <= reg['lon'][1])].copy().reset_index(drop=True)
    
    ann_rate = np.array([(sub['ML'] >= m).sum() / total_years for m in m_bins])
    fit_mask = (m_bins >= 3.0) & (m_bins <= 6.0) & (ann_rate > 0)
    poly = np.polyfit(m_bins[fit_mask], np.log10(ann_rate[fit_mask]), 1)
    b_val = -poly[0]
    a_yr = poly[1]
    params[reg['id']] = {'b': b_val, 'a': a_yr, 'sub': sub}
    
    valid_obs = ann_rate > 0
    ax1.scatter(m_bins[valid_obs], ann_rate[valid_obs], color=reg['color'], s=18, alpha=0.6)
    r_fit = 10 ** (a_yr - b_val * m_range)
    label_txt = f"【{reg['id']}】{reg['name'].split('(')[0]}: b={b_val:.2f}, a={a_yr:.2f}"
    ax1.plot(m_range, r_fit, color=reg['color'], lw=2.2, label=label_txt)

ax1.set_yscale('log')
ax1.set_xlim(2.5, 7.0)
ax1.set_ylim(0.002, 50000)
ax1.set_xlabel('地震規模 (ML)', fontsize=12.5, fontweight='bold')
ax1.set_ylabel('年累積發生率 N(M >= ML) [次/年]', fontsize=12.5, fontweight='bold')
ax1.set_title('【規模與頻率線性對比】五大區域古騰堡－芮克特定律擬合', fontsize=13.5, fontweight='bold', pad=12)
ax1.grid(True, which="both", ls=":", alpha=0.6)
ax1.legend(loc='upper right', fontsize=9.8, frameon=True, facecolor='white', framealpha=0.92)

# ----------------------------------------------------
# 子圖 2 (右上): 五大分區規模 vs 重現週期對比
# ----------------------------------------------------
ax2 = axs[0, 1]
m_pred = np.linspace(3.0, 7.0, 100)

for reg in regions:
    p = params[reg['id']]
    rate = 10 ** (p['a'] - p['b'] * m_pred)
    return_days = 365.25 / rate
    label_txt = f"【{reg['id']}】{reg['name'].split('(')[0]}"
    ax2.plot(m_pred, return_days, color=reg['color'], lw=2.4, label=label_txt)

ax2.set_yscale('log')
ax2.set_xlim(3.0, 7.0)
ax2.set_ylim(0.5, 300000)
ax2.set_xlabel('地震規模 (ML)', fontsize=12.5, fontweight='bold')
ax2.set_ylabel('平均重現間隔 (天數 / 年數, 對數坐標)', fontsize=12.5, fontweight='bold')
ax2.set_title('【大概多久發生一次？】五大區域規模與平均重現週期預估', fontsize=13.5, fontweight='bold', pad=12)
ax2.grid(True, which="both", ls=":", alpha=0.6)
ax2.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.92)

time_ticks = [
    (7, '1 週 (7天)', '#9e9e9e'),
    (30.4, '1 個月 (約30天)', '#78909c'),
    (182.6, '半年 (180天)', '#78909c'),
    (365.25, '1 年 (365天)', '#5c6bc0'),
    (1826.25, '5 年', '#5c6bc0'),
    (3652.5, '10 年', '#3949ab'),
    (18262.5, '50 年', '#283593'),
    (36525.0, '100 年', '#1a237e'),
]
for d_val, d_lab, col in time_ticks:
    ax2.axhline(d_val, color=col, linestyle=':', alpha=0.65, lw=1.1)
    ax2.text(6.92, d_val * 1.1, d_lab, fontsize=8.5, color=col, va='bottom', ha='right', fontweight='bold')

# ----------------------------------------------------
# 子圖 3 (左下): 嘉南與中南部 (R3) 累積應變釋放與赤字評估
# ----------------------------------------------------
ax3 = axs[1, 0]
sub_r3 = params['R3']['sub'].copy().reset_index(drop=True)
sub_r3['benioff'] = 10.0 ** (2.4 + 0.75 * sub_r3['ML'])
cum_r3 = sub_r3['benioff'].cumsum() / 1e7

t_start_r3 = sub_r3['datetime'].iloc[0].timestamp()
t_end_r3 = sub_r3['datetime'].iloc[-1].timestamp()
total_r3 = cum_r3.iloc[-1]
slope_r3 = total_r3 / (t_end_r3 - t_start_r3)
trend_r3 = slope_r3 * (sub_r3['datetime'].apply(lambda x: x.timestamp()) - t_start_r3)

step_r3 = max(1, len(sub_r3) // 2500)
idx_r3 = np.arange(0, len(sub_r3), step_r3)
if idx_r3[-1] != len(sub_r3)-1: idx_r3 = np.append(idx_r3, len(sub_r3)-1)

dt_r3 = sub_r3['datetime'].iloc[idx_r3]
c_r3 = cum_r3.iloc[idx_r3]
tr_r3 = trend_r3.iloc[idx_r3]

mask_r3_def = (dt_r3 >= pd.to_datetime('2017-01-01')) & (dt_r3 <= pd.to_datetime('2024-06-01'))
ax3.fill_between(dt_r3[mask_r3_def], c_r3[mask_r3_def], tr_r3[mask_r3_def],
                 color='#c8e6c9', alpha=0.6, label='2017–2024 嘉南應力蓄積與能量赤字')

ax3.plot(dt_r3, c_r3, color='#2e7d32', lw=2.3, label='嘉南實際累積應變釋放 (階梯跳躍)')
ax3.plot(dt_r3, tr_r3, color='#e65100', lw=1.8, linestyle='--', label='板塊構造長期恆定能量輸入線')

r3_key_quakes = [
    ('1999-10-22 02:18', 6.4, '1999 嘉義地震 (ML 6.4)'),
    ('2010-03-04 00:18', 6.4, '2010 甲仙地震 (ML 6.4)'),
    ('2016-02-05 19:57', 6.6, '2016 美濃/臺南地震 (ML 6.6)'),
]
for dt_s, ml, lab in r3_key_quakes:
    tdt = pd.to_datetime(dt_s)
    close_idx = (sub_r3['datetime'] - tdt).abs().idxmin()
    px = sub_r3['datetime'].iloc[close_idx]
    py = cum_r3.iloc[close_idx]
    ax3.scatter(px, py, color='#b71c1c', s=55, zorder=5)
    ax3.annotate(lab, xy=(px, py), xytext=(px - pd.Timedelta(days=1200), py + 4.5),
                 arrowprops=dict(arrowstyle="->", color='#b71c1c', lw=1.2),
                 fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#ffebee", ec="#ef5350", lw=1.0, alpha=0.92))

ax3.set_xlabel('時間年份 (1994 ~ 2026)', fontsize=12.5, fontweight='bold')
ax3.set_ylabel('累積 Benioff 應變 (x 10^7)', fontsize=12.5, fontweight='bold')
ax3.set_title('【嘉南與中南部陸域專題】應變階梯釋放 vs. 長期能量赤字', fontsize=13.5, fontweight='bold', pad=12)
ax3.grid(True, ls=":", alpha=0.6)
ax3.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.92)

# ----------------------------------------------------
# 子圖 4 (右下): 東北部海域 (R1) 累積應變釋放與花蓮回補
# ----------------------------------------------------
ax4 = axs[1, 1]
sub_r1 = params['R1']['sub'].copy().reset_index(drop=True)
sub_r1['benioff'] = 10.0 ** (2.4 + 0.75 * sub_r1['ML'])
cum_r1 = sub_r1['benioff'].cumsum() / 1e8

t_start_r1 = sub_r1['datetime'].iloc[0].timestamp()
t_end_r1 = sub_r1['datetime'].iloc[-1].timestamp()
total_r1 = cum_r1.iloc[-1]
slope_r1 = total_r1 / (t_end_r1 - t_start_r1)
trend_r1 = slope_r1 * (sub_r1['datetime'].apply(lambda x: x.timestamp()) - t_start_r1)

step_r1 = max(1, len(sub_r1) // 3000)
idx_r1 = np.arange(0, len(sub_r1), step_r1)
if idx_r1[-1] != len(sub_r1)-1: idx_r1 = np.append(idx_r1, len(sub_r1)-1)

dt_r1 = sub_r1['datetime'].iloc[idx_r1]
c_r1 = cum_r1.iloc[idx_r1]
tr_r1 = trend_r1.iloc[idx_r1]

ax4.plot(dt_r1, c_r1, color='#d32f2f', lw=2.3, label='東北部海域實際累積應變釋放')
ax4.plot(dt_r1, tr_r1, color='#1565c0', lw=1.8, linestyle='--', label='板塊構造長期恆定能量輸入線')

r1_key_quakes = [
    ('2002-03-31 06:52', 6.8, '2002 331花蓮外海地震 (ML 6.8)'),
    ('2009-12-19 13:02', 6.9, '2009 花蓮外海地震 (ML 6.9)'),
    ('2024-04-02 23:58', 7.2, '2024 0403花蓮地震 (ML 7.2 回補巨大赤字)'),
]
for dt_s, ml, lab in r1_key_quakes:
    tdt = pd.to_datetime(dt_s)
    close_idx = (sub_r1['datetime'] - tdt).abs().idxmin()
    px = sub_r1['datetime'].iloc[close_idx]
    py = cum_r1.iloc[close_idx]
    ax4.scatter(px, py, color='#b71c1c', s=60, zorder=5)
    offset_y = 6 if '2024' not in lab else -12
    offset_x = -pd.Timedelta(days=1400) if '2024' not in lab else -pd.Timedelta(days=1800)
    ax4.annotate(lab, xy=(px, py), xytext=(px + offset_x, py + offset_y),
                 arrowprops=dict(arrowstyle="->", color='#b71c1c', lw=1.2),
                 fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.25", fc="#ffebee", ec="#ef5350", lw=1.0, alpha=0.92))

ax4.set_xlabel('時間年份 (1994 ~ 2026)', fontsize=12.5, fontweight='bold')
ax4.set_ylabel('累積 Benioff 應變 (x 10^8)', fontsize=12.5, fontweight='bold')
ax4.set_title('【東北部海域專題】應變階梯釋放與 0403 花蓮地震回補', fontsize=13.5, fontweight='bold', pad=12)
ax4.grid(True, ls=":", alpha=0.6)
ax4.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.92)

out_path = os.path.join(script_dir, "..", "台灣五大區域地震規模頻率與重現週期預估比較.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"已儲存高畫質圖檔至: {out_path}")
