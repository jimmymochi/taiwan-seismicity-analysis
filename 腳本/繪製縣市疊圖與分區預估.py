import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import geopandas as gpd

# 中文字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 路徑設定（支援本地與 repo 相對路徑）
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dirs = [os.path.join(script_dir, "..", "地震目錄"), r"D:\JIMMY CHEN\達意專題\地震目錄"]
data_dir = next((d for d in data_dirs if os.path.exists(d)), data_dirs[0])

gpkg_paths = [
    os.path.join(script_dir, "..", "圖資", "TW_country_pop_TM2.gpkg"),
    r"D:\JIMMY CHEN\地理資訊系統參考資料-20260707T054908Z-3-001\地理資訊系統參考資料\TW_country_pop_TM2.gpkg"
]
gpkg_path = next((p for p in gpkg_paths if os.path.exists(p)), gpkg_paths[0])

# 1. 讀取地震資料
files = sorted(glob.glob(os.path.join(data_dir, "GDMS_*.csv")))
print(f"載入地震目錄: {len(files)} 個檔案 (來源: {data_dir})...")
dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'], errors='coerce')
df_all = df_all.dropna(subset=['datetime', 'lat', 'lon', 'depth', 'ML']).sort_values('datetime').reset_index(drop=True)

total_events = len(df_all)
total_years = (df_all['datetime'].max() - df_all['datetime'].min()).total_seconds() / (365.25 * 86400)
print(f"有效事件數: {total_events}, 年數: {total_years:.2f} 年")

# 2. 讀取縣市圖 GPKG (EPSG:3826 轉 EPSG:4326)
gdf_raw = gpd.read_file(gpkg_path)
gdf_wgs84 = gdf_raw.to_crs(epsg=4326)

county_map = {
    '09007': '連江縣', '09020': '金門縣', '10002': '宜蘭縣', '10004': '新竹縣',
    '10005': '苗栗縣', '10007': '彰化縣', '10008': '南投縣', '10009': '雲林縣',
    '10010': '嘉義縣', '10013': '屏東縣', '10014': '臺東縣', '10015': '花蓮縣',
    '10016': '澎湖縣', '10017': '基隆市', '10018': '新竹市', '10020': '嘉義市',
    '63': '臺北市', '64': '高雄市', '65': '新北市', '66': '臺中市',
    '67': '臺南市', '68': '桃園市'
}
gdf_wgs84['County_Name'] = gdf_wgs84['County_ID'].map(county_map)

# 3. 定義五大核心分區
regions = [
    {
        'id': 'R1',
        'name': '東北部海域 (宜蘭-花蓮-沖繩海槽)',
        'lat': (23.8, 25.5), 'lon': (121.3, 122.8),
        'color': '#d32f2f',
        'counties': '宜蘭縣、花蓮北部、基隆海域',
        'struct': '琉球隱沒帶弧前盆地與沖繩海槽擴張軸，全台地震能量釋放樞紐。'
    },
    {
        'id': 'R2',
        'name': '花東縱谷與台東外海 (板塊縫合帶)',
        'lat': (22.2, 23.8), 'lon': (121.0, 122.0),
        'color': '#f57c00',
        'counties': '花蓮中南部、臺東縣',
        'struct': '菲律賓海板塊碰撞歐亞板塊主要縫合帶，走滑兼逆衝，強震頻率高。'
    },
    {
        'id': 'R3',
        'name': '嘉南與中南部陸域 (西部前緣逆衝帶)',
        'lat': (22.8, 23.8), 'lon': (120.0, 120.9),
        'color': '#2e7d32',
        'counties': '嘉義縣市、臺南市、雲林南、高雄北',
        'struct': '梅山、新化、六甲等盲斷層系統，週期長但極淺層，危害風險最嚴峻。'
    },
    {
        'id': 'R4',
        'name': '中部地區 (921集集破裂帶/彰投中)',
        'lat': (23.8, 24.6), 'lon': (120.4, 121.2),
        'color': '#0288d1',
        'counties': '臺中市、南投縣、彰化縣、苗栗南',
        'struct': '車籠埔斷層、雙冬斷層帶，經歷 1999 年破裂後處於應力再調整期。'
    },
    {
        'id': 'R5',
        'name': '大台北與北部陸域 (都會盆地構造)',
        'lat': (24.7, 25.4), 'lon': (121.2, 121.9),
        'color': '#7b1fa2',
        'counties': '臺北市、新北市、桃園市、基隆市',
        'struct': '山腳正斷層、大屯火山地熱群，震頻極低，長週期大震風險具高度威脅。'
    }
]

# 計算各分區 G-R 參數
reg_stats = []
for reg in regions:
    sub = df_all[(df_all['lat'] >= reg['lat'][0]) & (df_all['lat'] <= reg['lat'][1]) &
                 (df_all['lon'] >= reg['lon'][0]) & (df_all['lon'] <= reg['lon'][1])]
    cnt = len(sub)
    pct = cnt / total_events * 100
    
    m_bins = np.arange(2.5, 7.0, 0.1)
    ann_rate = np.array([(sub['ML'] >= m).sum() / total_years for m in m_bins])
    fit_mask = (m_bins >= 3.0) & (m_bins <= 6.0) & (ann_rate > 0)
    
    poly = np.polyfit(m_bins[fit_mask], np.log10(ann_rate[fit_mask]), 1)
    b_val = -poly[0]
    a_yr = poly[1]
    
    r5 = 10 ** (a_yr - b_val * 5.0)
    t5_days = 365.25 / r5 if r5 > 0 else np.nan
    t5_str = f"{t5_days:.0f} 天" if t5_days < 365 else f"{t5_days/365.25:.1f} 年"
    
    r6 = 10 ** (a_yr - b_val * 6.0)
    t6_days = 365.25 / r6 if r6 > 0 else np.nan
    t6_str = f"{t6_days:.0f} 天 ({t6_days/30.4:.1f}個月)" if t6_days < 365 else f"{t6_days/365.25:.1f} 年"
    
    reg_stats.append({
        'id': reg['id'],
        'name': reg['name'],
        'color': reg['color'],
        'count': cnt,
        'pct': pct,
        'b_val': b_val,
        'a_yr': a_yr,
        't5_str': t5_str,
        't6_str': t6_str,
        'counties': reg['counties'],
        'struct': reg['struct']
    })

# 4. 繪圖：左地圖、右卡片看板
fig = plt.figure(figsize=(22, 15.5), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.18)

# 左子圖：地圖
ax_map = fig.add_subplot(gs[0])

# 4.1 繪製縣市面 (底圖)
gdf_wgs84.plot(ax=ax_map, facecolor='#f8f9fa', edgecolor='#78909c', linewidth=0.75, zorder=1)

# 4.2 疊加地震熱度圖 (微震背景)
bg_sample = df_all.sample(n=min(70000, len(df_all)), random_state=42)
hex_plot = ax_map.hexbin(bg_sample['lon'], bg_sample['lat'], gridsize=95, cmap='YlOrRd', mincnt=1, alpha=0.48, bins='log', zorder=2)

# 4.3 疊加顯著中大震 (ML >= 5.5)
sig_df = df_all[df_all['ML'] >= 5.5]
sc = ax_map.scatter(sig_df['lon'], sig_df['lat'], s=(sig_df['ML']-4.6)**3.1 * 6,
                    c=sig_df['depth'], cmap='coolwarm', edgecolors='#212121', linewidth=0.6, alpha=0.88, zorder=4)

# Colorbars
cb_hex = fig.colorbar(hex_plot, ax=ax_map, fraction=0.038, pad=0.02, orientation='vertical')
cb_hex.set_label('微小震活動密集度 (Log count)', fontsize=9.5)

cb_sc = fig.colorbar(sc, ax=ax_map, fraction=0.038, pad=0.07, orientation='vertical')
cb_sc.set_label('震源深度 (km)', fontsize=9.5)

# 4.4 標註主要縣市名稱
target_counties = ['宜蘭縣', '花蓮縣', '臺東縣', '臺北市', '新北市', '桃園市', '新竹縣', '苗栗縣', 
                   '臺中市', '南投縣', '彰化縣', '雲林縣', '嘉義縣', '臺南市', '高雄市', '屏東縣']
for idx, row in gdf_wgs84.iterrows():
    c_name = row['County_Name']
    if c_name in target_counties:
        pt = row.geometry.representative_point()
        x_off, y_off = 0, 0
        if c_name == '花蓮縣': x_off = -0.15
        elif c_name == '宜蘭縣': x_off = -0.15
        elif c_name == '臺南市': x_off = -0.05
        ax_map.text(pt.x + x_off, pt.y + y_off, c_name.replace('縣','').replace('市',''), 
                    fontsize=8.5, fontweight='bold', color='#37474f', ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#cfd8dc", alpha=0.75, lw=0.5))

# 4.5 繪製五大分區矩形外框與標籤
for reg in regions:
    lat_min, lat_max = reg['lat']
    lon_min, lon_max = reg['lon']
    width = lon_max - lon_min
    height = lat_max - lat_min
    rect = patches.Rectangle((lon_min, lat_min), width, height, linewidth=2.2, 
                             edgecolor=reg['color'], facecolor='none', linestyle='--', zorder=6)
    ax_map.add_patch(rect)
    
    tag_x = lon_min + 0.08
    tag_y = lat_max - 0.12
    ax_map.text(tag_x, tag_y, f"【{reg['id']}】{reg['name'].split('(')[0]}", 
                fontsize=9.5, fontweight='bold', color=reg['color'], zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=reg['color'], lw=1.2, alpha=0.92))

ax_map.set_xlim(119.2, 123.0)
ax_map.set_ylim(21.4, 25.8)
ax_map.set_xlabel('經度 (°E)', fontsize=12, fontweight='bold')
ax_map.set_ylabel('緯度 (°N)', fontsize=12, fontweight='bold')
ax_map.set_title('台灣縣市行政界線與五大重點地震分區活動圖\n(疊合 TW_country_pop_TM2.gpkg 圖資 & 1994–2026 目錄)', 
                 fontsize=13.5, fontweight='bold', pad=12)
ax_map.grid(True, ls=":", alpha=0.5)

# 右子圖：五大分區預估卡片面板
ax_cards = fig.add_subplot(gs[1])
ax_cards.axis('off')

ax_cards.text(0.5, 0.99, '台灣五大構造分區 地震危害與重現週期預估評估', 
              fontsize=15, fontweight='bold', ha='center', va='top', color='#1a237e')
ax_cards.text(0.5, 0.965, '基於 32.6 年全目錄 87.3 萬筆紀錄 G-R 線性模型演算 (1994–2026)', 
              fontsize=10.5, ha='center', va='top', color='#546e7a')

card_y_positions = [0.90, 0.73, 0.56, 0.39, 0.22]
card_height = 0.145

for idx, stat in enumerate(reg_stats):
    top_y = card_y_positions[idx]
    
    rect_box = patches.FancyBboxPatch((0.02, top_y - card_height), 0.96, card_height,
                                      boxstyle="round,pad=0.012,rounding_size=0.018",
                                      facecolor='#ffffff', edgecolor=stat['color'], linewidth=2.0,
                                      transform=ax_cards.transAxes)
    ax_cards.add_patch(rect_box)
    
    title_text = f"【{stat['id']}】{stat['name']}"
    ax_cards.text(0.04, top_y - 0.012, title_text, transform=ax_cards.transAxes,
                  fontsize=11.5, fontweight='bold', color=stat['color'], va='top')
    
    data_str1 = f"● 涵蓋縣市: {stat['counties']}"
    data_str2 = f"● 地震累積筆數: {stat['count']:,} 次 (佔全台 {stat['pct']:.1f}%)"
    data_str3 = f"● G-R 線性參數: b 值 = {stat['b_val']:.2f} (應力釋放結構) | a 值 = {stat['a_yr']:.2f}"
    
    ax_cards.text(0.05, top_y - 0.038, data_str1, transform=ax_cards.transAxes, fontsize=9.2, color='#37474f')
    ax_cards.text(0.05, top_y - 0.059, data_str2, transform=ax_cards.transAxes, fontsize=9.2, color='#37474f')
    ax_cards.text(0.05, top_y - 0.080, data_str3, transform=ax_cards.transAxes, fontsize=9.2, color='#37474f')
    
    pred_text = f"★ 預估重現期： M >= 5.0 平均約【 {stat['t5_str']} 】  |  M >= 6.0 平均約【 {stat['t6_str']} 】"
    ax_cards.text(0.05, top_y - 0.106, pred_text, transform=ax_cards.transAxes,
                  fontsize=9.6, fontweight='bold', color='#b71c1c',
                  bbox=dict(boxstyle="round,pad=0.2", fc="#fffde7", ec="#fbc02d", lw=1.0))
    
    ax_cards.text(0.05, top_y - 0.133, f"地質特徵: {stat['struct']}", transform=ax_cards.transAxes,
                  fontsize=8.5, color='#78909c')

# 底部科學總結
bottom_summary = (
    "【分區解讀重點】\n"
    "1. 東部與東北部（R1, R2）承受了台灣約 61.5% 的地震頻率，M>=6.0 平均每 8 個月至 1 年就會發生，屬常態性能量釋放。\n"
    "2. 嘉南平原（R3）地震佔比達 17.6%，M>=6.0 平均重現期約 4.1 年，特點是震源極淺且斷層閉鎖，需防範突發大震。\n"
    "3. 北部都會區（R5）M>=6.0 重現期長達 150 年以上，雖然平日罕見，但因人口密度最高，長週期危害評估不可輕忽。"
)
ax_cards.text(0.02, 0.012, bottom_summary, transform=ax_cards.transAxes,
              fontsize=9.2, color='#263238', va='bottom',
              bbox=dict(boxstyle="round,pad=0.4", fc="#eceff1", ec="#90a4ae", lw=1.2))

# 儲存輸出圖檔 (本專案目錄)
out_path = os.path.join(script_dir, "..", "台灣縣市地震活動度疊圖與分區空間預估圖.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"已儲存高畫質圖檔至: {out_path}")
