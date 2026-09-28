import glob
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import geopandas as gpd

# 中文字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
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

# 1. 讀取地震數據
files = sorted(glob.glob(os.path.join(data_dir, "GDMS_*.csv")))
print(f"載入地震目錄: {len(files)} 個檔案...")
dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'], errors='coerce')
df_all = df_all.dropna(subset=['datetime', 'lat', 'lon', 'depth', 'ML']).sort_values('datetime').reset_index(drop=True)
print(f"總有效事件: {len(df_all):,}")

# 2. 讀取台灣縣市圖資 (轉換至 WGS84)
gdf_tw = gpd.read_file(gpkg_path).to_crs(epsg=4326)

# 提取縣市邊界線坐標供 3D 地表繪製
coast_lines = []
for geom in gdf_tw.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        coast_lines.append((list(x), list(y)))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            coast_lines.append((list(x), list(y)))

# 3. 分層抽樣建立 3D 視覺化資料集 (保留 100% 中強震，微震隨機抽樣)
# M >= 5.0 全部保留
df_m5 = df_all[df_all['ML'] >= 5.0]
# 4.0 <= M < 5.0 抽樣 5,000 筆
df_m4 = df_all[(df_all['ML'] >= 4.0) & (df_all['ML'] < 5.0)].sample(n=min(5000, len(df_all[(df_all['ML'] >= 4.0) & (df_all['ML'] < 5.0)])), random_state=42)
# M < 4.0 抽樣 8,000 筆 (作為地殼構造輪廓背景)
df_m3 = df_all[df_all['ML'] < 4.0].sample(n=min(8000, len(df_all[df_all['ML'] < 4.0])), random_state=42)

df_3d = pd.concat([df_m5, df_m4, df_m3]).sort_values('datetime').reset_index(drop=True)
print(f"3D 渲染抽樣點數: {len(df_3d):,} (含 M>=5.0 強震 {len(df_m5):,} 筆)")

# -------------------------------------------------------------------------
# A. 產出靜態高解析度 3D 空間透視與剖面解析圖 (PNG)
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 16), dpi=300)

# 子圖 1: 3D 透視鳥瞰視圖
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
# 繪製地表 (Z=0) 縣市邊界
for cx, cy in coast_lines:
    cz = np.zeros(len(cx))
    ax1.plot(cx, cy, cz, color='#455a64', lw=0.65, alpha=0.7)

# 繪製地震散點 (Z 設為負深度，以呈現地底構造)
scatter1 = ax1.scatter(df_3d['lon'], df_3d['lat'], -df_3d['depth'],
                       c=df_3d['depth'], cmap='turbo_r', s=(df_3d['ML'] - 2.5)**2.4 * 2.2,
                       alpha=0.6, edgecolors='none')

# 標註代表性大地震 (3D)
major_quakes = [
    ('1999-09-20', 120.82, 23.85, 8.0, 7.3, '1999 921集集 (ML 7.3, 8km)'),
    ('2016-02-06', 120.54, 22.92, 16.7, 6.6, '2016 美濃 (ML 6.6, 17km)'),
    ('2022-09-18', 121.20, 23.14, 7.0, 6.8, '2022 池上 (ML 6.8, 7km)'),
    ('2024-04-03', 121.67, 23.77, 15.5, 7.2, '2024 花蓮 (ML 7.2, 16km)')
]
for date_s, q_lon, q_lat, q_dep, q_ml, q_txt in major_quakes:
    ax1.scatter([q_lon], [q_lat], [-q_dep], color='#d50000', s=120, edgecolors='black', lw=1.2, zorder=10)
    ax1.text(q_lon, q_lat, -q_dep - 4, q_txt, fontsize=8, fontweight='bold', color='#b71c1c')

ax1.set_xlim(119.5, 122.8)
ax1.set_ylim(21.6, 25.6)
ax1.set_zlim(-70, 0)
ax1.set_xlabel('經度 (°E)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_ylabel('緯度 (°N)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_zlabel('深度 (km, 地底向下)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_title('【A. 全台 3D 空間透視鳥瞰圖】震源深度向地底延伸構造', fontsize=12.5, fontweight='bold', pad=12)
ax1.view_init(elev=28, azim=-55)

# 子圖 2: 東北部琉球隱沒帶側向 3D 透視圖 (側視板塊向下傾角)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
df_ne = df_3d[(df_3d['lat'] >= 23.5) & (df_3d['lon'] >= 121.2)]
for cx, cy in coast_lines:
    cz = np.zeros(len(cx))
    ax2.plot(cx, cy, cz, color='#90a4ae', lw=0.5, alpha=0.5)

scatter2 = ax2.scatter(df_ne['lon'], df_ne['lat'], -df_ne['depth'],
                       c=df_ne['depth'], cmap='turbo_r', s=(df_ne['ML'] - 2.5)**2.4 * 2.5,
                       alpha=0.65, edgecolors='none')
ax2.set_xlim(121.0, 122.8)
ax2.set_ylim(23.5, 25.5)
ax2.set_zlim(-70, 0)
ax2.set_xlabel('經度 (°E)', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_ylabel('緯度 (°N)', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_zlabel('深度 (km)', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_title('【B. 東北部琉球隱沒帶 3D 側視圖】瓦達蒂－貝尼奧夫隱沒舌板', fontsize=12.5, fontweight='bold', pad=12)
ax2.view_init(elev=18, azim=-80)

# 子圖 3: 東－西向垂直深度剖面 (E-W Cross Section, 緯度 23°N ~ 24.5°N 剖面)
ax3 = fig.add_subplot(2, 2, 3)
df_ew = df_all[(df_all['lat'] >= 23.0) & (df_all['lat'] <= 24.5) & (df_all['ML'] >= 3.5)]
sc3 = ax3.scatter(df_ew['lon'], df_ew['depth'], c=df_ew['ML'], cmap='plasma', s=(df_ew['ML']-2.8)**2.8 * 3.5, alpha=0.55, edgecolors='none')
cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.03)
cb3.set_label('地震規模 (ML)', fontsize=10)

ax3.set_ylim(70, 0) # 0在頂部，70在底部
ax3.set_xlim(119.8, 122.5)
ax3.set_xlabel('經度 (°E, 西部平原 → 中央山脈 → 花東縱谷 → 太平洋)', fontsize=11.5, fontweight='bold')
ax3.set_ylabel('震源深度 (km)', fontsize=11.5, fontweight='bold')
ax3.set_title('【C. 臺灣中部東－西向深度垂直剖面】(緯度 23.0°N ~ 24.5°N)', fontsize=12.5, fontweight='bold', pad=10)
ax3.grid(True, ls=":", alpha=0.6)

# 剖面構造標註
ax3.axvline(120.3, color='#388e3c', ls='--', alpha=0.8)
ax3.text(120.0, 65, '西部平原/麓山帶\n(0-30km 淺層碰撞)', fontsize=9, color='#2e7d32', fontweight='bold')

ax3.axvline(121.0, color='#1976d2', ls='--', alpha=0.8)
ax3.text(120.7, 65, '中央山脈變質岩帶\n(造山頂點)', fontsize=9, color='#1565c0', fontweight='bold')

ax3.axvline(121.6, color='#d32f2f', ls='--', alpha=0.8)
ax3.text(121.65, 65, '花東縱谷縫合帶\n與東部外海隱沒帶', fontsize=9, color='#c62828', fontweight='bold')

# 子圖 4: 南－北向垂直深度剖面 (N-S Cross Section, 經度 121.0°E ~ 122.0°E 剖面)
ax4 = fig.add_subplot(2, 2, 4)
df_ns = df_all[(df_all['lon'] >= 121.0) & (df_all['lon'] <= 122.0) & (df_all['ML'] >= 3.5)]
sc4 = ax4.scatter(df_ns['lat'], df_ns['depth'], c=df_ns['depth'], cmap='turbo', s=(df_ns['ML']-2.8)**2.8 * 3.5, alpha=0.55, edgecolors='none')
cb4 = fig.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.03)
cb4.set_label('震源深度 (km)', fontsize=10)

ax4.set_ylim(70, 0)
ax4.set_xlim(21.8, 25.5)
ax4.set_xlabel('緯度 (°N, 南部恆春 → 臺東 → 花蓮 → 宜蘭北部)', fontsize=11.5, fontweight='bold')
ax4.set_ylabel('震源深度 (km)', fontsize=11.5, fontweight='bold')
ax4.set_title('【D. 臺灣東部沿線南－北向深度垂直剖面】(經度 121.0°E ~ 122.0°E)', fontsize=12.5, fontweight='bold', pad=10)
ax4.grid(True, ls=":", alpha=0.6)

# 標註隱沒帶傾斜箭頭
ax4.annotate('琉球隱沒帶：由南向北顯著向深處傾斜 (Benioff Zone)', 
             xy=(24.8, 48), xytext=(22.5, 62),
             arrowprops=dict(arrowstyle="->", color='#d32f2f', lw=2.0),
             fontsize=10, fontweight='bold', color='#b71c1c',
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#ef5350", lw=1.2))

plt.subplots_adjust(hspace=0.28, wspace=0.22)
out_png = os.path.join(project_dir, "台灣3D地震空間構造建模與剖面解析.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"已產出靜態 3D 解析圖: {out_png}")

# -------------------------------------------------------------------------
# B. 產出互動式 3D 空間建模網頁 (HTML + Plotly.js CDN)
# -------------------------------------------------------------------------
# 準備 JSON 數據 (保留縣市線段與抽樣地震點)
coast_traces = []
for idx, (cx, cy) in enumerate(coast_lines):
    if len(cx) > 1:
        coast_traces.append({
            'type': 'scatter3d',
            'mode': 'lines',
            'x': cx,
            'y': cy,
            'z': [0] * len(cx),
            'line': {'color': '#78909c', 'width': 2.5},
            'hoverinfo': 'none',
            'showlegend': False
        })

# 地震點 Trace
sample_3d = df_3d.sample(n=min(12000, len(df_3d)), random_state=42)
hover_texts = [
    f"日期: {row['date']} {row['time']}<br>規模: ML {row['ML']:.2f}<br>深度: {row['depth']:.1f} km<br>坐標: ({row['lat']:.2f}°N, {row['lon']:.2f}°E)"
    for _, row in sample_3d.iterrows()
]

quake_trace = {
    'type': 'scatter3d',
    'mode': 'markers',
    'name': '地震事件 (1994–2026)',
    'x': sample_3d['lon'].tolist(),
    'y': sample_3d['lat'].tolist(),
    'z': (-sample_3d['depth']).tolist(), # 負深度向地底
    'text': hover_texts,
    'hoverinfo': 'text',
    'marker': {
        'size': ((sample_3d['ML'] - 2.0)**2.2 * 1.5).tolist(),
        'color': sample_3d['depth'].tolist(),
        'colorscale': 'Turbo_r',
        'opacity': 0.7,
        'colorbar': {
            'title': '震源深度 (km)',
            'len': 0.6,
            'y': 0.5
        }
    }
}

# 顯著大震 Trace (M >= 6.5)
big_quakes = df_all[df_all['ML'] >= 6.5]
big_hover = [
    f"<b>【顯著強震】</b><br>日期: {row['date']} {row['time']}<br>規模: <b>ML {row['ML']:.2f}</b><br>深度: {row['depth']:.1f} km<br>坐標: ({row['lat']:.2f}°N, {row['lon']:.2f}°E)"
    for _, row in big_quakes.iterrows()
]
big_trace = {
    'type': 'scatter3d',
    'mode': 'markers',
    'name': '強震 (ML >= 6.5)',
    'x': big_quakes['lon'].tolist(),
    'y': big_quakes['lat'].tolist(),
    'z': (-big_quakes['depth']).tolist(),
    'text': big_hover,
    'hoverinfo': 'text',
    'marker': {
        'size': 10,
        'color': '#ff1744',
        'symbol': 'diamond',
        'line': {'color': '#ffffff', 'width': 1.5},
        'opacity': 0.95
    }
}

all_traces = coast_traces + [quake_trace, big_trace]

plotly_data_json = json.dumps(all_traces)

html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>臺灣 3D 地震空間構造與震源深度互動模型 (1994–2026)</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Microsoft JhengHei", "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
        }}
        #header {{
            position: absolute;
            top: 15px;
            left: 20px;
            z-index: 10;
            background: rgba(18, 24, 38, 0.88);
            backdrop-filter: blur(8px);
            padding: 14px 22px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            max-width: 480px;
        }}
        h1 {{
            margin: 0 0 6px 0;
            font-size: 19px;
            color: #64b5f6;
            letter-spacing: 0.5px;
        }}
        p {{
            margin: 3px 0;
            font-size: 12.5px;
            color: #b0bec5;
            line-height: 1.45;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            z-index: 10;
            display: flex;
            gap: 10px;
        }}
        button {{
            background: #1e293b;
            color: #f1f5f9;
            border: 1px solid #475569;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        button:hover {{
            background: #334155;
            border-color: #94a3b8;
            color: #38bdf8;
        }}
        #plot-div {{
            width: 100vw;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>臺灣 3D 地震空間構造模型</h1>
        <p>● <b>數據基準</b>：中央氣象署 GDMS 1994–2026 目錄（873,145 筆）</p>
        <p>● <b>立體坐標</b>：X: 經度 | Y: 緯度 | Z: 地震深度（由地表 Z=0 向地底延伸至 -70 km）</p>
        <p>● <b>構造亮點</b>：可清楚旋轉觀察東北部<b>琉球隱沒帶</b>向北傾斜之貝尼奧夫板塊舌板，以及中西部淺層逆衝碰撞帶。</p>
        <p style="color: #94a3b8; font-size: 11.5px; margin-top: 6px;">💡 操作提示：滑鼠左鍵旋轉視角、滾輪縮放、按住右鍵平移、懸停查看地震詳情。</p>
    </div>

    <div id="controls">
        <button onclick="setView('default')">🌐 預設 3D 透視</button>
        <button onclick="setView('ryukyu')">↘ 東北琉球隱沒帶視角</button>
        <button onclick="setView('ew_section')">↔ 東西碰撞橫剖面</button>
        <button onclick="setView('topdown')">🗺 地表俯視圖</button>
    </div>

    <div id="plot-div"></div>

    <script>
        const data = {plotly_data_json};
        
        const layout = {{
            paper_bgcolor: '#0e1117',
            plot_bgcolor: '#0e1117',
            font: {{ color: '#e2e8f0', family: 'Microsoft JhengHei, sans-serif' }},
            margin: {{ l: 0, r: 0, b: 0, t: 0 }},
            scene: {{
                xaxis: {{ title: '經度 (°E)', backgroundcolor: '#131b26', gridcolor: '#1e293b', showbackground: true, range: [119.5, 122.8] }},
                yaxis: {{ title: '緯度 (°N)', backgroundcolor: '#131b26', gridcolor: '#1e293b', showbackground: true, range: [21.5, 25.8] }},
                zaxis: {{ title: '深度 (km, 0至-70km)', backgroundcolor: '#0f172a', gridcolor: '#1e293b', showbackground: true, range: [-70, 0] }},
                camera: {{
                    eye: {{ x: 1.35, y: -1.35, z: 0.95 }}
                }},
                aspectratio: {{ x: 1.1, y: 1.4, z: 0.7 }}
            }},
            legend: {{
                x: 0.85,
                y: 0.92,
                bgcolor: 'rgba(15, 23, 42, 0.8)',
                bordercolor: '#334155'
            }}
        }};

        Plotly.newPlot('plot-div', data, layout, {{ responsive: true }});

        function setView(mode) {{
            let update = {{}};
            if (mode === 'default') {{
                update = {{ 'scene.camera.eye': {{ x: 1.35, y: -1.35, z: 0.95 }} }};
            }} else if (mode === 'ryukyu') {{
                update = {{ 'scene.camera.eye': {{ x: 1.9, y: 0.3, z: 0.65 }} }};
            }} else if (mode === 'ew_section') {{
                update = {{ 'scene.camera.eye': {{ x: 0.05, y: -2.3, z: 0.1 }} }};
            }} else if (mode === 'topdown') {{
                update = {{ 'scene.camera.eye': {{ x: 0.0, y: 0.0, z: 2.2 }} }};
            }}
            Plotly.relayout('plot-div', update);
        }}
    </script>
</body>
</html>
"""

out_html = os.path.join(project_dir, "台灣3D地震空間構造建模.html")
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"已產出互動式 3D 網頁模型: {out_html}")
