# -*- coding: utf-8 -*-
"""
重新繪製台灣各縣市在地化地震危害與未來機率預測圖
=====================================================
基於 1994–2026 年（32.58 年、873,145 筆紀錄）地震目錄與縣市感震生活圈統計結果，
進行全新視覺化重繪：
1. 解決字型與符號相容性，徹底消除文字重疊（以專屬指引箭頭延伸小型都會區）。
2. 四子圖高解析度圖版（10年M5機率地圖、30年M6強震地圖、複合危害雙指標長條圖、三大構造群演化曲線）。
3. 輸出超高解析度靜態地圖 PNG 與現代化 Web 互動地圖 HTML。
"""

import os
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

gpkg_paths = [
    os.path.join(project_dir, "圖資", "TW_country_pop_TM2.gpkg"),
    r"D:\JIMMY CHEN\地理資訊系統參考資料-20260707T054908Z-3-001\地理資訊系統參考資料\TW_country_pop_TM2.gpkg"
]
gpkg_path = next((p for p in gpkg_paths if os.path.exists(p)), gpkg_paths[0])

CSV_PATH = os.path.join(project_dir, "數據", "台灣各縣市未來地震機率與危害度分析表.csv")
CATALOG_PATH = os.path.join(project_dir, "數據", "全台灣地震彙整目錄_1994_2026.csv")
if not os.path.exists(CATALOG_PATH):
    import glob
    raw_files = sorted(glob.glob(os.path.join(project_dir, "地震目錄", "GDMS_*.csv")))
    dfs = [pd.read_csv(f, usecols=['date', 'time', 'lat', 'lon', 'depth', 'ML']) for f in raw_files]
    df_cat = pd.concat(dfs, ignore_index=True)
    df_cat['ML'] = pd.to_numeric(df_cat['ML'], errors='coerce')
    df_cat['lat'] = pd.to_numeric(df_cat['lat'], errors='coerce')
    df_cat['lon'] = pd.to_numeric(df_cat['lon'], errors='coerce')
    df_cat['depth'] = pd.to_numeric(df_cat['depth'], errors='coerce')
    df_cat = df_cat.dropna(subset=['lat', 'lon', 'ML', 'depth'])
else:
    df_cat = pd.read_csv(CATALOG_PATH)
    df_cat['ML'] = pd.to_numeric(df_cat['ML'], errors='coerce')
    df_cat['lat'] = pd.to_numeric(df_cat['lat'], errors='coerce')
    df_cat['lon'] = pd.to_numeric(df_cat['lon'], errors='coerce')

OUTPUT_PNG = os.path.join(project_dir, "台灣各縣市未來地震機率與危害預測圖.png")
OUTPUT_HTML = os.path.join(project_dir, "台灣各縣市未來地震機率與危害預測互動地圖.html")

def redraw_county_hazard_charts():
    print("[1/3] 讀取分析數據與縣市幾何圖資...")
    df_res = pd.read_csv(CSV_PATH)
    gdf = gpd.read_file(gpkg_path).to_crs(epsg=4326)
    
    # 合併屬性
    gdf_m = gdf.merge(df_res, left_on='C_Name', right_on='縣市名稱', how='left')
    
    # 台灣本島 19 縣市
    mainland_names = [
        '基隆市', '臺北市', '新北市', '桃園市', '新竹市', '新竹縣', '苗栗縣', 
        '臺中市', '彰化縣', '南投縣', '雲林縣', '嘉義市', '嘉義縣', 
        '臺南市', '高雄市', '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣'
    ]
    gdf_tw = gdf_m[gdf_m['C_Name'].isin(mainland_names)].copy()
    df_tw = df_res[df_res['縣市名稱'].isin(mainland_names)].copy()
    
    print("[2/3] 開始繪製全新高解析度 4 子圖看板...")
    fig = plt.figure(figsize=(24, 20), dpi=300, facecolor='#fafafa')
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.22, left=0.06, right=0.96, top=0.93, bottom=0.06)
    
    # 標籤擺放字典 (針對各縣市最佳化標註座標與指向線)
    # 格式: c_name: (label_x, label_y, has_arrow)
    label_layout = {
        '花蓮縣': (121.42, 23.75, False),
        '臺東縣': (121.05, 22.85, False),
        '南投縣': (120.95, 23.83, False),
        '宜蘭縣': (121.65, 24.60, False),
        '屏東縣': (120.65, 22.38, False),
        '高雄市': (120.68, 22.95, False),
        '臺南市': (120.30, 23.15, False),
        '嘉義縣': (120.60, 23.45, False),
        '雲林縣': (120.35, 23.70, False),
        '臺中市': (120.85, 24.25, False),
        '彰化縣': (120.45, 24.00, False),
        '苗栗縣': (120.88, 24.50, False),
        '新竹縣': (121.18, 24.68, False),
        '桃園市': (121.25, 24.85, False),
        '新北市': (121.55, 24.95, False),
        # 小型都會區採用引線指引
        '臺北市': (121.20, 25.28, True),
        '基隆市': (121.95, 25.25, True),
        '新竹市': (120.60, 24.88, True),
        '嘉義市': (120.15, 23.50, True),
    }

    # ----------------------------------------------------
    # 子圖 A (左上): 未來 10 年內發生 ML >= 5.0 中強震機率地圖
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f4f7f9')
    ax1.set_title("(A) 全臺 19 縣市生活圈「未來 10 年內發生 ML >= 5.0 中強震」破裂機率與回歸週期", 
                  fontsize=13, fontweight='bold', pad=12, color='#1a1a1a')
    
    # 繪製底圖多邊形
    cmap1 = plt.get_cmap('YlOrRd')
    norm1 = mcolors.Normalize(vmin=0, vmax=100)
    
    gdf_tw.plot(
        column='未來10年_M5機率(%)',
        ax=ax1,
        cmap=cmap1,
        norm=norm1,
        edgecolor='#444444',
        linewidth=0.8,
        legend=True,
        legend_kwds={
            'label': "未來 10 年內 ML >= 5.0 破裂機率 (%)", 
            'orientation': "horizontal", 
            'shrink': 0.65, 
            'pad': 0.04,
            'aspect': 25
        }
    )
    
    # 加上各縣市標註卡片
    for idx, row in gdf_tw.iterrows():
        c_name = row['C_Name']
        p5 = row['未來10年_M5機率(%)']
        ret5 = row['平均回歸週期_M5(年)']
        rep_pt = row.geometry.representative_point()
        
        if c_name in label_layout:
            lx, ly, use_arrow = label_layout[c_name]
        else:
            lx, ly, use_arrow = rep_pt.x, rep_pt.y, False
            
        ret_str = f"{ret5:.1f}年" if ret5 >= 1.0 else f"{ret5*12:.0f}個月" if not np.isnan(ret5) else "罕見"
        card_text = f"{c_name}\n{p5:.0f}% ({ret_str})"
        
        box_bg = '#111111' if p5 >= 75 else '#ffffff'
        text_fg = '#ffffff' if p5 >= 75 else '#111111'
        
        if use_arrow:
            ax1.annotate(
                card_text,
                xy=(rep_pt.x, rep_pt.y),
                xytext=(lx, ly),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_fg,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                bbox=dict(boxstyle='round,pad=0.25', facecolor=box_bg, edgecolor='#666666', alpha=0.9, lw=0.6)
            )
        else:
            ax1.text(
                lx, ly, card_text,
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_fg,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=box_bg, edgecolor='none', alpha=0.8)
            )

    ax1.set_xlim(119.8, 122.3)
    ax1.set_ylim(21.8, 25.4)
    ax1.set_xlabel("經度 (°E)", fontsize=10)
    ax1.set_ylabel("緯度 (°N)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#bbbbbb')

    # ----------------------------------------------------
    # 子圖 B (右上): 未來 30 年內發生 ML >= 6.0 災害型強震機率地圖
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#f4f7f9')
    ax2.set_title("(B) 全臺 19 縣市生活圈「未來 30 年內發生 ML >= 6.0 災害型強震」機率與歷史強震分佈", 
                  fontsize=13, fontweight='bold', pad=12, color='#1a1a1a')
    
    cmap2 = plt.get_cmap('magma_r')
    norm2 = mcolors.Normalize(vmin=0, vmax=100)
    
    gdf_tw.plot(
        column='未來30年_M6機率(%)',
        ax=ax2,
        cmap=cmap2,
        norm=norm2,
        edgecolor='#444444',
        linewidth=0.8,
        legend=True,
        legend_kwds={
            'label': "未來 30 年內 ML >= 6.0 破裂機率 (%) (TEM標準週期)", 
            'orientation': "horizontal", 
            'shrink': 0.65, 
            'pad': 0.04,
            'aspect': 25
        }
    )
    
    # 標繪歷年 ML >= 6.0 震央光暈散點
    m6_events = df_cat[df_cat['ML'] >= 6.0]
    ax2.scatter(
        m6_events['lon'], m6_events['lat'],
        s=m6_events['ML']**2 * 0.9,
        facecolors='#1a53ff', edgecolors='#ffffff',
        linewidths=0.7, alpha=0.55, zorder=5,
        label=f"歷史 ML >= 6.0 震央 ({len(m6_events)} 次)"
    )
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)

    for idx, row in gdf_tw.iterrows():
        c_name = row['C_Name']
        p6 = row['未來30年_M6機率(%)']
        rep_pt = row.geometry.representative_point()
        
        if c_name in label_layout:
            lx, ly, use_arrow = label_layout[c_name]
        else:
            lx, ly, use_arrow = rep_pt.x, rep_pt.y, False
            
        card_text = f"{c_name}\n{p6:.0f}%"
        box_bg = '#111111' if p6 >= 70 else '#ffffff'
        text_fg = '#ffffff' if p6 >= 70 else '#111111'
        
        if use_arrow:
            ax2.annotate(
                card_text,
                xy=(rep_pt.x, rep_pt.y),
                xytext=(lx, ly),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_fg,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                bbox=dict(boxstyle='round,pad=0.25', facecolor=box_bg, edgecolor='#666666', alpha=0.9, lw=0.6)
            )
        else:
            ax2.text(
                lx, ly, card_text,
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_fg,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=box_bg, edgecolor='none', alpha=0.8)
            )

    ax2.set_xlim(119.8, 122.3)
    ax2.set_ylim(21.8, 25.4)
    ax2.set_xlabel("經度 (°E)", fontsize=10)
    ax2.set_ylabel("緯度 (°N)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#bbbbbb')

    # ----------------------------------------------------
    # 子圖 C (左下): 全台 19 縣市地震風險複合矩陣 (機率 + 極淺層比例)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#ffffff')
    ax3.set_title("(C) 臺灣 19 縣市地震複合危害度：未來 10 年破裂機率 與 極淺層震源(<15km)致命性", 
                  fontsize=13, fontweight='bold', pad=12, color='#1a1a1a')
    
    sorted_df = df_tw.sort_values(by="未來10年_M5機率(%)", ascending=True)
    y_pos = np.arange(len(sorted_df))
    probs = sorted_df['未來10年_M5機率(%)'].values
    names = sorted_df['縣市名稱'].values
    shallow = sorted_df['極淺層震源比例(<15km, %)'].values
    ret_m5 = sorted_df['平均回歸週期_M5(年)'].values
    
    # 依破裂機率著色長條
    bar_colors = []
    for p in probs:
        if p >= 95: bar_colors.append('#b30000')    # 極高
        elif p >= 70: bar_colors.append('#e34a33')  # 高
        elif p >= 40: bar_colors.append('#fdbb84')  # 中高
        else: bar_colors.append('#74a9cf')          # 中低
        
    bars = ax3.barh(y_pos, probs, color=bar_colors, alpha=0.88, height=0.62, label='未來10年 ML>=5.0 機率(%)')
    
    # 雙軸標註：右側 Y 軸繪製極淺層比例散點線
    ax3_twin = ax3.twiny()
    ax3_twin.plot(shallow, y_pos, color='#6a0dad', marker='D', markersize=6, linestyle='--', linewidth=1.5,
                  label='極淺層震源佔比 (<15km, %)')
    ax3_twin.set_xlim(0, 100)
    ax3_twin.set_xlabel("極淺層震源比例 (%) [紫色菱形線]", fontsize=9.5, color='#6a0dad', fontweight='bold')
    ax3_twin.tick_params(axis='x', labelcolor='#6a0dad')
    
    # 標注文欄資訊
    for i, (p, sh, ret) in enumerate(zip(probs, shallow, ret_m5)):
        ret_txt = f"{ret:.1f}年一遇" if ret >= 1.0 else f"{ret*12:.0f}個月一遇" if not np.isnan(ret) else "極罕見"
        ax3.text(p + 1.2, i, f"{p:.0f}% ({ret_txt} | 淺層{sh:.0f}%)",
                 va='center', fontsize=8, color='#222222', fontweight='bold')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=9.5)
    ax3.set_xlabel("未來 10 年內發生 ML >= 5.0 中強震機率 (%)", fontsize=10)
    ax3.set_xlim(0, 130)
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # 合併圖例
    patch1 = mpatches.Patch(color='#b30000', label='機率 ≥ 95% (極高風險區)')
    patch2 = mpatches.Patch(color='#e34a33', label='機率 70-94% (高風險區)')
    patch3 = mpatches.Patch(color='#74a9cf', label='機率 < 40% (較低頻區)')
    ax3.legend(handles=[patch1, patch2, patch3], loc='lower right', fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # 子圖 D (右下): 三大構造地質體系未來時間累積強震機率曲線
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#ffffff')
    ax4.set_title("(D) 三大地質構造體系焦點縣市：未來時間累積強震破裂機率演化曲線 P(Δt)", 
                  fontsize=13, fontweight='bold', pad=12, color='#1a1a1a')
    
    t_span = np.linspace(0.1, 30, 200)
    
    # 分群配置
    groups = [
        # (縣市, 色碼, 線型, 線寬, 體系說明)
        ('花蓮縣', '#d73027', '-', 2.5, '東部板塊聚合前線'),
        ('臺東縣', '#f46d43', '-', 2.0, '東部板塊聚合前線'),
        ('宜蘭縣', '#fdae61', '-', 2.0, '東部板塊聚合前線'),
        ('南投縣', '#1a9850', '--', 2.2, '西部前陸逆衝盲斷層帶'),
        ('嘉義縣', '#66bd63', '--', 2.2, '西部前陸逆衝盲斷層帶'),
        ('臺南市', '#a6d96a', '--', 2.2, '西部前陸逆衝盲斷層帶'),
        ('高雄市', '#d9ef8b', '--', 1.8, '西部前陸逆衝盲斷層帶'),
        ('臺中市', '#4575b4', '-.', 1.8, '西部前陸逆衝盲斷層帶'),
        ('新北市', '#74add1', ':', 1.8, '北部盆地都會區 (場址放大效應)'),
        ('臺北市', '#313695', ':', 1.8, '北部盆地都會區 (場址放大效應)'),
    ]
    
    for c_name, col, ls, lw, grp in groups:
        c_row = df_tw[df_tw['縣市名稱'] == c_name]
        if len(c_row) > 0:
            lam = c_row['年均發生率_M6(次/年)'].values[0]
            if lam > 0:
                p_curve = (1.0 - np.exp(-lam * t_span)) * 100.0
                ax4.plot(t_span, p_curve, label=f"[{grp.split(' ')[0]}] {c_name} (λ={lam:.2f}/年)",
                         color=col, linestyle=ls, linewidth=lw)
            else:
                ax4.plot(t_span, np.zeros_like(t_span), label=f"[{grp.split(' ')[0]}] {c_name} (近32年本地無M6)",
                         color=col, linestyle=ls, linewidth=lw, alpha=0.4)

    ax4.set_xlabel("未來預測時間跨度 Δt (年)", fontsize=10)
    ax4.set_ylabel("發生至少一次 ML >= 6.0 災害型強震機率 (%)", fontsize=10)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(-2, 105)
    
    # 參考時間線
    ax4.axvline(1.0, color='#888888', linestyle=':', alpha=0.7)
    ax4.text(1.1, 15, "1年", fontsize=8.5, color='#555555', fontweight='bold')
    ax4.axvline(5.0, color='#888888', linestyle=':', alpha=0.7)
    ax4.text(5.1, 15, "5年", fontsize=8.5, color='#555555', fontweight='bold')
    ax4.axvline(10.0, color='#888888', linestyle=':', alpha=0.7)
    ax4.text(10.1, 15, "10年", fontsize=8.5, color='#555555', fontweight='bold')
    ax4.axvline(30.0, color='#888888', linestyle=':', alpha=0.7)
    ax4.text(27.8, 15, "30年(TEM)", fontsize=8.5, color='#555555', fontweight='bold')
    
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right', fontsize=8.2, framealpha=0.92, ncol=1)

    plt.suptitle("臺灣各縣市在地化地震危害度與未來破裂機率預測綜合分析看板\n(基於中央氣象署 1994–2026 年目錄，行政轄區 20km 感震生活圈空間聯結與帕松極值過程)",
                 fontsize=15, fontweight='bold', y=0.97, color='#111111')
    
    plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[3/3] 靜態看板重繪完成：{OUTPUT_PNG}")

def generate_interactive_html():
    print("[+] 生成 Web 現代化互動地圖 (HTML)...")
    df_res = pd.read_csv(CSV_PATH)
    gdf = gpd.read_file(gpkg_path).to_crs(epsg=4326)
    
    mainland_names = [
        '基隆市', '臺北市', '新北市', '桃園市', '新竹市', '新竹縣', '苗栗縣', 
        '臺中市', '彰化縣', '南投縣', '雲林縣', '嘉義市', '嘉義縣', 
        '臺南市', '高雄市', '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣'
    ]
    gdf_m = gdf[gdf['C_Name'].isin(mainland_names)].merge(df_res, left_on='C_Name', right_on='縣市名稱', how='left')
    
    geojson_data = json.loads(gdf_m.to_json())
    
    # 準備歷史 M6 震央資料
    m6_events = df_cat[df_cat['ML'] >= 6.0][['lat', 'lon', 'ML', 'depth']].to_dict(orient='records')
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>臺灣各縣市未來地震危害度與破裂機率互動地圖 (1994–2026)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif; background: #121212; color: #fff; }}
        #header {{ padding: 12px 20px; background: #1e1e1e; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
        #header h1 {{ margin: 0; font-size: 1.15rem; color: #f0f0f0; }}
        #header p {{ margin: 3px 0 0 0; font-size: 0.8rem; color: #999; }}
        #map {{ height: calc(100vh - 65px); width: 100vw; background: #18191a; }}
        .info-card {{ background: rgba(30, 30, 30, 0.92); backdrop-filter: blur(6px); border: 1px solid #444; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; line-height: 1.5; color: #eee; box-shadow: 0 4px 16px rgba(0,0,0,0.5); }}
        .info-card h4 {{ margin: 0 0 6px 0; font-size: 1.05rem; color: #ff6b6b; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-bottom: 6px; }}
        .badge-red {{ background: #d32f2f; color: #fff; }}
        .badge-orange {{ background: #f57c00; color: #fff; }}
        .badge-blue {{ background: #1976d2; color: #fff; }}
        .legend {{ background: rgba(25, 25, 25, 0.9); padding: 10px 14px; border-radius: 6px; border: 1px solid #444; font-size: 0.8rem; line-height: 1.5; }}
        .legend i {{ width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.85; border-radius: 3px; }}
        .controls {{ position: absolute; top: 80px; right: 20px; z-index: 1000; background: rgba(30,30,30,0.92); border: 1px solid #444; padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; }}
        .controls label {{ cursor: pointer; display: block; margin: 4px 0; }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h1>臺灣各縣市未來地震危害度與破裂機率互動式評估地圖</h1>
            <p>資料來源：中央氣象署 1994–2026 年目錄（共 873,145 筆）｜ 20km 感震生活圈空間聯結模型</p>
        </div>
    </div>
    
    <div class="controls">
        <b>切換機率圖層：</b>
        <label><input type="radio" name="layerMode" value="m5_10y" checked onchange="updateChoropleth()"> 未來 10 年 ML ≥ 5.0 中強震機率</label>
        <label><input type="radio" name="layerMode" value="m6_30y" onchange="updateChoropleth()"> 未來 30 年 ML ≥ 6.0 災害強震機率</label>
        <label><input type="checkbox" id="showM6Epicenters" checked onchange="toggleEpicenters()"> 顯示歷史 ML ≥ 6.0 震央</label>
    </div>

    <div id="map"></div>

    <script>
        const geojsonData = {json.dumps(geojson_data, ensure_ascii=False)};
        const m6Epicenters = {json.dumps(m6_events)};

        const map = L.map('map').setView([23.8, 121.0], 8);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; CartoDB & OSM contributors',
            maxZoom: 14
        }}).addTo(map);

        function getColorM5(d) {{
            return d >= 95 ? '#800026' :
                   d >= 80 ? '#BD0026' :
                   d >= 65 ? '#E31A1C' :
                   d >= 50 ? '#FC4E2A' :
                   d >= 35 ? '#FD8D3C' :
                   d >= 20 ? '#FEB24C' : '#FED976';
        }}

        function getColorM6(d) {{
            return d >= 90 ? '#49006a' :
                   d >= 75 ? '#7a0177' :
                   d >= 60 ? '#ae017e' :
                   d >= 40 ? '#dd3497' :
                   d >= 20 ? '#f768a1' :
                   d > 0   ? '#fa9fb5' : '#444444';
        }}

        let geojsonLayer;
        let epicenterLayer = L.layerGroup().addTo(map);

        function style(feature) {{
            const mode = document.querySelector('input[name="layerMode"]:checked').value;
            const val = mode === 'm5_10y' ? feature.properties['未來10年_M5機率(%)'] : feature.properties['未來30年_M6機率(%)'];
            const color = mode === 'm5_10y' ? getColorM5(val) : getColorM6(val);
            return {{
                fillColor: color,
                weight: 1.2,
                opacity: 1,
                color: '#222',
                fillOpacity: 0.78
            }};
        }}

        function highlightFeature(e) {{
            const layer = e.target;
            layer.setStyle({{
                weight: 2.5,
                color: '#fff',
                fillOpacity: 0.95
            }});
            layer.bringToFront();
        }}

        function resetHighlight(e) {{
            geojsonLayer.resetStyle(e.target);
        }}

        function onEachFeature(feature, layer) {{
            const p = feature.properties;
            const popupContent = `
                <div class="info-card">
                    <h4>${{p['C_Name']}} 感震生活圈</h4>
                    <div><b>全臺危害度排名：</b> 第 ${{p['排名'] || '—'}} 名</div>
                    <div><b>32年 M≥5.0 累積：</b> ${{p['生活圈中強震數(ML>=5.0)']}} 次 (約 ${{p['平均回歸週期_M5(年)']}} 年一遇)</div>
                    <div><b>32年 M≥6.0 累積：</b> ${{p['生活圈強震數(ML>=6.0)']}} 次</div>
                    <div><b>震源深度中位數：</b> ${{p['平均震源深度(km)']}} km</div>
                    <div><b>極淺層比例(&lt;15km)：</b> <span style="color:#ffb74d;font-weight:bold">${{p['極淺層震源比例(<15km, %)']}}%</span></div>
                    <hr style="border:0;border-top:1px solid #444;margin:8px 0;">
                    <div><b>未來 1 年 M≥5.0 機率：</b> <span style="color:#ff5252">${{p['未來1年_M5機率(%)']}}%</span></div>
                    <div><b>未來 10 年 M≥5.0 機率：</b> <span style="color:#ff1744">${{p['未來10年_M5機率(%)']}}%</span></div>
                    <div><b>未來 10 年 M≥6.0 強震機率：</b> <span style="color:#e040fb">${{p['未來10年_M6機率(%)']}}%</span></div>
                    <div><b>未來 30 年 M≥6.0 強震機率：</b> <span style="color:#d500f9;font-weight:bold">${{p['未來30年_M6機率(%)']}}%</span></div>
                </div>
            `;
            layer.bindPopup(popupContent);
            layer.on({{
                mouseover: highlightFeature,
                mouseout: resetHighlight
            }});
        }}

        function updateChoropleth() {{
            if (geojsonLayer) map.removeLayer(geojsonLayer);
            geojsonLayer = L.geoJson(geojsonData, {{
                style: style,
                onEachFeature: onEachFeature
            }}).addTo(map);
        }}

        function plotEpicenters() {{
            epicenterLayer.clearLayers();
            m6Epicenters.forEach(e => {{
                L.circleMarker([e.lat, e.lon], {{
                    radius: Math.pow(e.ML, 1.4),
                    fillColor: '#00d2ff',
                    color: '#fff',
                    weight: 0.8,
                    opacity: 0.9,
                    fillOpacity: 0.5
                }}).bindTooltip(`規模 ML: ${{e.ML}}<br>深度: ${{e.depth}} km`).addTo(epicenterLayer);
            }});
        }}

        function toggleEpicenters() {{
            const checked = document.getElementById('showM6Epicenters').checked;
            if (checked) map.addLayer(epicenterLayer);
            else map.removeLayer(epicenterLayer);
        }}

        updateChoropleth();
        plotEpicenters();
    </script>
</body>
</html>
"""
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[+] 互動地圖已輸出：{OUTPUT_HTML}")

if __name__ == '__main__':
    redraw_county_hazard_charts()
    generate_interactive_html()
