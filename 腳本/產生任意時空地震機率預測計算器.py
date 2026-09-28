# -*- coding: utf-8 -*-
"""
產生臺灣任意時空地震機率預測計算器 (HTML)
=====================================================
將中央氣象署 1994–2026 年（32.58年）8,252 筆中強震數據（ML >= 4.0）內嵌至
獨立 HTML/JavaScript 應用中。
使用者可在瀏覽器地圖上：
1. 自由拉框選取「任意經緯度矩形」或點擊設置「任意中心圓形半徑」。
2. 設定「任意未來時間跨度」（如未來 7 天、30 天、1 年、5 年、10 年等）。
3. 即時毫秒級演算並動態繪製出該特定區域的破裂機率、G-R 定律擬合與預期次數。
"""

import os
import glob
import json
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

DATA_DIR = os.path.join(project_dir, "地震目錄")
OUTPUT_HTML = os.path.join(project_dir, "臺灣任意時空地震機率預測計算器.html")

def build_calculator_html():
    print("[1/2] 讀取並緊湊壓縮 ML >= 4.0 地震數據 (8,252筆)...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "GDMS_*.csv")))
    dfs = [pd.read_csv(f, usecols=['date', 'lat', 'lon', 'depth', 'ML']) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['ML'] = pd.to_numeric(df['ML'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon', 'ML', 'depth'])
    
    df_m4 = df[df['ML'] >= 4.0].sort_values('date').reset_index(drop=True)
    
    compact_records = []
    for idx, row in df_m4.iterrows():
        compact_records.append([
            round(float(row['lat']), 3),
            round(float(row['lon']), 3),
            round(float(row['ML']), 1),
            round(float(row['depth']), 1)
        ])
        
    data_json = json.dumps(compact_records, separators=(',', ':'))
    print(f"壓縮後 JSON 大小: {len(data_json)/1024:.1f} KB")

    print("[2/2] 生成臺灣任意時空地震機率即時預測計算器 (HTML)...")
    
    # 使用一般字串模板取代 f-string 以免與 JS 的 ${} 衝突
    template = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>臺灣特定時空地震機率即時預測計算系統 (1994–2026)</title>
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        * { box-sizing: border-box; }
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif; background: #0f1115; color: #e1e4ea; display: flex; height: 100vh; overflow: hidden; }
        #sidebar { width: 440px; min-width: 440px; background: #161922; border-right: 1px solid #282d3d; display: flex; flex-direction: column; overflow-y: auto; padding: 18px 20px; }
        #map-container { flex: 1; position: relative; height: 100vh; }
        #map { width: 100%; height: 100%; background: #090a0d; }

        h1 { font-size: 1.22rem; margin: 0 0 4px 0; color: #64b5f6; display: flex; align-items: center; gap: 8px; }
        .subtitle { font-size: 0.78rem; color: #8b949e; margin-bottom: 16px; border-bottom: 1px solid #282d3d; padding-bottom: 10px; }

        .control-group { background: #1f2430; border: 1px solid #2c3345; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; }
        .control-title { font-size: 0.85rem; font-weight: bold; color: #ffb74d; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }

        .btn-group { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
        button.btn { background: #2a3142; border: 1px solid #3d4760; color: #e1e4ea; padding: 6px 12px; font-size: 0.8rem; border-radius: 5px; cursor: pointer; transition: all 0.2s; }
        button.btn:hover { background: #3b455d; border-color: #64b5f6; }
        button.btn.active { background: #1976d2; border-color: #64b5f6; color: #fff; font-weight: bold; }

        select, input[type="number"], input[type="range"] { width: 100%; background: #12151c; border: 1px solid #3d4760; color: #fff; padding: 7px 10px; border-radius: 5px; font-size: 0.82rem; margin-top: 4px; }
        .range-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-top: 6px; }
        .range-val { color: #4fc3f7; font-weight: bold; }

        .stat-badge { display: inline-block; padding: 3px 8px; background: #232a3b; border-radius: 4px; font-size: 0.78rem; color: #90caf9; margin-right: 6px; margin-bottom: 4px; }

        .result-box { background: #1a2130; border: 1px solid #294366; border-radius: 8px; padding: 14px; margin-top: 4px; }
        .res-highlight { font-size: 1.45rem; font-weight: bold; color: #ff5252; margin: 4px 0 8px 0; }
        .res-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.78rem; }
        .res-item { background: #121722; padding: 6px 8px; border-radius: 4px; border: 1px solid #252d3d; }
        .res-item b { color: #b0bec5; display: block; font-size: 0.72rem; }

        .chart-box { margin-top: 14px; background: #1f2430; border-radius: 8px; padding: 12px; border: 1px solid #2c3345; }
        .chart-box h4 { margin: 0 0 8px 0; font-size: 0.82rem; color: #ffb74d; }

        .floating-tip { position: absolute; top: 16px; right: 16px; z-index: 1000; background: rgba(22, 25, 34, 0.92); border: 1px solid #3d4760; backdrop-filter: blur(6px); padding: 10px 16px; border-radius: 6px; font-size: 0.82rem; box-shadow: 0 4px 20px rgba(0,0,0,0.5); pointer-events: none; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>臺灣時空地震機率預測系統</h1>
        <div class="subtitle">基於 1994–2026 年（32.58年）873,145 筆觀測目錄 ｜ 帕松極值過程與 G-R 規律</div>

        <!-- 1. 空間選擇 -->
        <div class="control-group">
            <div class="control-title">
                <span>① 定義預測空間區域</span>
                <span id="region-status" style="font-size:0.75rem;color:#81c784;">請於地圖選取</span>
            </div>
            
            <div style="font-size:0.78rem;color:#aaa;margin-bottom:8px;">快捷特定構造斷層帶：</div>
            <div class="btn-group">
                <button class="btn active" onclick="setPresetRegion('hualien')">花蓮近海破裂帶</button>
                <button class="btn" onclick="setPresetRegion('meishan')">嘉南梅山斷層帶</button>
                <button class="btn" onclick="setPresetRegion('yilan')">宜蘭沖繩海槽區</button>
                <button class="btn" onclick="setPresetRegion('chilungpu')">中部車籠埔破裂區</button>
                <button class="btn" onclick="setPresetRegion('taipei')">雙北山腳斷層區</button>
                <button class="btn" onclick="setPresetRegion('hengchun')">恆春南端外海區</button>
            </div>
            <div style="font-size:0.72rem;color:#78909c;margin-top:4px;">
                💡 亦可直接使用地圖左上角工具箱：<b>拉框選取矩形</b> 或 <b>繪製圓形搜索區</b>。
            </div>
        </div>

        <!-- 2. 時間跨度 -->
        <div class="control-group">
            <div class="control-title">② 設定未來預測時間跨度 (Δt)</div>
            <div class="btn-group" id="time-btns">
                <button class="btn" onclick="setTimeSpan(7, '7 天')">7 天</button>
                <button class="btn active" onclick="setTimeSpan(30, '30 天 (1個月)')">30 天</button>
                <button class="btn" onclick="setTimeSpan(90, '90 天 (3個月)')">90 天</button>
                <button class="btn" onclick="setTimeSpan(365, '1 年')">1 年</button>
                <button class="btn" onclick="setTimeSpan(365*5, '5 年')">5 年</button>
                <button class="btn" onclick="setTimeSpan(365*10, '10 年')">10 年</button>
                <button class="btn" onclick="setTimeSpan(365*30, '30 年')">30 年</button>
            </div>
            <div class="range-row">
                <span>自選天數：</span>
                <span class="range-val" id="time-val-display">30 天 (1個月)</span>
            </div>
            <input type="range" id="time-slider" min="1" max="10950" value="30" oninput="onSliderTimeChange(this.value)">
        </div>

        <!-- 3. 規模門檻 -->
        <div class="control-group">
            <div class="control-title">③ 設定預測地震規模門檻 (ML ≥ M0)</div>
            <div class="range-row">
                <span>目標破裂規模：</span>
                <span class="range-val" id="mag-val-display" style="font-size:1.1rem;color:#ff7043;">ML ≥ 5.0</span>
            </div>
            <input type="range" id="mag-slider" min="4.0" max="7.0" step="0.1" value="5.0" oninput="onSliderMagChange(this.value)">
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#78909c;margin-top:2px;">
                <span>ML 4.0 (普遍有感)</span>
                <span>ML 5.0 (中強震)</span>
                <span>ML 6.0 (破壞強震)</span>
                <span>ML 7.0 (災難大震)</span>
            </div>
        </div>

        <!-- 4. 預測計算結果面板 -->
        <div class="result-box">
            <div style="font-size:0.8rem;color:#b0bec5;">【特定時空破裂預測結果】</div>
            <div class="res-highlight" id="prob-result-text">計算中...</div>
            <div id="prob-subtext" style="font-size:0.78rem;color:#cfd8dc;margin-bottom:8px;"></div>
            
            <div class="res-grid">
                <div class="res-item">
                    <b>預期破裂事件數 (μ = λ·Δt)</b>
                    <span id="exp-events-val" style="color:#64b5f6;font-size:1rem;font-weight:bold;">0.00 次</span>
                </div>
                <div class="res-item">
                    <b>歷史年平均發生率 (λ)</b>
                    <span id="annual-rate-val" style="color:#81c784;font-size:1rem;font-weight:bold;">0.00 次/年</span>
                </div>
                <div class="res-item">
                    <b>區域 G-R 定律 b 值 (應力指標)</b>
                    <span id="gr-b-val" style="color:#ffb74d;font-size:1rem;font-weight:bold;">0.85</span>
                </div>
                <div class="res-item">
                    <b>該規模平均回歸週期 (TR)</b>
                    <span id="ret-period-val" style="color:#ba68c8;font-size:1rem;font-weight:bold;">0.0 年</span>
                </div>
            </div>
        </div>

        <!-- 5. 動態規模-機率分佈圖表 -->
        <div class="chart-box">
            <h4>📊 該特定時空之「各規模破裂機率光譜」</h4>
            <canvas id="probChart" height="170"></canvas>
        </div>
    </div>

    <div id="map-container">
        <div class="floating-tip" id="floating-tip">
            📌 目前選定區域：<b>花蓮近海破裂帶</b> ｜ 雙擊地圖可隨時自訂點位
        </div>
        <div id="map"></div>
    </div>

    <script>
        const T_DURATION_YEARS = 32.58;
        const rawCatalog = __DATA_JSON__; // [lat, lon, ML, depth]
        
        let currentSpatialFilter = null; // func(lat, lon) -> bool
        let currentRegionName = "花蓮近海破裂帶";
        let currentTimeDays = 30;
        let currentTargetMag = 5.0;
        let drawnLayer = null;
        let probChart = null;

        // 初始化 Leaflet 地圖
        const map = L.map('map', {
            center: [23.75, 121.2],
            zoom: 8,
            zoomControl: false
        });
        L.control.zoom({ position: 'bottomright' }).addTo(map);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; CartoDB & OpenStreetMap',
            maxZoom: 14
        }).addTo(map);

        // 繪製控制項
        const drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);
        const drawControl = new L.Control.Draw({
            position: 'topleft',
            draw: {
                polyline: false,
                polygon: false,
                circlemarker: false,
                marker: false,
                rectangle: {
                    shapeOptions: { color: '#64b5f6', weight: 2, fillOpacity: 0.15 }
                },
                circle: {
                    shapeOptions: { color: '#ffb74d', weight: 2, fillOpacity: 0.15 }
                }
            },
            edit: { featureGroup: drawnItems, remove: false }
        });
        map.addControl(drawControl);

        map.on(L.Draw.Event.CREATED, function (event) {
            drawnItems.clearLayers();
            const layer = event.layer;
            drawnItems.addLayer(layer);
            
            if (event.layerType === 'rectangle') {
                const bounds = layer.getBounds();
                const minLat = bounds.getSouth(), maxLat = bounds.getNorth();
                const minLon = bounds.getWest(), maxLon = bounds.getEast();
                currentSpatialFilter = (lat, lon) => (lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon);
                currentRegionName = `自訂矩形 [${minLon.toFixed(2)}°E~${maxLon.toFixed(2)}°E, ${minLat.toFixed(2)}°N~${maxLat.toFixed(2)}°N]`;
            } else if (event.layerType === 'circle') {
                const center = layer.getLatLng();
                const radiusKm = layer.getRadius() / 1000.0;
                currentSpatialFilter = (lat, lon) => haversine(center.lat, center.lng, lat, lon) <= radiusKm;
                currentRegionName = `自訂圓形 [(${center.lng.toFixed(2)}°E, ${center.lat.toFixed(2)}°N) 半徑 ${radiusKm.toFixed(1)}km]`;
            }
            
            updatePrediction();
        });

        function haversine(lat1, lon1, lat2, lon2) {
            const R = 6371.0;
            const dLat = (lat2 - lat1) * Math.PI / 180.0;
            const dLon = (lon2 - lon1) * Math.PI / 180.0;
            const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180.0) * Math.cos(lat2*Math.PI/180.0) * Math.sin(dLon/2)**2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }

        // 預設構造區域
        const presets = {
            'hualien': { name: '花蓮近海破裂帶', bounds: [[23.75, 121.45], [24.25, 121.85]] },
            'meishan': { name: '嘉南梅山斷層帶', circle: { center: [23.48, 120.55], radiusKm: 25.0 } },
            'yilan': { name: '宜蘭沖繩海槽區', bounds: [[24.45, 121.65], [24.95, 122.25]] },
            'chilungpu': { name: '中部車籠埔破裂區', bounds: [[23.85, 120.60], [24.35, 121.10]] },
            'taipei': { name: '雙北山腳斷層區', bounds: [[24.95, 121.35], [25.25, 121.65]] },
            'hengchun': { name: '恆春南端外海區', circle: { center: [21.80, 120.70], radiusKm: 40.0 } }
        };

        function setPresetRegion(key) {
            const p = presets[key];
            currentRegionName = p.name;
            drawnItems.clearLayers();

            if (p.bounds) {
                const b = p.bounds;
                currentSpatialFilter = (lat, lon) => (lat >= b[0][0] && lat <= b[1][0] && lon >= b[0][1] && lon <= b[1][1]);
                const rect = L.rectangle(b, { color: '#64b5f6', weight: 2, fillOpacity: 0.18 }).addTo(drawnItems);
                map.fitBounds(b, { padding: [40, 40] });
            } else if (p.circle) {
                const c = p.circle;
                currentSpatialFilter = (lat, lon) => haversine(c.center[0], c.center[1], lat, lon) <= c.radiusKm;
                const circ = L.circle(c.center, { radius: c.radiusKm * 1000, color: '#ffb74d', weight: 2, fillOpacity: 0.18 }).addTo(drawnItems);
                map.fitBounds(circ.getBounds(), { padding: [40, 40] });
            }

            document.querySelectorAll('.control-group button.btn').forEach(b => b.classList.remove('active'));
            if (window.event && window.event.target) window.event.target.classList.add('active');

            updatePrediction();
        }

        function setTimeSpan(days, label) {
            currentTimeDays = days;
            document.getElementById('time-slider').value = days;
            document.getElementById('time-val-display').innerText = label;
            document.querySelectorAll('#time-btns button').forEach(b => b.classList.remove('active'));
            if (window.event && window.event.target) window.event.target.classList.add('active');
            updatePrediction();
        }

        function onSliderTimeChange(val) {
            currentTimeDays = parseInt(val);
            const label = currentTimeDays < 365 ? `${currentTimeDays} 天` : `${(currentTimeDays/365.25).toFixed(1)} 年`;
            document.getElementById('time-val-display').innerText = label;
            document.querySelectorAll('#time-btns button').forEach(b => b.classList.remove('active'));
            updatePrediction();
        }

        function onSliderMagChange(val) {
            currentTargetMag = parseFloat(val);
            document.getElementById('mag-val-display').innerText = `ML ≥ ${currentTargetMag.toFixed(1)}`;
            updatePrediction();
        }

        // 核心機率演算
        function updatePrediction() {
            document.getElementById('floating-tip').innerHTML = `📌 選定區域：<b>${currentRegionName}</b>`;
            document.getElementById('region-status').innerText = currentRegionName;

            // 1. 空間篩選
            const filtered = [];
            for (let i = 0; i < rawCatalog.length; i++) {
                const pt = rawCatalog[i]; // [lat, lon, ml, depth]
                if (currentSpatialFilter(pt[0], pt[1])) {
                    filtered.push(pt);
                }
            }

            const nTotal = filtered.length;
            // 2. 統計目標規模事件
            let nTarget = 0;
            let sumMag = 0;
            for (let i = 0; i < filtered.length; i++) {
                if (filtered[i][2] >= currentTargetMag) nTarget++;
                sumMag += filtered[i][2];
            }

            // G-R b 值概算 MLE
            let bVal = 0.85;
            if (filtered.length >= 10) {
                const meanM = sumMag / filtered.length;
                bVal = Math.log10(Math.E) / (meanM - (4.0 - 0.05));
                if (bVal < 0.5) bVal = 0.5;
                if (bVal > 1.8) bVal = 1.8;
            }

            // 年均發生率 lambda
            const lambda = nTarget / T_DURATION_YEARS;
            const deltaTYear = currentTimeDays / 365.25;
            const mu = lambda * deltaTYear;
            const probAtLeastOne = (1.0 - Math.exp(-mu)) * 100.0;
            const retPeriod = lambda > 0 ? (1.0 / lambda) : NaN;

            // 渲染介面數值
            document.getElementById('prob-result-text').innerText = `${probAtLeastOne.toFixed(2)} %`;
            document.getElementById('prob-subtext').innerText = 
                `在未來 ${currentTimeDays < 365 ? currentTimeDays + ' 天' : (currentTimeDays/365.25).toFixed(1) + ' 年'} 內，發生至少 1 次 ML ≥ ${currentTargetMag.toFixed(1)} 地震之機率`;
            
            document.getElementById('exp-events-val').innerText = `${mu.toFixed(3)} 次`;
            document.getElementById('annual-rate-val').innerText = `${lambda.toFixed(3)} 次/年`;
            document.getElementById('gr-b-val').innerText = bVal.toFixed(2);
            document.getElementById('ret-period-val').innerText = !isNaN(retPeriod) ? 
                (retPeriod >= 1 ? `${retPeriod.toFixed(1)} 年` : `${(retPeriod*12).toFixed(1)} 個月`) : "極罕見";

            // 繪製多規模機率直方圖 (ML 4.0 ~ 7.0)
            renderChart(filtered, deltaTYear);
        }

        function renderChart(filteredEvents, deltaTYear) {
            const magSteps = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0];
            const probList = [];

            magSteps.forEach(m => {
                let count = 0;
                for (let i = 0; i < filteredEvents.length; i++) {
                    if (filteredEvents[i][2] >= m) count++;
                }
                const lam = count / T_DURATION_YEARS;
                const p = (1.0 - Math.exp(-lam * deltaTYear)) * 100.0;
                probList.push(p);
            });

            const ctx = document.getElementById('probChart').getContext('2d');
            if (probChart) probChart.destroy();

            probChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: magSteps.map(m => `ML ≥ ${m.toFixed(1)}`),
                    datasets: [{
                        label: '破裂發生機率 (%)',
                        data: probList,
                        backgroundColor: probList.map(p => p > 50 ? '#ff5252' : (p > 20 ? '#ffb74d' : '#64b5f6')),
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `發生機率: ${context.parsed.y.toFixed(2)} %`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: '#2c3345' },
                            ticks: { color: '#90a4ae', callback: (v) => v + '%' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#b0bec5' }
                        }
                    }
                }
            });
        }

        // 初始啟動
        setPresetRegion('hualien');
    </script>
</body>
</html>
"""
    final_html = template.replace("__DATA_JSON__", data_json)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"計算器 HTML 已輸出至：{OUTPUT_HTML}")

if __name__ == '__main__':
    build_calculator_html()
