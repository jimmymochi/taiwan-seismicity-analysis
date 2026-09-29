# -*- coding: utf-8 -*-
"""
臺灣地震時空預測管線 - 模組一：2.2 km 空間網格生成器 (grid_generator.py)
========================================================================
依據任務規範：
1. 空間範圍：臺灣及周邊海域 (約 Lon 119.0°E 至 123.0°E，Lat 21.5°N 至 25.5°N)
2. 空間解析度：2.2 km × 2.2 km 正方網格
3. 投影座標系：臺灣標準 TWD97 / TM2 zone 121 (EPSG:3826)
4. 輸出產物：
   - 網格查找表 CSV: 數據/臺灣2.2km空間網格查找表.csv
   - 網格 GeoPackage / Parquet / GeoJSON 供 GIS 及後續特徵工程空間查詢
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

OUTPUT_DIR = os.path.join(project_dir, "數據")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "臺灣2.2km空間網格查找表.csv")
OUTPUT_GPKG = os.path.join(OUTPUT_DIR, "臺灣2.2km空間網格圖資.gpkg")

# 定義地理邊界 (WGS84)
LON_MIN, LON_MAX = 119.0, 123.0
LAT_MIN, LAT_MAX = 21.5, 25.5
GRID_SIZE_METERS = 2200.0  # 2.2 km

def generate_grid():
    print("[1/3] 轉換經緯度範圍至 TWD97 / TM2 (EPSG:3826)...")
    # 建立邊界多邊形並投影至 EPSG:3826
    bbox_geom = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    gdf_bounds = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326").to_crs(epsg=3826)
    minx, miny, maxx, maxy = gdf_bounds.total_bounds
    print(f"  TM2 邊界: X=[{minx:,.0f}, {maxx:,.0f}], Y=[{miny:,.0f}, {maxy:,.0f}] 米")

    print("[2/3] 生成 2.2 km × 2.2 km 正方形網格拓撲...")
    x_coords = np.arange(minx, maxx, GRID_SIZE_METERS)
    y_coords = np.arange(miny, maxy, GRID_SIZE_METERS)
    print(f"  X 方向格數: {len(x_coords)}, Y 方向格數: {len(y_coords)}, 總理論格數: {len(x_coords)*len(y_coords):,}")

    grid_records = []
    grid_geoms = []
    grid_id = 0

    for x in x_coords:
        for y in y_coords:
            grid_poly = box(x, y, x + GRID_SIZE_METERS, y + GRID_SIZE_METERS)
            cx = x + GRID_SIZE_METERS / 2.0
            cy = y + GRID_SIZE_METERS / 2.0
            
            grid_records.append({
                'grid_id': grid_id,
                'x_min': x,
                'x_max': x + GRID_SIZE_METERS,
                'y_min': y,
                'y_max': y + GRID_SIZE_METERS,
                'x_center': cx,
                'y_center': cy,
            })
            grid_geoms.append(grid_poly)
            grid_id += 1

    gdf_grid = gpd.GeoDataFrame(grid_records, geometry=grid_geoms, crs="EPSG:3826")
    
    print("[3/3] 反向投影計算各網格中心之 WGS84 經緯度座標...")
    gdf_wgs = gdf_grid.to_crs(epsg=4326)
    centroids_wgs = gdf_wgs.geometry.centroid
    gdf_grid['lon_center'] = centroids_wgs.x.round(5)
    gdf_grid['lat_center'] = centroids_wgs.y.round(5)
    
    # 輸出 CSV 輕量查找表
    df_lookup = pd.DataFrame(gdf_grid.drop(columns='geometry'))
    df_lookup.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  網格查找表已儲存至: {OUTPUT_CSV} (共 {len(df_lookup):,} 格)")

    return gdf_grid, df_lookup

if __name__ == '__main__':
    generate_grid()
