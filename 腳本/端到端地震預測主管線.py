# -*- coding: utf-8 -*-
"""
臺灣地震時空預測管線 - 模組六：端到端地震預測主管線 (pipeline.py)
========================================================================
依據任務規範：
本模組為整個地震時空機器學習預測系統之主調度管線 (Main Orchestrator)。
支援清晰的命令列參數 (CLI arguments)，可一鍵依序或分步執行：
1. 空間網格生成 (grid_generator.py)
2. 多尺度特徵工程提取 (feature_extractor.py)
3. 機器學習模型訓練與等張機率校準 (model_trainer.py)
4. 國際 CSEP 評估檢定 (csep_evaluator.py)
5. 五大時間尺度時空演化與實震驗證看板繪製 (plot_progression_maps.py)
"""

import os
import sys
import io

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import subprocess
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))

PYTHON_EXE = sys.executable

STEPS = [
    {
        'num': 1,
        'name': '空間離散網格建立 (2.2km × 2.2km TWD97 / EPSG:3826)',
        'script': os.path.join(script_dir, '網格生成器.py')
    },
    {
        'num': 2,
        'name': '嚴格無外洩多尺度物理特徵提取 (Purged Walk-Forward & Tectonic Features)',
        'script': os.path.join(script_dir, '特徵工程提取器.py')
    },
    {
        'num': 3,
        'name': '機器學習模型訓練與等張機率校準 (LightGBM & Isotonic Calibration)',
        'script': os.path.join(script_dir, '機器學習模型訓練與校準器.py')
    },
    {
        'num': 4,
        'name': '國際 CSEP 標準評估與假設檢定 (Molchan、BSS、PR-AUC、Information Gain)',
        'script': os.path.join(script_dir, 'CSEP國際評估與檢定器.py')
    },
    {
        'num': 5,
        'name': '五大時間尺度時空演化圖與重大實震驗證看板繪製 (Progression Maps)',
        'script': os.path.join(script_dir, '繪製五大時間尺度地震機率演化圖.py')
    }
]

def run_step(step_info):
    print("\n" + "="*75)
    print(f"[管線執行] 步驟 {step_info['num']}: {step_info['name']}")
    print(f"   執行腳本: {step_info['script']}")
    print("="*75)
    
    start_t = time.time()
    cmd = [PYTHON_EXE, step_info['script']]
    result = subprocess.run(cmd, cwd=project_dir)
    elapsed = time.time() - start_t
    
    if result.returncode != 0:
        print(f"[失敗] 步驟 {step_info['num']} 執行中斷 (結束代碼: {result.returncode})，管線終止。")
        sys.exit(result.returncode)
    else:
        print(f"[完成] 步驟 {step_info['num']} 圓滿完成！耗時: {elapsed:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(
        description="臺灣地震時空機器學習預測端到端主管線系統 (Taiwan Earthquake Spatiotemporal ML Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--all', action='store_true', help='一鍵全自動執行步驟 1 至步驟 5 全部流程')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5], help='指定執行單一步驟編號 (1-5)')
    parser.add_argument('--from-step', type=int, choices=[1, 2, 3, 4, 5], default=1, help='自指定步驟開始連續執行至結束')

    args = parser.parse_args()

    print("*"*75)
    print("   臺灣地震時空機器學習機率預測系統 (Taiwan Seismicity ML Pipeline)")
    print("   中央氣象署觀測歷史 (1994–2026) | 38,178 空間網格 | 66 條活動斷層")
    print("*"*75)

    if args.step:
        step_to_run = next(s for s in STEPS if s['num'] == args.step)
        run_step(step_to_run)
    elif args.all:
        for s in STEPS:
            run_step(s)
    else:
        for s in STEPS:
            if s['num'] >= args.from_step:
                run_step(s)

    print("\n" + "#"*75)
    print("【執行完成】端到端地震時空機器學習預測管線已全數順利完成！")
    print("   產物報告、圖資指標表與高解析度看板已完整生成至專案目錄。")
    print("#"*75)

if __name__ == '__main__':
    main()
