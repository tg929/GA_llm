#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对初始种群分子与目标蛋白质进行对接，并计算统计数据
"""

import os
import sys
import time
import numpy as np
import subprocess
import multiprocessing
import argparse

# 设置项目根目录
PROJECT_ROOT = "/data1/ytg/GA_llm"
sys.path.insert(0, PROJECT_ROOT)

# 导入GA_llm中的函数
from GA_llm import run_docking, setup_logging

# 输出目录
OUTPUT_DIR = "/data1/ytg/GA_llm/test_output_initial"

# 对接结果进行排序
def sort_docking_results(input_file, output_file):
    """对对接结果按得分排序"""
    print(f"对对接结果进行排序: {input_file}")
    
    # 读取文件
    molecules = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # 假设最后一个元素是得分
                    smile = ' '.join(parts[:-1])
                    score = float(parts[-1])
                    molecules.append((smile, score))
                except ValueError:
                    print(f"警告: 无法解析行: {line.strip()}")
    
    # 按得分排序 (从低到高，因为得分越低越好)
    molecules.sort(key=lambda x: x[1])
    
    # 写入排序后的结果
    with open(output_file, 'w') as f:
        for smile, score in molecules:
            f.write(f"{smile} {score}\n")
    
    print(f"排序完成，结果保存至: {output_file}")
    return output_file

# 计算统计数据
def calculate_statistics(file_path):
    """计算分子对接得分的统计数据"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 读取文件内容
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 最后一个元素应该是得分
                try:
                    score = float(parts[-1])
                    scores.append(score)
                except ValueError:
                    print(f"警告: 无法将'{parts[-1]}'转换为浮点数")
    
    # 检查是否成功读取到得分
    if not scores:
        raise ValueError("无法从文件中读取到任何有效得分")
    
    # 计算统计数据
    scores = np.array(scores)
    stats = {
        "总样本数": len(scores),
        "所有得分均值": np.mean(scores),
        "最好得分 (Top 1)": scores[0],  # 文件已排序，所以第一个是最好的
        "Top 5 均值": np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores),
        "Top 10 均值": np.mean(scores[:10]) if len(scores) >= 10 else np.mean(scores[:len(scores)]),
        "Top 20 均值": np.mean(scores[:20]) if len(scores) >= 20 else np.mean(scores[:len(scores)]),
        "Top 50 均值": np.mean(scores[:50]) if len(scores) >= 50 else np.mean(scores[:len(scores)])
    }
    
    return stats

# 打印统计结果
def print_stats(stats):
    """打印统计数据"""
    print("\n" + "="*60)
    print("初始种群分子对接得分统计数据")
    print("="*60)
    
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("="*60)

# 创建一个与GA_llm.py中相同的运行分析函数
def run_analysis(input_file, output_prefix, gen_num, logger):
    """运行对接结果分析"""
    print(f"开始对接结果分析: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(input_file)
    
    # 构建命令并执行
    analysis_script = os.path.join(PROJECT_ROOT, "operations/docking/analyse_result_0.py")
    cmd = [
        "python", analysis_script,
        "--input", input_file,
        "--output", output_dir,
        "--prefix", f"initial"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"对接结果分析失败: {process.stderr}")
        raise Exception("对接结果分析失败")
    
    print(f"对接结果分析完成，结果保存至: {output_dir}/initial_stats.txt")
    return f"{output_dir}/initial_sorted.smi"

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置参数
    args = argparse.Namespace(
        initial_population="/data1/ytg/GA_llm/datasets/source_compounds/naphthalene_smiles.smi",
        receptor_file="/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb",
        mgltools_path="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6",
        number_of_processors=-1,  # 自动使用所有CPU核心
        multithread_mode="multithreading"
    )
    
    # 输出文件路径
    docked_file = os.path.join(OUTPUT_DIR, "initial_docked.smi")
    sorted_file = os.path.join(OUTPUT_DIR, "initial_sorted.smi")
    
    try:
        # 设置日志记录器
        logger = setup_logging(OUTPUT_DIR, "initial")
        
        # 1. 对分子进行对接，使用GA_llm.py中的函数
        run_docking(
            args.initial_population, 
            docked_file, 
            args.receptor_file, 
            args.mgltools_path, 
            logger,
            args.number_of_processors,
            args.multithread_mode
        )
        
        # 2. 运行分析函数
        run_analysis(docked_file, OUTPUT_DIR, "initial", logger)
        
        # 3. 计算统计数据
        stats = calculate_statistics(sorted_file)
        
        # 4. 打印统计结果
        print_stats(stats)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
