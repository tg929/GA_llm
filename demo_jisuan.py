#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计多代分子进化对接分数脚本
计算第6代到第16代的对接分数统计信息：均值、top1、top10均值、top20均值、top50均值、top100均值
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate

# 基础路径
BASE_PATH = "/data1/tgy/GA_llm/output"

# 要分析的代数范围
START_GEN = 6
END_GEN = 16

def read_scores(gen_num):
    """读取指定代数的对接分数文件并返回排序后的分数列表"""
    file_path = f"{BASE_PATH}/generation_{gen_num}/generation_{gen_num}_docked.smi"
    
    scores = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1])
                            scores.append(score)
                        except ValueError:
                            continue
    except Exception as e:
        print(f"读取第{gen_num}代文件失败: {str(e)}")
        return []
    
    return sorted(scores)  # 分数从小到大排序

def calculate_stats(gen_num):
    """计算指定代数的统计信息"""
    sorted_scores = read_scores(gen_num)
    
    if not sorted_scores:
        return [gen_num, 0, 0, 0, 0, 0, 0]
    
    # 总均值
    mean_score = np.mean(sorted_scores)
    
    # top1
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else 0
    
    # top10均值
    top10_scores = sorted_scores[:10] if len(sorted_scores) >= 10 else sorted_scores
    top10_mean = np.mean(top10_scores)
    
    # top20均值
    top20_scores = sorted_scores[:20] if len(sorted_scores) >= 20 else sorted_scores
    top20_mean = np.mean(top20_scores)
    
    # top50均值
    top50_scores = sorted_scores[:50] if len(sorted_scores) >= 50 else sorted_scores
    top50_mean = np.mean(top50_scores)
    
    # top100均值
    top100_scores = sorted_scores[:100] if len(sorted_scores) >= 100 else sorted_scores
    top100_mean = np.mean(top100_scores)
    
    return [gen_num, len(sorted_scores), mean_score, top1_score, top10_mean, top20_mean, top50_mean, top100_mean]

def main():
    """主函数，分析所有代数并输出结果"""
    print(f"开始分析第{START_GEN}代到第{END_GEN}代的对接分数...")
    
    # 收集所有代数的统计信息
    all_stats = []
    for gen in range(START_GEN, END_GEN + 1):
        stats = calculate_stats(gen)
        all_stats.append(stats)
        
    # 创建DataFrame便于展示
    df = pd.DataFrame(all_stats, columns=[
        "代数", "分子数量", "均值", "top1", "top10均值", "top20均值", "top50均值", "top100均值"
    ])
    
    # 设置数值格式
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # 输出美观的表格
    print("\n对接分数统计表:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))
    
    # 保存结果到文件
    output_file = f"{BASE_PATH}/docking_score_stats_gen{START_GEN}-{END_GEN}.csv"
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
