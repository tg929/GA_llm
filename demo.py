#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算分子对接得分统计数据
"""

import os
import numpy as np

def calculate_statistics(file_path):
    """
    计算文件中对接得分的统计数据
    
    参数:
    file_path : str, 文件路径
    
    返回:
    statistics : dict, 包含各种统计数据
    """
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

def print_stats(stats):
    """
    打印统计数据
    
    参数:
    stats : dict, 统计数据字典
    """
    print("\n" + "="*50)
    print("分子对接得分统计数据")
    print("="*50)
    
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("="*50)

def main():
    # 文件路径
    file_path = "/data1/ytg/GA_llm/output/generation_3/generation_3_sorted.smi"
    
    try:
        # 计算统计数据
        stats = calculate_statistics(file_path)
        
        # 打印结果
        print_stats(stats)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
