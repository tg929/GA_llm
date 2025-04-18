#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算分子对接得分统计数据 - 多代分析版本
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

def print_stats(generation, stats):
    """
    打印统计数据
    
    参数:
    generation : str, 世代标识
    stats : dict, 统计数据字典
    """
    print("\n" + "="*60)
    print(f"第 {generation} 代分子对接得分统计数据")
    print("="*60)
    
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("="*60)

def main():
    # 要分析的所有世代文件路径
    generations = [0, 1, 2, 3, 4]
    
    # 打印各代结果的比较表头
    print("\n" + "="*100)
    print("多代分子对接得分统计数据比较")
    print("="*100)
    print(f"{'世代':^10}{'总样本数':^15}{'所有得分均值':^15}{'最好得分':^15}{'Top 5均值':^15}{'Top 10均值':^15}{'Top 20均值':^15}{'Top 50均值':^15}")
    print("-"*100)
    
    # 处理每一代的数据
    for gen in generations:
        file_path = f"/data1/ytg/GA_llm/output/generation_{gen}/generation_{gen}_sorted.smi"
        
        try:
            # 计算统计数据
            stats = calculate_statistics(file_path)
            
            # 打印详细结果
            print_stats(gen, stats)
            
            # 打印比较数据行
            print(f"{f'第{gen}代':^10}{stats['总样本数']:^15}{stats['所有得分均值']:^15.4f}{stats['最好得分 (Top 1)']:^15.4f}{stats['Top 5 均值']:^15.4f}{stats['Top 10 均值']:^15.4f}{stats['Top 20 均值']:^15.4f}{stats['Top 50 均值']:^15.4f}")
            
        except Exception as e:
            print(f"第 {gen} 代数据分析错误: {str(e)}")
    
    print("="*100)
    print("分析完成!")

if __name__ == "__main__":
    main()
