#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GA_llm.py - 分子进化与生成流程整合脚本

完整流程: 
1. 读取当前种群
2. 分子分解(decompose)
3. GPT生成新分子
4. 种群融合与交叉
5. 再次分子分解
6. GPT再次生成新分子
7. 再次种群融合
8. 变异
9. 过滤
10. 分子对接
11. 结果分析与排名

作者: 根据用户需求自动生成
"""

import os
import sys
import argparse
import time
import logging
import subprocess
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 设置项目根目录
PROJECT_ROOT = "/data1/ytg/GA_llm"
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
def setup_logging(output_dir, generation_num):
    log_file = os.path.join(output_dir, f"ga_evolution_{generation_num}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GA_llm")

def run_decompose(input_file, output_prefix, logger):
    """运行分子分解模块"""
    logger.info(f"开始分子分解: {input_file}")
    
    # 准备输出目录
    decompose_dir = os.path.join(PROJECT_ROOT, "datasets/decompose/decompose_results_0")
    os.makedirs(decompose_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(decompose_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_dir, f"truncated_frags_{output_prefix}.smi")
    output_file4 = os.path.join(decompose_dir, f"decomposable_mols_{output_prefix}.smi")
    
    # 构建命令并执行
    decompose_script = os.path.join(PROJECT_ROOT, "datasets/decompose/demo_frags.py")
    cmd = [
        "python", decompose_script,
        "-i", input_file,
        "-o", output_file,
        "-o2", output_file2,
        "-o3", output_file3,
        "-o4", output_file4
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子分解失败: {process.stderr}")
        raise Exception("分子分解失败")
    
    logger.info(f"分子分解完成，生成文件: {output_file3}")
    return output_file3

def run_gpt_generation(input_file, output_prefix, gen_num, logger):
    """运行GPT生成新分子"""
    logger.info(f"开始GPT生成: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output/test0")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径 - 修正输出文件名为实际生成的文件名
    output_file = os.path.join(output_dir, f"crossovered{gen_num}_frags_new_{gen_num}.smi")
    
    # 构建命令并执行
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--device", "0",  # 使用第一个GPU
        "--seed", str(gen_num)  # 使用代数作为种子，确保每代生成不同结果
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"GPT生成失败: {process.stderr}")
        raise Exception("GPT生成失败")
    
    # 检查文件是否实际存在
    if not os.path.exists(output_file):
        logger.warning(f"警告: 预期的输出文件 {output_file} 不存在，尝试查找替代文件...")
        # 尝试找到可能存在的文件
        alternative_file = os.path.join(output_dir, f"crossovered{output_prefix}_frags_new_{gen_num}.smi")
        if os.path.exists(alternative_file):
            logger.info(f"找到替代文件: {alternative_file}")
            output_file = alternative_file
        else:
            # 列出目录中的文件，查找最近生成的可能匹配的文件
            dir_files = [f for f in os.listdir(output_dir) if f.endswith(f"_new_{gen_num}.smi")]
            if dir_files:
                # 按修改时间排序，取最新的文件
                newest_file = max(dir_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
                output_file = os.path.join(output_dir, newest_file)
                logger.info(f"找到最新生成的文件: {output_file}")
            else:
                raise Exception(f"找不到GPT生成的输出文件,生成可能失败")
    
    logger.info(f"GPT生成完成,输出文件: {output_file}")
    return output_file

def run_crossover(source_file, llm_file, output_file, gen_num, num_crossovers, logger):
    """运行分子交叉"""
    logger.info(f"开始分子交叉: 源文件 {source_file}, LLM生成文件 {llm_file}, 交叉次数 {num_crossovers}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo.py")
    cmd = [
        "python", crossover_script,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_rate", "0.8",
        "--crossover_attempts", str(num_crossovers)  # 使用传入的交叉次数
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子交叉失败: {process.stderr}")
        raise Exception("分子交叉失败")
    
    logger.info(f"分子交叉完成，生成文件: {output_file}")
    return output_file

def run_mutation(input_file, llm_file, output_file, num_mutations, logger):
    """运行分子变异"""
    logger.info(f"开始分子变异: 输入文件 {input_file}, LLM生成文件 {llm_file}, 变异次数 {num_mutations}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo.py")
    cmd = [
        "python", mutation_script,
        "--input_file", input_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--mutation_attempts", str(num_mutations),  # 使用传入的变异次数
        "--max_mutations", "2"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子变异失败: {process.stderr}")
        raise Exception("分子变异失败")
    
    logger.info(f"分子变异完成，生成文件: {output_file}")
    return output_file

def run_filter(input_file, output_file, logger):
    """运行分子过滤"""
    logger.info(f"开始分子过滤: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令并执行
    filter_script = os.path.join(PROJECT_ROOT, "operations/filter/filter_demo.py")
    cmd = [
        "python", filter_script,
        "--input", input_file,
        "--output", output_file
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子过滤失败: {process.stderr}")
        raise Exception("分子过滤失败")
    
    logger.info(f"分子过滤完成，生成文件: {output_file}")
    return output_file

# 并行对接的工作函数
def dock_molecule(molecule_idx, molecule, args, temp_dir, logger):
    """对单个分子进行对接"""
    try:
        # 创建临时输入文件
        temp_input = os.path.join(temp_dir, f"mol_{molecule_idx}.smi")
        with open(temp_input, 'w') as f:
            f.write(molecule.strip() + '\n')
            
        # 创建临时输出文件
        temp_output = os.path.join(temp_dir, f"mol_{molecule_idx}_docked.smi")
        
        # 构建对接命令
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "--input", temp_input,
            "--receptor", args.receptor_file,
            "--output", temp_output,
            "--mgltools", args.mgltools_path,
            "--max_failures", "5"
        ]
        
        # 执行对接
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.warning(f"分子 {molecule_idx} 对接失败: {process.stderr}")
            return None
        
        # 读取结果
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                result = f.read().strip()
                if result:
                    return result
        
        return None
    except Exception as e:
        logger.error(f"分子 {molecule_idx} 对接过程出错: {str(e)}")
        return None

def run_docking(input_file, output_file, receptor_file, mgltools_path, logger, num_processors=1, multithread_mode="serial"):
    """运行分子对接，支持并行处理"""
    logger.info(f"开始分子对接: {input_file}, 处理器数量: {num_processors}, 模式: {multithread_mode}")
    
    # 准备输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定处理器数量 - 提前处理，避免重复代码
    if num_processors == -1 or num_processors > multiprocessing.cpu_count():
        num_processors = multiprocessing.cpu_count()
        logger.info(f"自动设置使用所有可用的CPU核心: {num_processors}")
    
    # 根据处理器数量自动选择并行模式
    if num_processors > 1 and multithread_mode == "serial":
        logger.info(f"检测到使用多核({num_processors})但模式为seria,自动切换为multithreading模式")
        multithread_mode = "multithreading"
        
    # 如果选择串行模式或只使用一个处理器，使用原始的对接方法
    if multithread_mode == "serial" or num_processors == 1:
        logger.info("使用串行模式进行对接")
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "--input", input_file,
            "--receptor", receptor_file,
            "--output", output_file,
            "--mgltools", mgltools_path,
            "--max_failures", "5"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"分子对接失败: {process.stderr}")
            raise Exception("分子对接失败")
        
        logger.info(f"分子对接完成，生成文件: {output_file}")
        return output_file
    
    # 并行处理
    logger.info(f"使用并行模式进行对接，处理器数量: {num_processors}")
    
    # 读取输入文件中的分子
    with open(input_file, 'r') as f:
        molecules = [line for line in f.readlines() if line.strip()]
    
    total_molecules = len(molecules)
    logger.info(f"共有 {total_molecules} 个分子需要对接")
    
    # 创建临时目录存放分割后的文件
    temp_dir = os.path.join(output_dir, "temp_docking")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 设置工作函数参数
    dock_func = partial(dock_molecule, args=argparse.Namespace(
        receptor_file=receptor_file,
        mgltools_path=mgltools_path
    ), temp_dir=temp_dir, logger=logger)
    
    # 计算每个处理器应该处理的分子数量，确保负载平衡
    molecules_per_processor = max(1, total_molecules // num_processors)
    
    # 并行执行对接
    results = []
    start_time = time.time()
    
    # 优化：使用批处理方式进行对接
    if multithread_mode == "multithreading":
        logger.info("使用多线程模式")
        with ThreadPoolExecutor(max_workers=num_processors) as executor:
            # 批量提交任务，改善负载均衡
            future_to_idx = {
                executor.submit(dock_func, idx, mol): idx 
                for idx, mol in enumerate(molecules)
            }
            
            # 处理结果时显示进度
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    logger.info(f"已完成: {completed}/{total_molecules} 分子 "
                               f"({completed/total_molecules*100:.1f}%), "
                               f"耗时: {elapsed:.1f}秒, "
                               f"预计剩余时间: {elapsed/completed*(total_molecules-completed):.1f}秒")
                if result:
                    results.append(result)
    else:  # mpi 模式使用进程池实现
        logger.info("使用多进程模式")
        # 使用更高效的maxtasksperchild参数，避免内存泄漏
        with ProcessPoolExecutor(max_workers=num_processors, 
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            # 批量提交任务
            future_to_idx = {
                executor.submit(dock_func, idx, mol): idx 
                for idx, mol in enumerate(molecules)
            }
            
            # 处理结果时显示进度
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    logger.info(f"已完成: {completed}/{total_molecules} 分子 "
                               f"({completed/total_molecules*100:.1f}%), "
                               f"耗时: {elapsed:.1f}秒, "
                               f"预计剩余时间: {elapsed/completed*(total_molecules-completed):.1f}秒")
                if result:
                    results.append(result)
    
    end_time = time.time()
    logger.info(f"对接计算完成，总耗时: {end_time - start_time:.2f}秒，平均每个分子: {(end_time - start_time)/total_molecules:.2f}秒")
    
    # 合并结果到输出文件
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    logger.info(f"并行对接完成，成功对接 {len(results)}/{total_molecules} 个分子，结果保存至: {output_file}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_file

def run_analysis(input_file, output_prefix, gen_num, logger):
    """运行对接结果分析"""
    logger.info(f"开始对接结果分析: {input_file}")
    
    # 准备输出目录
    output_dir = os.path.dirname(input_file)
    
    # 构建命令并执行
    analysis_script = os.path.join(PROJECT_ROOT, "operations/docking/analyse_result_0.py")
    cmd = [
        "python", analysis_script,
        "--input", input_file,
        "--output", output_dir,
        "--prefix", f"generation_{gen_num}"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"对接结果分析失败: {process.stderr}")
        raise Exception("对接结果分析失败")
    
    logger.info(f"对接结果分析完成，结果保存至: {output_dir}/generation_{gen_num}_stats.txt")
    return f"{output_dir}/generation_{gen_num}_sorted.smi"

# 限制种群大小的函数
def limit_population_size(file_path, max_size, output_path=None):
    """根据设置的最大种群数量限制文件中的分子数量"""
    if max_size <= 0:  # 如果max_size为0或负数，不做任何处理
        return file_path
        
    if output_path is None:
        output_path = file_path
        
    # 读取全部分子
    with open(file_path, 'r') as f:
        molecules = [line.strip() for line in f if line.strip()]
        
    total = len(molecules)
    if total <= max_size:  # 如果当前数量已经小于限制，不做任何处理
        return file_path
        
    # 随机选择max_size个分子
    import random
    selected = random.sample(molecules, max_size)
    
    # 写回文件
    with open(output_path, 'w') as f:
        for mol in selected:
            f.write(f"{mol}\n")
            
    print(f"种群大小已从{total}限制为{max_size}")
    return output_path

def run_evolution(generation_num, args, logger):
    """执行一次完整的进化迭代"""
    logger.info(f"开始第 {generation_num} 代进化")
    
    # 创建各代输出目录
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)
    
    # 各阶段输出文件
    crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
    mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
    filter_output = os.path.join(output_base, f"generation_{generation_num}_filtered.smi")
    docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
    
    # 确定当前代的种群文件
    if generation_num == 0:
        # 第一代使用初始种群
        current_population = args.initial_population
    else:
        # 后续代使用上一代的对接结果
        current_population = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_docked.smi")
    
    # 1. 第一次分子分解
    decompose_output1 = run_decompose(current_population, f"crossover{generation_num}", logger)
    
    # 2. 第一次GPT生成
    gpt_output1 = run_gpt_generation(decompose_output1, f"crossover{generation_num}", generation_num, logger)
    
    # 3. 分子交叉
    crossover_output = run_crossover(current_population, gpt_output1, crossover_output, generation_num, args.num_crossovers, logger)
    
    # 4. 第二次分子分解
    decompose_output2 = run_decompose(crossover_output, f"mutation{generation_num}", logger)
    
    # 5. 第二次GPT生成
    gpt_output2 = run_gpt_generation(decompose_output2, f"mutation{generation_num}", generation_num, logger)
    
    # 6. 分子变异
    mutation_output = run_mutation(crossover_output, gpt_output2, mutation_output, args.num_mutations, logger)
    
    # 7. 分子过滤
    filter_output = run_filter(mutation_output, filter_output, logger)
    
    # 8. 分子对接（使用并行处理）
    docking_output = run_docking(
        filter_output, 
        docking_output, 
        args.receptor_file, 
        args.mgltools_path, 
        logger,
        args.number_of_processors,
        args.multithread_mode
    )
    
    # 9. 对接结果分析
    analysis_output = run_analysis(docking_output, output_base, generation_num, logger)
    
    logger.info(f"第 {generation_num} 代进化完成")
    return analysis_output

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm - 分子进化与生成流程')
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=10, 
                        help='进化代数')
    parser.add_argument('--output_dir', type=str, default='/data1/ytg/GA_llm/output',
                        help='输出目录')
    parser.add_argument('--initial_population', type=str, 
                        default='/data1/ytg/GA_llm/datasets/source_compounds/naphthalene_smiles.smi',
                        help='初始种群文件路径')
    
    # 对接参数
    parser.add_argument('--receptor_file', type=str,
                        default='/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb',
                        help='受体PDB文件路径')
    parser.add_argument('--mgltools_path', type=str,
                        default='/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6',
                        help='MGLTools安装路径')
    
    # 进化参数
    parser.add_argument('--num_crossovers', type=int, default=1,
                       help='每代执行的交叉次数')
    parser.add_argument('--num_mutations', type=int, default=1,
                       help='每代执行的变异次数')
    parser.add_argument('--max_population', type=int, default=0,
                       help='控制每代种群的最大数量,设置为0表示不限制(可能导致种群规模迅速增长）')
    
    # 并行处理参数
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于并行计算的处理器数量。设置为-1表示自动检测并使用所有可用CPU核心(推荐）。')
    parser.add_argument('--multithread_mode', default="multithreading",
                        choices=["mpi", "multithreading", "serial"],
                        help='多线程模式选择: mpi, multithreading, 或 serial。serial模式将忽略处理器数量设置,强制使用单处理器。')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果number_of_processors为-1，则自动检测并使用所有可用的CPU核心
    if args.number_of_processors == -1:
        args.number_of_processors = multiprocessing.cpu_count()
        print(f"自动检测到{args.number_of_processors}个CPU核心,将全部使用")
    
    # 如果使用多核但未指定多线程模式，自动切换为multithreading模式
    if args.number_of_processors > 1 and args.multithread_mode == "serial":
        print(f"检测到使用多核({args.number_of_processors})但模式为serial,自动切换为multithreading模式")
        args.multithread_mode = "multithreading"
    
    # 如果设置了种群大小限制，检查初始种群
    if args.max_population > 0:
        # 检查初始种群大小
        with open(args.initial_population, 'r') as f:
            initial_count = sum(1 for line in f if line.strip())
        if initial_count > args.max_population:
            limited_file = os.path.join(args.output_dir, "limited_initial_population.smi")
            args.initial_population = limit_population_size(args.initial_population, args.max_population, limited_file)
            print(f"初始种群已从{initial_count}限制为{args.max_population}")
    
    # 执行多代进化
    for gen in range(args.generations):
        logger = setup_logging(args.output_dir, gen)
        try:
            logger.info(f"开始第 {gen} 代进化")
            start_time = time.time()
            
            # 如果前一代种群存在且超过限制大小，先限制它
            if gen > 0 and args.max_population > 0:
                prev_gen_file = os.path.join(args.output_dir, f"generation_{gen-1}", f"generation_{gen-1}_docked.smi")
                if os.path.exists(prev_gen_file):
                    with open(prev_gen_file, 'r') as f:
                        prev_count = sum(1 for line in f if line.strip())
                    if prev_count > args.max_population:
                        limit_population_size(prev_gen_file, args.max_population)
                        logger.info(f"第{gen-1}代种群已从{prev_count}限制为{args.max_population}")
            
            final_output = run_evolution(gen, args, logger)
            
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {final_output}")
            
        except Exception as e:
            logger.error(f"第 {gen} 代进化失败: {str(e)}")
            break

if __name__ == "__main__":
    main() 