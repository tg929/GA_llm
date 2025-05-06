#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GA_llm_finetune.py - 优化后的分子进化与生成流程整合脚本

主要优化点：
1. 引入种子选择机制，控制种群规模
2. 优化交叉和变异操作，限制生成的新个体数量
3. 保持原有的进化流程，但使用更高效的种群管理策略

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
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# 设置项目根目录
PROJECT_ROOT = "/data1/tgy/GA_llm"
sys.path.insert(0, PROJECT_ROOT)

# 设置随机种子确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# 尝试设置RDKit的随机种子
try:
    from rdkit.Chem import rdBase
    rdBase.SetRandomSeed(SEED)
except:
    pass

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
    return logging.getLogger("GA_llm_finetune")

def run_decompose(input_file, output_prefix, logger):    
    logger.info(f"开始分子分解: {input_file}")    
    
    decompose_dir = os.path.join(PROJECT_ROOT, "datasets/decompose/decompose_results")
    os.makedirs(decompose_dir, exist_ok=True)    
    
    output_file = os.path.join(decompose_dir, f"frags_result_{output_prefix}.smi")
    output_file2 = os.path.join(decompose_dir, f"frags_seq_{output_prefix}.smi")
    output_file3 = os.path.join(decompose_dir, f"truncated_frags_{output_prefix}.smi")
    output_file4 = os.path.join(decompose_dir, f"decomposable_mols_{output_prefix}.smi")
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
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
    logger.info(f"开始GPT生成: {input_file}")
    
    
    output_dir = os.path.join(PROJECT_ROOT, "fragment_GPT/output")
    os.makedirs(output_dir, exist_ok=True)   
    
    output_file = os.path.join(output_dir, f"crossovered{gen_num}_frags_new_{gen_num}.smi")
    
    
    generate_script = os.path.join(PROJECT_ROOT, "fragment_GPT/generate_all.py")
    cmd = [
        "python", generate_script,
        "--input_file", input_file,
        "--device", "0",  # 使用第一个GPU
        "--seed", str(gen_num)  # 使用当前代数作为种子，确保文件名一致性
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"GPT生成失败: {process.stderr}")
        raise Exception("GPT生成失败")    
    
    if not os.path.exists(output_file):
        logger.warning(f"警告: 预期的输出文件 {output_file} 不存在，尝试查找替代文件...")
       
        alternative_file = os.path.join(output_dir, f"crossovered{output_prefix}_frags_new_{gen_num}.smi")
        if os.path.exists(alternative_file):
            logger.info(f"找到替代文件: {alternative_file}")
            output_file = alternative_file
        else:            
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

# 种子选择相关函数
def get_molecules_from_file(file_path, with_score=False):
    """从文件中读取分子SMILES"""
    molecules = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if with_score and len(parts) >= 2:                       
                        molecules.append((parts[0], float(parts[1])))
                    else:                        
                        molecules.append(parts[0])
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return []
    
    return molecules

def select_seed_molecules(current_population, fitness_seeds, diversity_seeds, args):
    """
    从当前种群中选择种子分子
    
    Args:
        current_population: 当前种群文件路径
        fitness_seeds: 基于适应度选择的种子数量
        diversity_seeds: 基于多样性选择的种子数量
        args: 命令行参数
        
    Returns:
        list: 选定的种子分子SMILES列表
    """
    # 读取当前种群
    if isinstance(current_population, str):
        with open(current_population, 'r') as f:
            current_population = [(line.strip().split()[0], float(line.strip().split()[1])) for line in f if line.strip()]
    
    # 检查是否所有分子都有相同的分数（初始种群情况）
    all_scores = [score for _, score in current_population]
    is_initial_population = all(score == 0.0 for score in all_scores)
    
    # 根据适应度排序
    current_population.sort(key=lambda x: x[1], reverse=True)
    
    # 选择适应度最高的分子
    fitness_selected = []
    if is_initial_population:
        # 对于初始种群，随机选择适应度种子
        if len(current_population) > fitness_seeds:
            fitness_selected = [mol[0] for mol in random.sample(current_population, fitness_seeds)]
            # 从当前种群中移除已选择的分子
            current_population = [mol for mol in current_population if mol[0] not in fitness_selected]
        else:
            fitness_selected = [mol[0] for mol in current_population]
            current_population = []
    else:
        if args.selector_choice == "Rank_Selector":
            # 排名选择
            for i in range(min(fitness_seeds, len(current_population))):
                fitness_selected.append(current_population[i][0])
                current_population.pop(i)
        else:  # Roulette_Selector
            # 轮盘选择
            total_fitness = sum(mol[1] for mol in current_population)
            if total_fitness > 0:
                for _ in range(fitness_seeds):
                    if not current_population:
                        break
                    r = random.random() * total_fitness
                    cumsum = 0
                    for mol in current_population:
                        cumsum += mol[1]
                        if cumsum >= r:
                            fitness_selected.append(mol[0])
                            current_population.remove(mol)
                            total_fitness -= mol[1]
                    break
            
    # 然后从剩余分子中选择多样性最高的
    diversity_selected = []
    if diversity_seeds > 0 and current_population:
        # 计算分子指纹
        mols = [Chem.MolFromSmiles(smile) for smile, _ in current_population]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]
        
        # 选择多样性最高的分子
        selected_indices = []
        while len(selected_indices) < min(diversity_seeds, len(current_population)):
            if not selected_indices:
                # 选择第一个分子
                selected_indices.append(0)
            else:
                # 计算每个未选择分子与已选择分子的平均相似度
                max_diversity = -1
                best_idx = -1
                for i in range(len(current_population)):
                    if i not in selected_indices:
                        avg_similarity = sum(DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in selected_indices) / len(selected_indices)
                        if avg_similarity < max_diversity or max_diversity == -1:
                            max_diversity = avg_similarity
                            best_idx = i
                if best_idx != -1:
                    selected_indices.append(best_idx)
        
        # 获取选定的分子
        diversity_selected = [current_population[i][0] for i in selected_indices]
    
    # 合并所有选定的种子
    selected_seeds = fitness_selected + diversity_selected
    
    print(f"已选择 {len(selected_seeds)} 个种子分子，其中基于适应度选择 {len(fitness_selected)} 个，基于多样性选择 {len(diversity_selected)} 个")
    
    return selected_seeds

def save_seed_list(output_dir, generation_num, seed_list, tag="seeds"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generation_{generation_num}_{tag}.smi")
    
    with open(output_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")
    
    return output_file

def run_crossover_with_seeds(source_file, llm_file, output_file, seed_list, num_crossovers, gen_num, logger):
    """
    运行分子交叉，使用选定的种子分子
    
    Args:
        source_file: 包含上一代所有分子的文件
        llm_file: LLM生成的分子文件
        output_file: 输出文件路径
        seed_list: 选定的种子分子SMILES列表
        num_crossovers: 需要生成的交叉产物数量
        gen_num: 当前代数
        logger: 日志记录器
    
    Returns:
        包含生成的交叉产物的文件路径
    """
    logger.info(f"开始分子交叉，目标生成 {num_crossovers} 个新分子")
        
    # 使用统一的种子文件命名
    seed_file = os.path.join(os.path.dirname(output_file), f"generation_{gen_num}_seeds.smi")
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")    
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)   
    
    # 动态调整交叉尝试次数
    if gen_num > 0:
        # 读取上一代的成功率
        prev_gen_dir = os.path.join(os.path.dirname(output_dir), f"generation_{gen_num-1}")
        prev_stats_file = os.path.join(prev_gen_dir, f"generation_{gen_num-1}_stats.txt")
        if os.path.exists(prev_stats_file):
            with open(prev_stats_file, 'r') as f:
                stats = f.read()
                if "交叉成功率" in stats:
                    # 根据上一代的成功率调整本代的尝试次数
                    success_rate = float(stats.split("交叉成功率:")[1].split("%")[0].strip())
                    if success_rate < 30:
                        num_crossovers = int(num_crossovers * 1.5)  # 成功率低，增加尝试次数
                    elif success_rate > 70:
                        num_crossovers = int(num_crossovers * 0.8)  # 成功率高，减少尝试次数
    
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        "python", crossover_script,
        "--seed_file", seed_file,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_rate", "0.8",
        "--crossover_attempts", str(num_crossovers)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子交叉失败: {process.stderr}")
        raise Exception("分子交叉失败")
    
    logger.info(f"分子交叉完成，生成文件: {output_file}")
    return output_file

def run_mutation_with_seeds(input_file, llm_file, output_file, seed_list, num_mutations, gen_num, logger):
    """
    运行分子变异，使用选定的种子分子
    
    Args:
        input_file: 包含上一代所有分子的文件
        llm_file: LLM生成的分子文件
        output_file: 输出文件路径
        seed_list: 选定的种子分子SMILES列表
        num_mutations: 需要生成的变异产物数量
        gen_num: 当前代数
        logger: 日志记录器
    
    Returns:
        包含生成的变异产物的文件路径
    """
    logger.info(f"开始分子变异，目标生成 {num_mutations} 个新分子")
        
    # 使用统一的种子文件命名
    seed_file = os.path.join(os.path.dirname(output_file), f"generation_{gen_num}_seeds.smi")
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")
   
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  
   
    # 动态调整变异尝试次数
    if gen_num > 0:
        # 读取上一代的成功率
        prev_gen_dir = os.path.join(os.path.dirname(output_dir), f"generation_{gen_num-1}")
        prev_stats_file = os.path.join(prev_gen_dir, f"generation_{gen_num-1}_stats.txt")
        if os.path.exists(prev_stats_file):
            with open(prev_stats_file, 'r') as f:
                stats = f.read()
                if "变异成功率" in stats:
                    # 根据上一代的成功率调整本代的尝试次数
                    success_rate = float(stats.split("变异成功率:")[1].split("%")[0].strip())
                    if success_rate < 30:
                        num_mutations = int(num_mutations * 1.5)  # 成功率低，增加尝试次数
                    elif success_rate > 70:
                        num_mutations = int(num_mutations * 0.8)  # 成功率高，减少尝试次数
   
    # 1. 对变异种子进行分子分解
    decompose_output = run_decompose(seed_file, f"mutation_gen{gen_num}", logger)
    
    # 2. 使用GPT生成新的分子片段
    try:
        gpt_output = run_gpt_generation(decompose_output, f"mutation_{gen_num}", gen_num, logger)
    except Exception as e:
        logger.warning(f"变异前的GPT生成失败,使用种子文件作为替代: {str(e)}")
        gpt_output = seed_file
   
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo_finetune.py")
    cmd = [
        "python", mutation_script,
        "--input_file", seed_file,
        "--llm_generation_file", gpt_output,
        "--output_file", output_file,
        "--mutation_attempts", str(num_mutations),
        "--max_mutations", "2"
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子变异失败: {process.stderr}")
        raise Exception("分子变异失败")
    
    logger.info(f"分子变异完成，生成文件: {output_file}")
    return output_file

def run_filter(input_file, output_file, logger, args):
    """运行分子过滤"""
    logger.info(f"开始分子过滤: {input_file}")
    
   
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    
    filter_params = []
    
    
    if args.LipinskiStrictFilter:
        filter_params.extend(["--LipinskiStrictFilter"])
    if args.LipinskiLenientFilter:
        filter_params.extend(["--LipinskiLenientFilter"])
    if args.GhoseFilter:
        filter_params.extend(["--GhoseFilter"])
    if args.GhoseModifiedFilter:
        filter_params.extend(["--GhoseModifiedFilter"])
    if args.MozziconacciFilter:
        filter_params.extend(["--MozziconacciFilter"])
    if args.VandeWaterbeemdFilter:
        filter_params.extend(["--VandeWaterbeemdFilter"])
    if args.PAINSFilter:
        filter_params.extend(["--PAINSFilter"])
    if args.NIHFilter:
        filter_params.extend(["--NIHFilter"])
    if args.BRENKFilter:
        filter_params.extend(["--BRENKFilter"])
    if args.No_Filters:
        filter_params.extend(["--No_Filters"])
    
    
    if args.alternative_filter:
        for filter_entry in args.alternative_filter:
            filter_params.extend(["--alternative_filter", filter_entry])
    
    
    if not filter_params and not args.No_Filters:
        logger.warning("没有指定任何过滤器参数，将使用默认过滤器")    
    
    filter_script = os.path.join(PROJECT_ROOT, "operations/filter/filter_demo.py")
    cmd = [
        "python", filter_script,
        "--input", input_file,
        "--output", output_file
    ]
        
    cmd.extend(filter_params)
    
    logger.info(f"执行过滤命令: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"分子过滤失败: {process.stderr}")
        raise Exception("分子过滤失败")
    
    logger.info(f"分子过滤完成，生成文件: {output_file}")
    return output_file

def dock_molecule(molecule_idx, molecule, args, temp_dir, logger):
    """对单个分子进行对接"""
    try:
       
        temp_input = os.path.join(temp_dir, f"mol_{molecule_idx}.smi")
        with open(temp_input, 'w') as f:
            f.write(molecule.strip() + '\n')            
        
        temp_output = os.path.join(temp_dir, f"mol_{molecule_idx}_docked.smi")        
        
        docking_script = os.path.join(PROJECT_ROOT, "operations/docking/docking_demo.py")
        cmd = [
            "python", docking_script,
            "--input", temp_input,
            "--receptor", args.receptor_file,
            "--output", temp_output,
            "--mgltools", args.mgltools_path,
            "--max_failures", "5"
        ]       
       
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.warning(f"分子 {molecule_idx} 对接失败: {process.stderr}")
            return None        
        
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
        
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)    
    available_cpus = multiprocessing.cpu_count()
    if num_processors == -1 or num_processors > available_cpus:
        num_processors = available_cpus
        logger.info(f"自动设置使用所有可用的CPU核心: {num_processors}")  
    
    if num_processors > 1 and multithread_mode == "serial":
        logger.info(f"检测到使用多核({num_processors})但模式为serial,自动切换为multithreading模式")
        multithread_mode = "multithreading"        
   
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
    
    logger.info(f"使用并行模式进行对接，处理器数量: {num_processors}")    
    
    with open(input_file, 'r') as f:
        molecules = [line for line in f.readlines() if line.strip()]
    
    total_molecules = len(molecules)
    logger.info(f"共有 {total_molecules} 个分子需要对接")    
    
    temp_dir = os.path.join(output_dir, "temp_docking")
    os.makedirs(temp_dir, exist_ok=True)    
    
    dock_func = partial(dock_molecule, args=argparse.Namespace(
        receptor_file=receptor_file,
        mgltools_path=mgltools_path
    ), temp_dir=temp_dir, logger=logger)    
    
    molecules_per_processor = max(1, total_molecules // num_processors)        
    results = []
    start_time = time.time()  
  
    batch_size = max(1, min(100, molecules_per_processor))  
    
    molecule_batches = []
    for i in range(0, total_molecules, batch_size):
        end = min(i + batch_size, total_molecules)
        molecule_batches.append((i, molecules[i:end]))
    
    logger.info(f"将 {total_molecules} 个分子分为 {len(molecule_batches)} 批进行处理，每批大约 {batch_size} 个分子")
    
    
    if multithread_mode == "multithreading":
        logger.info(f"使用多线程模式，线程数: {num_processors}")
        with ThreadPoolExecutor(max_workers=num_processors) as executor:            
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx            
           
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1                
                
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    else:  
        logger.info(f"使用多进程模式，进程数: {num_processors}")
        # 使用spawn上下文避免潜在的内存泄漏问题
        mp_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_processors, mp_context=mp_context) as executor:
           
            future_to_idx = {}
            for batch_idx, (start_idx, batch) in enumerate(molecule_batches):
                for mol_idx, mol in enumerate(batch):
                    future = executor.submit(dock_func, start_idx + mol_idx, mol)
                    future_to_idx[future] = start_idx + mol_idx            
            
            completed = 0
            successful = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                completed += 1
                if result:
                    results.append(result)
                    successful += 1              
                
                if completed % max(1, total_molecules // 20) == 0 or completed == total_molecules:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / completed) * (total_molecules - completed) if completed > 0 else 0
                    logger.info(f"已完成: {completed}/{total_molecules} ({completed/total_molecules*100:.1f}%), "
                               f"成功: {successful}/{completed} ({successful/completed*100:.1f}% 成功率), "
                               f"耗时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"对接计算完成，总耗时: {total_time:.2f}秒，"
               f"平均每个分子: {total_time/total_molecules:.2f}秒，"
               f"总成功率: {len(results)/total_molecules*100:.1f}%")
    
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    logger.info(f"并行对接完成，成功对接 {len(results)}/{total_molecules} 个分子，结果保存至: {output_file}")
       
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

def calculate_and_print_stats(docking_output, generation_num, logger):
    """计算并输出当前种群的分数统计信息"""
    molecules = []
    scores = []
    try:
        with open(docking_output, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        molecules.append(parts[0])
                        scores.append(float(parts[1]))
    except Exception as e:
        logger.error(f"读取对接结果文件失败: {str(e)}")
        return
    
    if not scores:
        logger.warning("对接结果中没有发现有效分数")
        return
   
    sorted_scores = sorted(scores)
    
   
    mean_score = np.mean(sorted_scores)
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else None
    
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
    stats_message = (
        f"\n==================== Generation {generation_num} 统计信息 ====================\n"
        f"总分子数: {len(scores)}\n"
        f"所有分子得分均值: {mean_score:.4f}\n"
        f"Top1得分: {top1_score:.4f}\n"
        f"Top10得分均值: {top10_mean:.4f}\n"
        f"Top20得分均值: {top20_mean:.4f}\n"
        f"Top50得分均值: {top50_mean:.4f}\n"
        f"Top100得分均值: {top100_mean:.4f}\n"
        f"========================================================================\n"
    )
    logger.info(stats_message)
    print(stats_message)

def limit_population_size(file_path, max_size, output_path=None):
    """根据设置的最大种群数量限制文件中的分子数量"""
    if max_size <= 0:  
        return file_path
        
    if output_path is None:
        output_path = file_path
        
    
    with open(file_path, 'r') as f:
        molecules = [line.strip() for line in f if line.strip()]
        
    total = len(molecules)
    if total <= max_size:  
        return file_path        
    
    import random
    selected = random.sample(molecules, max_size)    
    
    with open(output_path, 'w') as f:
        for mol in selected:
            f.write(f"{mol}\n")
            
    print(f"种群大小已从{total}限制为{max_size}")
    return output_path

def determine_seed_numbers(args, generation_num):
    """
    根据代数动态确定种子数量
    
    Args:
        args: 命令行参数
        generation_num: 当前代数
    
    Returns:
        tuple: (fitness_seeds, diversity_seeds) 适应度种子数量和多样性种子数量
    """
    # 计算多样性种子数量
    diversity_depreciation = int(generation_num) * args.diversity_seed_depreciation_per_gen
    diversity_seeds = max(0, args.diversity_mols_to_seed_first_generation - diversity_depreciation)
    
    # 确定适应度种子数量
    if generation_num == 0:
        # 第一代使用特殊参数
        if args.top_mols_to_seed_next_generation_first_generation is not None:
            fitness_seeds = args.top_mols_to_seed_next_generation_first_generation
        else:
            fitness_seeds = args.top_mols_to_seed_next_generation
    else:
        # 后续代使用标准参数
        fitness_seeds = args.top_mols_to_seed_next_generation
    
    return fitness_seeds, diversity_seeds

def run_generation(gen_num, prev_gen_dir, output_dir, num_mutations, num_crossovers, logger, args):
    """
    运行一代进化过程
    
    Args:
        gen_num: 当前代数
        prev_gen_dir: 上一代结果目录
        output_dir: 输出目录
        num_mutations: 需要生成的变异产物数量
        num_crossovers: 需要生成的交叉产物数量
        logger: 日志记录器
        args: 命令行参数
    
    Returns:
        包含所有生成分子的文件路径
    """
    logger.info(f"开始第 {gen_num} 代进化")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 动态计算种子数量
    fitness_seeds, diversity_seeds = determine_seed_numbers(args, gen_num)
    logger.info(f"第 {gen_num} 代种子选择: 适应度种子 {fitness_seeds} 个, 多样性种子 {diversity_seeds} 个")
    
    # 1. 选择种子分子
    seed_file = os.path.join(output_dir, f"generation_{gen_num}_seeds.smi")
    seed_list = select_seed_molecules(prev_gen_dir, fitness_seeds, diversity_seeds, args)
    
    # 保存种子分子到文件
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")
        
    # 2. 分子分解
    decompose_output = run_decompose(seed_file, f"gen{gen_num}", logger)
        
    # 3. GPT生成
    gpt_output = run_gpt_generation(decompose_output, f"gen{gen_num}", gen_num, logger)
        
    # 4. 交叉
    crossover_output = run_crossover_with_seeds(prev_gen_dir, gpt_output, os.path.join(output_dir, f"generation_{gen_num}_crossover.smi"), 
                                              seed_list, num_crossovers, gen_num, logger)
    
    # 5. 变异
    mutation_output = run_mutation_with_seeds(prev_gen_dir, gpt_output, os.path.join(output_dir, f"generation_{gen_num}_mutation.smi"), 
                                            seed_list, num_mutations, gen_num, logger)
    
    # 6. 合并所有生成的分子
    all_molecules_file = os.path.join(output_dir, f"generation_{gen_num}_all.smi")
    with open(all_molecules_file, 'w') as outfile:
        # 写入种子分子
        with open(seed_file, 'r') as f:
            outfile.write(f.read())
    
        # 写入交叉产物
        with open(crossover_output, 'r') as f:
            outfile.write(f.read())
        
        # 写入变异产物
        with open(mutation_output, 'r') as f:
            outfile.write(f.read())
    
    logger.info(f"第 {gen_num} 代进化完成，生成文件: {all_molecules_file}")
    return all_molecules_file

def run_evolution(generation_num, args, logger):
    """
    执行一次完整的进化迭代
    
    Args:
        generation_num: 当前代数
        args: 命令行参数
        logger: 日志记录器
    
    Returns:
        包含所有生成分子的文件路径
    """
    logger.info(f"开始第 {generation_num} 代进化")
    
    # 创建各代输出目录
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)
    
    # 确定当前代的种群文件
    if generation_num == 0:
        # 第一代使用初始种群
        current_population = args.initial_population
        # 第0代直接使用源文件所有分子作为种子
        seed_list = get_molecules_from_file(current_population)
        logger.info(f"第0代：使用源文件所有 {len(seed_list)} 个分子作为种子")
    else:
        # 后续代使用上一代的对接结果
        current_population = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_docked.smi")
        # 选择种子分子
        num_seeds = args.top_mols_to_seed_next_generation
        diversity_seeds = max(0, args.diversity_mols_to_seed_first_generation - (generation_num - 1) * args.diversity_seed_depreciation_per_gen)
        seed_list = select_seed_molecules(current_population, num_seeds, diversity_seeds, args)
        if not seed_list:
            logger.error("无法选择种子分子，进化终止")
            return None
        logger.info(f"已选择 {len(seed_list)} 个种子分子")
    
    # 保存种子分子到文件
    seed_file = os.path.join(output_base, f"generation_{generation_num}_seeds.smi")
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")
    
    # 设置各阶段输出文件
    crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
    mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
    merged_output = os.path.join(output_base, f"generation_{generation_num}_merged.smi")
    filter_output = os.path.join(output_base, f"generation_{generation_num}_filtered.smi")
    docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
    
    # 1. 第一次分子分解（用于交叉）
    decompose_output1 = run_decompose(seed_file, f"crossover{generation_num}", logger)
    
    # 2. 第一次GPT生成（用于交叉）
    gpt_output1 = run_gpt_generation(decompose_output1, f"crossover{generation_num}", generation_num, logger)
    
    # 3. 分子交叉
    logger.info(f"开始分子交叉，目标生成 {args.num_crossovers} 个新分子")
    crossover_output = run_crossover_with_seeds(seed_file, gpt_output1, crossover_output, seed_list, args.num_crossovers, generation_num, logger)
    
    # 4. 第二次分子分解（用于变异）
    decompose_output2 = run_decompose(seed_file, f"mutation{generation_num}", logger)
    
    # 5. 第二次GPT生成（用于变异）
    gpt_output2 = run_gpt_generation(decompose_output2, f"mutation{generation_num}", generation_num, logger)
    
    # 6. 分子变异
    logger.info(f"开始分子变异，目标生成 {args.num_mutations} 个新分子")
    mutation_output = run_mutation_with_seeds(seed_file, gpt_output2, mutation_output, seed_list, args.num_mutations, generation_num, logger)
    
    # 7. 合并新生成的分子（只包含交叉和变异产物）
    with open(merged_output, 'w') as outfile:
        # 写入交叉产物
        with open(crossover_output, 'r') as f:
            outfile.write(f.read())
        # 写入变异产物
        with open(mutation_output, 'r') as f:
            outfile.write(f.read())
    
    logger.info(f"合并后的种群大小: {sum(1 for _ in open(merged_output))} 个新分子")
    
    # 8. 分子过滤
    filter_output = run_filter(merged_output, filter_output, logger, args)
    
    # 9. 分子对接
    docking_output = run_docking(
        filter_output, 
        docking_output, 
        args.receptor_file, 
        args.mgltools_path, 
        logger,
        args.number_of_processors,
        args.multithread_mode
    )
    
    # 10. 对接结果分析
    analysis_output = run_analysis(docking_output, output_base, generation_num, logger)
    
    # 11. 计算并输出统计信息
    calculate_and_print_stats(docking_output, generation_num, logger)
    
    logger.info(f"第 {generation_num} 代进化完成")
    return analysis_output

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm_finetune - 优化后的分子进化与生成流程')
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=5, 
                        help='进化代数(不包括第0代)')
    parser.add_argument('--output_dir', type=str, default='/data1/tgy/GA_llm/output',
                        help='输出目录')
    parser.add_argument('--initial_population', type=str, 
                        default='/data1/tgy/GA_llm/datasets/source_compounds/naphthalene_smiles.smi',
                        help='初始种群文件路径')
    
    # 种子选择参数 
    parser.add_argument('--selector_choice', type=str, default="Rank_Selector",
                        choices=["Rank_Selector", "Roulette_Selector"],
                        help='选择器类型：Rank_Selector(排名选择) 或 Roulette_Selector(轮盘选择)')
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=10,
                        help='每代选择作为种子的分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=10,
                        help='第一代基于多样性选择的种子数量')
    parser.add_argument('--diversity_seed_depreciation_per_gen', type=int, default=2,
                        help='每代多样性种子数量的递减值')
    
    # 第一代特殊参数
    parser.add_argument('--number_of_crossovers_first_generation', type=int,
                        help='第一代通过交叉产生的配体数量')
    parser.add_argument('--number_of_mutants_first_generation', type=int,
                        help='第一代通过变异产生的配体数量')
    
    # 后续代参数
    parser.add_argument('--num_crossovers', type=int, default=10,
                        help='每代通过交叉产生的配体数量')
    parser.add_argument('--num_mutations', type=int, default=10,
                        help='每代通过变异产生的配体数量')
    
    # 对接参数
    parser.add_argument('--receptor_file', type=str,
                        default='/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb',
                        help='受体PDB文件路径')
    parser.add_argument('--mgltools_path', type=str,
                        default='/data1/tgy/GA_llm/mgltools_x86_64Linux2_1.5.6',
                        help='MGLTools安装路径')
    
    # 并行处理参数
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于并行计算的处理器数量')
    parser.add_argument('--multithread_mode', default="multithreading",
                        choices=["mpi", "multithreading", "serial"],
                        help='多线程模式选择')
    
    # 过滤器参数
    parser.add_argument('--LipinskiStrictFilter', action='store_true', default=False,
                        help='使用严格版Lipinski过滤器')
    parser.add_argument('--LipinskiLenientFilter', action='store_true', default=False,
                        help='使用宽松版Lipinski过滤器')
    parser.add_argument('--GhoseFilter', action='store_true', default=False,
                        help='使用Ghose过滤器')
    parser.add_argument('--GhoseModifiedFilter', action='store_true', default=False,
                        help='使用修改版Ghose过滤器')
    parser.add_argument('--MozziconacciFilter', action='store_true', default=False,
                        help='使用Mozziconacci过滤器')
    parser.add_argument('--VandeWaterbeemdFilter', action='store_true', default=False,
                        help='使用VandeWaterbeemd过滤器')
    parser.add_argument('--PAINSFilter', action='store_true', default=False,
                        help='使用PAINS过滤器')
    parser.add_argument('--NIHFilter', action='store_true', default=False,
                        help='使用NIH过滤器')
    parser.add_argument('--BRENKFilter', action='store_true', default=False,
                        help='使用BRENK过滤器')
    parser.add_argument('--No_Filters', action='store_true', default=False,
                        help='不使用任何过滤器')
    parser.add_argument('--alternative_filter', action='append', default=[],
                        help='添加自定义过滤器')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)    
    
    # 执行多代进化
    for gen in range(args.generations + 1):
        logger = setup_logging(args.output_dir, gen)
        try:
            logger.info(f"开始第 {gen} 代进化")
            start_time = time.time()           
            
            final_output = run_evolution(gen, args, logger)
            
            end_time = time.time()
            logger.info(f"第 {gen} 代进化完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"结果保存至: {final_output}")
            
        except Exception as e:
            logger.error(f"第 {gen} 代进化失败: {str(e)}")
            break

if __name__ == "__main__":
    main()
