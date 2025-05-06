#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GA_llm_finetune.py - 优化的分子进化与生成流程整合脚本

完整流程: 
1. 读取当前种群
2. 分子分解(decompose)
3. GPT生成新分子
4. 精选种子分子
5. 基于种子进行交叉
6. 基于种子进行变异 
7. 过滤
8. 分子对接
9. 分析排名并选择下一代种子

处理流程说明:
- 优化了种群大小控制，防止种群规模暴增
- 引入了种子选择机制，提高进化效率
- 每代产生固定数量的新个体，而不是执行固定次数的操作
- Generation_0: 对初始种群进行处理后进行对接评分
- Generation_1到Generation_N: 执行完整的进化流程


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


PROJECT_ROOT = "/data1/tgy/GA_llm"
sys.path.insert(0, PROJECT_ROOT)


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
        "--seed", str(gen_num)  # 使用代数作为种子，确保每代生成不同结果
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

def select_seed_molecules(molecules, num_seed_by_fitness, num_seed_by_diversity, generation_num, logger, selector_choice="Roulette_Selector", tourn_size=0.1):
    """
    从分子列表中选择种子分子，同时考虑适应度和多样性
    
    Args:
        molecules: 带有得分的分子列表，格式为[(smiles, score), ...]
        num_seed_by_fitness: 基于适应度选择的种子数量
        num_seed_by_diversity: 基于多样性选择的种子数量
        generation_num: 当前代数
        logger: 日志记录器
        selector_choice: 选择器类型 ("Roulette_Selector", "Rank_Selector", "Tournament_Selector")
        tourn_size: 锦标赛选择器的大小参数
        
    Returns:
        选择的种子分子SMILES列表
    """
    logger.info(f"选择种子分子: 根据适应度选择 {num_seed_by_fitness} 个, 根据多样性选择 {num_seed_by_diversity} 个")
    logger.info(f"使用选择器: {selector_choice}")
    
    if not molecules:
        logger.warning("没有提供分子，无法选择种子")
        return []   
    
    # 基于适应度选择
    sorted_molecules = sorted(molecules, key=lambda x: x[1])    
    
    # 根据选择器类型选择fitness种子
    if selector_choice == "Rank_Selector":
        # 排名选择：直接选择排名靠前的分子
        fitness_seeds = [mol[0] for mol in sorted_molecules[:num_seed_by_fitness]]
    
    elif selector_choice == "Tournament_Selector":
        # 锦标赛选择
        fitness_seeds = []
        remaining_mols = sorted_molecules.copy()
        
        while len(fitness_seeds) < num_seed_by_fitness and remaining_mols:
            # 确定锦标赛大小
            tourn_pool_size = max(1, int(len(remaining_mols) * tourn_size))
            # 随机选择锦标赛池
            tourn_pool = random.sample(remaining_mols, min(tourn_pool_size, len(remaining_mols)))
            # 选择锦标赛池中得分最好的分子
            winner = min(tourn_pool, key=lambda x: x[1])
            fitness_seeds.append(winner[0])
            remaining_mols.remove(winner)
    
    else:  # Roulette_Selector
        # 轮盘赌选择
        fitness_seeds = []
        remaining_mols = sorted_molecules.copy()
        
        while len(fitness_seeds) < num_seed_by_fitness and remaining_mols:
            # 计算适应度权重（得分越低权重越大）
            max_score = max(mol[1] for mol in remaining_mols)
            weights = [max_score - mol[1] + 1e-10 for mol in remaining_mols]  # 添加小值避免权重为0
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            # 根据权重随机选择
            selected_idx = random.choices(range(len(remaining_mols)), weights=weights)[0]
            fitness_seeds.append(remaining_mols[selected_idx][0])
            remaining_mols.pop(selected_idx)
    
    # 如果没有多样性选择需求，直接返回适应度种子
    if num_seed_by_diversity <= 0:
        return fitness_seeds   
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError:
        logger.warning("无法导入RDKit,多样性选择将基于随机抽样")
        remaining = [mol[0] for mol in sorted_molecules if mol[0] not in fitness_seeds]
        if remaining:
            diversity_seeds = random.sample(
                remaining, 
                min(num_seed_by_diversity, len(remaining))
            )
            return fitness_seeds + diversity_seeds
        return fitness_seeds
    
    # 计算分子指纹并选择多样性种子
    try:        
        remaining_mols = []
        remaining_smiles = []
        for mol_smile, _ in sorted_molecules:
            if mol_smile not in fitness_seeds:
                mol = Chem.MolFromSmiles(mol_smile)
                if mol:
                    remaining_mols.append(mol)
                    remaining_smiles.append(mol_smile)
        
        if not remaining_mols:
            return fitness_seeds        
        
        # 计算Morgan指纹
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in remaining_mols]
        
        # 计算已选fitness种子的指纹
        fitness_mol_fps = []        
        for smile in fitness_seeds:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                fitness_mol_fps.append(fp)
        
        # 使用相同的选择器机制选择多样性种子
        diversity_seeds = []
        remaining_fps = list(zip(remaining_smiles, fingerprints))
        
        if selector_choice == "Rank_Selector":
            # 计算每个分子与已选种子的最小相似度
            min_similarities = []
            for smile, fp in remaining_fps:
                similarities = [DataStructs.TanimotoSimilarity(fp, f) for f in fitness_mol_fps]
                min_similarities.append((smile, min(similarities)))
            
            # 按最小相似度排序（相似度越低越好）
            min_similarities.sort(key=lambda x: x[1])
            diversity_seeds = [x[0] for x in min_similarities[:num_seed_by_diversity]]
            
        elif selector_choice == "Tournament_Selector":
            while len(diversity_seeds) < num_seed_by_diversity and remaining_fps:
                tourn_pool_size = max(1, int(len(remaining_fps) * tourn_size))
                tourn_pool = random.sample(remaining_fps, min(tourn_pool_size, len(remaining_fps)))
                
                # 计算每个候选分子与已选种子的最小相似度
                min_similarities = []
                for smile, fp in tourn_pool:
                    similarities = [DataStructs.TanimotoSimilarity(fp, f) for f in fitness_mol_fps]
                    min_similarities.append((smile, min(similarities)))
                
                # 选择最小相似度最大的分子（最不相似的）
                winner = max(min_similarities, key=lambda x: x[1])
                diversity_seeds.append(winner[0])
                remaining_fps = [(s, f) for s, f in remaining_fps if s != winner[0]]
        
        else:  # Roulette_Selector
            while len(diversity_seeds) < num_seed_by_diversity and remaining_fps:
                # 计算每个分子与已选种子的最小相似度
                min_similarities = []
                for smile, fp in remaining_fps:
                    similarities = [DataStructs.TanimotoSimilarity(fp, f) for f in fitness_mol_fps]
                    min_similarities.append((smile, min(similarities)))
                
                # 将最小相似度转换为权重（相似度越低权重越大）
                max_sim = max(sim for _, sim in min_similarities)
                weights = [max_sim - sim + 1e-10 for _, sim in min_similarities]
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
                
                # 根据权重随机选择
                selected_idx = random.choices(range(len(remaining_fps)), weights=weights)[0]
                diversity_seeds.append(remaining_fps[selected_idx][0])
                remaining_fps.pop(selected_idx)
        
        return fitness_seeds + diversity_seeds
    
    except Exception as e:
        logger.error(f"多样性选择时出错: {str(e)}")
        # 出错时随机选择作为备选方案
        remaining = [mol[0] for mol in sorted_molecules if mol[0] not in fitness_seeds]
        if remaining:
            diversity_seeds = random.sample(
                remaining, 
                min(num_seed_by_diversity, len(remaining))
            )
            return fitness_seeds + diversity_seeds
        return fitness_seeds

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
    logger.info(f"开始分子交叉: 使用 {len(seed_list)} 个种子分子生成 {num_crossovers} 个新分子")
        
    seed_file = os.path.join(os.path.dirname(output_file), f"generation_{gen_num}_crossover_seeds.smi")
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")    
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)   
    
    crossover_script = os.path.join(PROJECT_ROOT, "operations/crossover/crossover_demo_finetune.py")
    cmd = [
        "python", crossover_script,
        "--seed_file", seed_file,
        "--source_compound_file", source_file,
        "--llm_generation_file", llm_file,
        "--output_file", output_file,
        "--crossover_rate", "0.8",
        "--crossover_attempts", str(num_crossovers)  # 需要生成的分子数量
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
    logger.info(f"开始分子变异: 使用 {len(seed_list)} 个种子分子生成 {num_mutations} 个新分子")
        
    seed_file = os.path.join(os.path.dirname(output_file), f"generation_{gen_num}_mutation_seeds.smi")
    with open(seed_file, 'w') as f:
        for smile in seed_list:
            f.write(f"{smile}\n")
    
   
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  
   
    mutation_script = os.path.join(PROJECT_ROOT, "operations/mutation/mutation_demo_finetune.py")
    cmd = [
        "python", mutation_script,
        "--seed_file", seed_file,
        "--input_file", input_file,
        "--llm_generation_file", llm_file,
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
    确定当前代需要的种子数量
    
    Args:
        args: 命令行参数
        generation_num: 当前代数
    
    Returns:
        (top_seeds, diversity_seeds) 元组，表示基于适应度和多样性选择的种子数量
    """
    
    if generation_num == 1:
        top_seeds = args.top_mols_to_seed_next_generation_first_generation if args.top_mols_to_seed_next_generation_first_generation is not None else args.top_mols_to_seed_next_generation
        diversity_seeds = args.diversity_mols_to_seed_first_generation
    else:
        top_seeds = args.top_mols_to_seed_next_generation        
        diversity_depreciation = (generation_num - 1) * args.diversity_seed_depreciation_per_gen
        diversity_seeds = max(0, args.diversity_mols_to_seed_first_generation - diversity_depreciation)
    
    return top_seeds, diversity_seeds

def run_evolution(generation_num, args, logger):
    """执行一次完整的进化迭代"""
    logger.info(f"开始第 {generation_num} 代进化")
    
    # 创建各代输出目录
    output_base = os.path.join(args.output_dir, f"generation_{generation_num}")
    os.makedirs(output_base, exist_ok=True)
    
    # 确定当前代的种群文件
    if generation_num == 0:
        # 第一代使用初始种群
        current_population = args.initial_population
    else:
        # 后续代使用上一代的对接结果
        current_population = os.path.join(args.output_dir, f"generation_{generation_num-1}", f"generation_{generation_num-1}_docked.smi")
        
        # 检查上一代是否有结果
        if not os.path.exists(current_population) or os.path.getsize(current_population) == 0:
            logger.error(f"上一代没有有效的对接结果，无法继续进化")
            empty_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
            with open(empty_output, 'w') as f:
                pass
            return empty_output
    
    # 设置各阶段输出文件
    crossover_output = os.path.join(output_base, f"generation_{generation_num}_crossover.smi")
    mutation_output = os.path.join(output_base, f"generation_{generation_num}_mutation.smi")
    merged_output = os.path.join(output_base, f"generation_{generation_num}_merged.smi")
    filter_output = os.path.join(output_base, f"generation_{generation_num}_filtered.smi")
    docking_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
    
    # 特殊处理第0代
    if generation_num == 0:
        logger.info("Generation 0: 对初始文件进行处理后进行对接")
        
        # 1. 分子分解
        decompose_output = run_decompose(current_population, f"gen{generation_num}", logger)
        
        # 2. GPT生成
        gpt_output = run_gpt_generation(decompose_output, f"{generation_num}", generation_num, logger)
        
        # 3. 分子过滤
        filter_output = run_filter(current_population, filter_output, logger, args)
        
        # 4. 简化版分子对接
        success = run_docking(
            filter_output, 
            docking_output, 
            args.receptor_file, 
            args.mgltools_path, 
            logger,
            args.number_of_processors,
            args.multithread_mode
        )
        
        # 确保产生了对接结果，如果没有，创建一个假对接结果文件包含初始分子
        if not success:
            logger.warning("对接未产生有效结果，使用初始分子作为备选")
            with open(current_population, 'r') as f_in, open(docking_output, 'w') as f_out:
                for i, line in enumerate(f_in):
                    if line.strip():
                        # 添加一个虚拟的得分(-10.0)
                        parts = line.strip().split()
                        smile = parts[0]
                        f_out.write(f"{smile}\t-10.0\n")
                        if i >= 19:  # 最多保留20个分子
                            break
        
        # 5. 对接结果分析
        analysis_output = run_analysis(docking_output, output_base, generation_num, logger)
        
        # 6. 计算统计信息
        calculate_and_print_stats(docking_output, generation_num, logger)
        
        logger.info(f"第 {generation_num} 代完成")
        return docking_output
    
    # 对于后续代数，执行完整的进化流程
    
    # 确定种子数量
    top_seeds, diversity_seeds = determine_seed_numbers(args, generation_num)
    total_seeds = top_seeds + diversity_seeds
    
    logger.info(f"第 {generation_num} 代种子设置: "
               f"基于适应度选择 {top_seeds} 个, "
               f"基于多样性选择 {diversity_seeds} 个")
    
    # 1. 获取上一代分子及评分
    previous_docked_molecules = get_molecules_from_file(current_population, with_score=True)
    
    # 2. 选择种子分子
    seed_list = select_seed_molecules(
        previous_docked_molecules, 
        top_seeds, 
        diversity_seeds, 
        generation_num, 
        logger,
        args.selector_choice,
        args.tourn_size
    )
    
    # 处理没有种子的情况
    if len(seed_list) == 0:
        logger.warning("没有找到有效的种子分子，使用初始分子作为备选")
        with open(args.initial_population, 'r') as f:
            fallback_molecules = [line.strip().split()[0] for line in f if line.strip()][:total_seeds]
            if fallback_molecules:
                seed_list = fallback_molecules
                logger.info(f"使用{len(seed_list)}个初始分子作为种子")
            else:
                logger.error(f"无法找到任何种子分子，第{generation_num}代进化失败")
                empty_output = os.path.join(output_base, f"generation_{generation_num}_docked.smi")
                with open(empty_output, 'w') as f:
                    pass
                return empty_output
    
    # 保存种子列表
    seed_file = save_seed_list(output_base, generation_num, seed_list)
    logger.info(f"已选择 {len(seed_list)} 个种子分子，保存至: {seed_file}")
    
    # 确定新一代需要生成的分子数量
    num_crossovers = args.num_crossovers 
    num_mutations = args.num_mutations
    
    # 3. 分子分解
    decompose_output = run_decompose(seed_file, f"gen{generation_num}", logger)
    
    # 4. GPT生成
    try:
        gpt_output = run_gpt_generation(decompose_output, f"{generation_num}", generation_num, logger)
    except Exception as e:
        logger.warning(f"GPT生成失败,使用种子文件作为替代: {str(e)}")
        gpt_output = seed_file
    
    # 5. 分子交叉
    try:
        crossover_output = run_crossover_with_seeds(
            current_population, 
            gpt_output, 
            crossover_output, 
            seed_list, 
            num_crossovers, 
            generation_num, 
            logger
        )
    except Exception as e:
        logger.warning(f"交叉操作失败: {str(e)}，使用上一代结果")
        crossover_output = current_population
    
    # 6. 分子变异
    try:
        mutation_output = run_mutation_with_seeds(
            current_population, 
            gpt_output, 
            mutation_output, 
            seed_list, 
            num_mutations, 
            generation_num, 
            logger
        )
    except Exception as e:
        logger.warning(f"变异操作失败: {str(e)}，使用上一代结果")
        mutation_output = current_population
    
    # 7. 合并交叉和变异结果
    crossover_molecules = get_molecules_from_file(crossover_output)
    mutation_molecules = get_molecules_from_file(mutation_output)
    
    # 确保有分子可用
    if not crossover_molecules and not mutation_molecules:
        logger.warning("交叉和变异都未产生有效分子，使用上一代分子")
        previous_molecules = get_molecules_from_file(current_population)
        with open(merged_output, 'w') as f:
            for smile in previous_molecules:
                f.write(f"{smile}\n")
    else:
        with open(merged_output, 'w') as f:
            for smile in crossover_molecules + mutation_molecules:
                f.write(f"{smile}\n")
    
    logger.info(f"合并交叉和变异结果: 交叉 {len(crossover_molecules)} 个, 变异 {len(mutation_molecules)} 个")
    
    # 8. 分子过滤
    try:
        filter_output = run_filter(merged_output, filter_output, logger, args)
    except Exception as e:
        logger.warning(f"过滤操作失败: {str(e)}，使用合并结果")
        filter_output = merged_output
    
    # 9. 分子对接
    success = run_docking(
        filter_output, 
        docking_output, 
        args.receptor_file, 
        args.mgltools_path, 
        logger,
        args.number_of_processors,
        args.multithread_mode
    )
    
    # 确保产生了对接结果
    if not success:
        logger.warning("对接未产生有效结果，使用上一代结果作为备选")
        if os.path.exists(current_population) and os.path.getsize(current_population) > 0:
            import shutil
            shutil.copy(current_population, docking_output)
            logger.info(f"复制上一代结果到当前代: {current_population} -> {docking_output}")
    
    # 10. 对接结果分析
    analysis_output = run_analysis(docking_output, output_base, generation_num, logger)
    
    # 11. 计算并输出统计信息
    calculate_and_print_stats(docking_output, generation_num, logger)
    
    logger.info(f"第 {generation_num} 代进化完成")
    return docking_output

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA_llm_finetune - 优化的分子进化与生成流程')
    
    # 基本参数
    parser.add_argument('--generations', type=int, default=5, 
                        help='generation_0到generation_5)')
    parser.add_argument('--output_dir', type=str, default='/data1/tgy/GA_llm/output_finetune/',
                        help='输出目录')
    parser.add_argument('--initial_population', type=str, 
                        default='/data1/tgy/GA_llm/datasets/source_compounds/naphthalene_smiles.smi'
                        )
    
    # 对接参数
    parser.add_argument('--receptor_file', type=str,
                        default='/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb'
                        )
    parser.add_argument('--mgltools_path', type=str,
                        default='/data1/tgy/GA_llm/mgltools_x86_64Linux2_1.5.6/')
    
    # 进化参数 
    parser.add_argument('--num_crossovers', type=int, default=10,
                       help='每代通过交叉产生的新配体数量(第1代及以后)')
    parser.add_argument('--num_mutations', type=int, default=10,
                       help='每代通过变异产生的新配体数量(第1代及以后)')
    parser.add_argument('--number_of_crossovers_first_generation', type=int,
                       help='第0代中通过交叉产生的配体数量,如果未指定则默认使用num_crossovers的值')
    parser.add_argument('--number_of_mutants_first_generation', type=int,
                       help='第0代中通过变异产生的配体数量,如果未指定则默认使用num_mutations的值')
    
    # 种子选择参数 
    parser.add_argument('--top_mols_to_seed_next_generation_first_generation', type=int,
                       help='第一代中基于适应度选择的种子分子数量,如未指定则使用top_mols_to_seed_next_generation的值')
    parser.add_argument('--top_mols_to_seed_next_generation', type=int, default=10,
                       help='后续各代中基于适应度选择的种子分子数量')
    parser.add_argument('--diversity_mols_to_seed_first_generation', type=int, default=10,
                       help='第一代中基于多样性选择的种子分子数量')
    parser.add_argument('--diversity_seed_depreciation_per_gen', type=int, default=2,
                       help='每代多样性种子数量的减少量')
    
    # 选择器参数
    parser.add_argument('--selector_choice', 
                       choices=["Roulette_Selector", "Rank_Selector", "Tournament_Selector"],
                       default="Roulette_Selector",
                       help='选择器类型：轮盘赌选择(Roulette)、排名选择(Rank)或锦标赛选择(Tournament)')
    parser.add_argument('--tourn_size', type=float, default=0.1,
                       help='如果使用锦标赛选择器,这决定了每个锦标赛的大小。每个锦标赛使用的配体数量将是tourn_size * 考虑的配体数量')
    
    parser.add_argument('--max_population', type=int, default=0,
                       help='控制每代种群的最大数量,设置为0表示不限制')
    
    
    parser.add_argument('--number_of_processors', '-p', type=int, default=-1,
                        help='用于并行计算的处理器数量。设置为-1表示自动检测并使用所有可用CPU核心(推荐）。')
    parser.add_argument('--multithread_mode', default="multithreading",
                        choices=["mpi", "multithreading", "serial"],
                        help='多线程模式选择: mpi, multithreading, 或 serial。serial模式将忽略处理器数量设置,强制使用单处理器。')
    
    
    parser.add_argument('--LipinskiStrictFilter', action='store_true', default=False,
                        help='严格版Lipinski五规则过滤器,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。要求必须通过所有条件。')
    parser.add_argument('--LipinskiLenientFilter', action='store_true', default=False,
                        help='宽松版Lipinski五规则过滤器,筛选口服可用药物。评估分子量、logP、氢键供体和受体数量。允许一个条件不满足。')
    parser.add_argument('--GhoseFilter', action='store_true', default=False,
                        help='Ghose药物相似性过滤器,通过分子量、logP和原子数量进行筛选。')
    parser.add_argument('--GhoseModifiedFilter', action='store_true', default=False,
                        help='修改版Ghose过滤器,将分子量上限从480Da放宽到500Da。设计用于与Lipinski过滤器配合使用。')
    parser.add_argument('--MozziconacciFilter', action='store_true', default=False,
                        help='Mozziconacci药物相似性过滤器,评估可旋转键、环、氧原子和卤素原子的数量。')
    parser.add_argument('--VandeWaterbeemdFilter', action='store_true', default=False,
                        help='筛选可能透过血脑屏障的药物，基于分子量和极性表面积(PSA)。')
    parser.add_argument('--PAINSFilter', action='store_true', default=False,
                        help='PAINS过滤器,用于过滤泛测试干扰化合物，使用子结构搜索。')
    parser.add_argument('--NIHFilter', action='store_true', default=False,
                        help='NIH过滤器,过滤含有不良功能团的分子,使用子结构搜索。')
    parser.add_argument('--BRENKFilter', action='store_true', default=False,
                        help='BRENK前导物相似性过滤器,排除常见假阳性分子。')
    parser.add_argument('--No_Filters', action='store_true', default=False,
                        help='设置为True时,不应用任何过滤器。')
    parser.add_argument('--alternative_filter', action='append',
                        help='添加自定义过滤器，需要提供列表格式：[[过滤器1名称, 过滤器1路径], [过滤器2名称, 过滤器2路径]]')
    
    parser.add_argument('--max_failures', type=int, default=5, help='每个分子的最大对接尝试次数')
    
    args = parser.parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)    
    
    # 如果没有指定number_of_crossovers_first_generation，使用num_crossovers的值
    if args.number_of_crossovers_first_generation is None:
        args.number_of_crossovers_first_generation = args.num_crossovers
    
    # 如果没有指定number_of_mutants_first_generation，使用num_mutations的值
    if args.number_of_mutants_first_generation is None:
        args.number_of_mutants_first_generation = args.num_mutations
    
    # 如果没有指定top_mols_to_seed_next_generation_first_generation，使用top_mols_to_seed_next_generation的值
    if args.top_mols_to_seed_next_generation_first_generation is None:
        args.top_mols_to_seed_next_generation_first_generation = args.top_mols_to_seed_next_generation
    
   
    if args.number_of_processors != 1 and args.multithread_mode == "serial":
        print(f"检测到可能使用多核但模式为serial,自动切换为multithreading模式")
        args.multithread_mode = "multithreading"    
    
    if args.max_population > 0:
        
        with open(args.initial_population, 'r') as f:
            initial_count = sum(1 for line in f if line.strip())
        if initial_count > args.max_population:
            limited_file = os.path.join(args.output_dir, "limited_initial_population.smi")
            args.initial_population = limit_population_size(args.initial_population, args.max_population, limited_file)
            print(f"初始种群已从{initial_count}限制为{args.max_population}")
        
    logger = setup_logging(args.output_dir, 0)
    try:
        logger.info(f"开始第0代(对初始种群进行处理后对接)")
        start_time = time.time()
        
        run_evolution(0, args, logger)
        
        end_time = time.time()
        logger.info(f"第0代完成,耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        logger.error(f"第0代失败: {str(e)}")    
    
    for gen in range(1, args.generations + 1):
        logger = setup_logging(args.output_dir, gen)
        try:
            logger.info(f"开始第 {gen} 代进化")
            start_time = time.time()           
            
            if args.max_population > 0:
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
