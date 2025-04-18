#!/usr/bin/env python
import sys
import os
import time
import argparse
import multiprocessing  # 添加这行以支持自动检测CPU核心数
sys.path.insert(0, "/data1/ytg/GA_llm")
from GA_llm import setup_logging, run_evolution

# 创建与GA_llm.py中相同的参数解析器
def create_parser():
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
    
    return parser

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

if __name__ == "__main__":
    parser = create_parser()
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
    
    # 执行所有代数的进化 - 从第0代开始
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

# 示例运行命令：
# python run_remaining.py \
#   --generations 5 \
#   --num_crossovers 10 \
#   --num_mutations 10 \
#   --max_population 300 \
#   --output_dir /data1/ytg/GA_llm/output \
#   --initial_population /data1/ytg/GA_llm/datasets/source_compounds/naphthalene_smiles.smi \
#   --receptor_file /data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb \
#   --mgltools_path /data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6 \
#   --number_of_processors -1 \
#   --multithread_mode multithreading