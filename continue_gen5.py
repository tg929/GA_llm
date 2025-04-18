#!/usr/bin/env python
import sys, os
sys.path.insert(0, "/data1/tgy/GA_llm")
from GA_llm import run_mutation, run_filter, run_docking, run_analysis, calculate_and_print_stats, setup_logging
import argparse

# 设置参数
args = argparse.Namespace(
    output_dir="/data1/tgy/GA_llm/output",
    num_mutations=50,  # 使用原始参数
    receptor_file="/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb",  # 替换为您的受体文件
    mgltools_path="/data1/tgy/GA_llm/mgltools_x86_64Linux2_1.5.6",
    number_of_processors=-1,
    multithread_mode="multithreading",
    No_Filters=True  # 根据您的需求设置过滤器参数
)

# 设置文件路径
gen_num = 5
output_base = os.path.join(args.output_dir, f"generation_{gen_num}")
os.makedirs(output_base, exist_ok=True)

crossover_file = os.path.join(output_base, f"generation_{gen_num}_crossover.smi")
gpt_output = "/data1/tgy/GA_llm/fragment_GPT/output/crossovered0_frags_new_5.smi"
mutation_output = os.path.join(output_base, f"generation_{gen_num}_mutation.smi")
filter_output = os.path.join(output_base, f"generation_{gen_num}_filtered.smi")
docking_output = os.path.join(output_base, f"generation_{gen_num}_docked.smi")

# 设置日志
logger = setup_logging(args.output_dir, gen_num)

try:
    # 1. 执行变异
    logger.info("从变异步骤继续执行第5代...")
    mutation_output = run_mutation(crossover_file, gpt_output, mutation_output, args.num_mutations, logger)
    
    # 2. 执行过滤
    filter_output = run_filter(mutation_output, filter_output, logger, args)
    
    # 3. 执行对接
    docking_output = run_docking(
        filter_output, 
        docking_output, 
        args.receptor_file, 
        args.mgltools_path, 
        logger,
        args.number_of_processors,
        args.multithread_mode
    )
    
    # 4. 执行分析
    analysis_output = run_analysis(docking_output, output_base, gen_num, logger)
    
    # 5. 输出统计信息
    calculate_and_print_stats(docking_output, gen_num, logger)
    
    logger.info(f"第 {gen_num} 代恢复执行完成")
except Exception as e:
    logger.error(f"恢复执行失败: {str(e)}")
