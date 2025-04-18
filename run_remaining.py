#!/usr/bin/env python
import sys
import os
import time
sys.path.insert(0, "/data1/ytg/GA_llm")
from GA_llm import parser, setup_logging, run_evolution

if __name__ == "__main__":
    args = parser.parse_args()
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 只运行从第3代到第5代
    for gen in range(3, args.generations):
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

# 示例运行命令：
# python run_remaining.py \
#   --generations 5 \
#   --num_crossovers 50 \
#   --num_mutations 50 \
#   --output_dir /data1/ytg/GA_llm/output \
#   --initial_population /data1/ytg/GA_llm/datasets/source_compounds/naphthalene_smiles.smi \
#   --receptor_file /data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb \
#   --mgltools_path /data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6 \
#   --number_of_processors 8 \
#   --multithread_mode multithreading