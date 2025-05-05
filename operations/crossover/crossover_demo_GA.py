import sys
import os
PROJECT_ROOT = "/data1/tgy/GA_llm"
sys.path.insert(0, PROJECT_ROOT)
from tdc import Evaluator, Oracle  
import random
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge 
import autogrow.operators.crossover.execute_crossover as execute_crossover
import autogrow.operators.filter.execute_filters as Filter


PARSER = argparse.ArgumentParser()

PARSER = argparse.ArgumentParser(description='GA crossover parameters for pure GA implementation')
PARSER.add_argument("--source_compound_file", "-s", type=str, required=True, help="源分子数据集路径")
PARSER.add_argument("--output_file", "-o", type=str, default="/data1/tgy/GA_llm/output/generation_crossover_0.smi", help="输出文件路径")
PARSER.add_argument("--crossover_rate", type=float, default=0.8, help="交叉率")
PARSER.add_argument("--crossover_attempts", type=int, default=1, help="设置交叉次数（尝试交叉次数)")

# 初始化评估器
def init_evaluators():
    try:
        div_evaluator = Evaluator(name='Diversity')
        nov_evaluator = Evaluator(name='Novelty') 
        qed_evaluator = Oracle(name='qed')
        sa_evaluator = Oracle(name='sa')
        return div_evaluator, nov_evaluator, qed_evaluator, sa_evaluator
    except ImportError:
        print("请先安装TDC包:pip install tdc")
        exit(1)

# 评估种群函数
def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    results = {
        'diversity': div_eval(smiles_list),
        'novelty': nov_eval(smiles_list, ref_smiles),
        'avg_qed': np.mean([qed_eval(s) for s in smiles_list]),
        'avg_sa': np.mean([sa_eval(s) for s in smiles_list]),
        'num_valid': len(smiles_list)
    }
    return results

# 主逻辑修改 - 适配纯GA流程
def main():
    args = PARSER.parse_args()
    
    # 加载源分子数据集
    source_smiles = []
    with open(args.source_compound_file, 'r') as f:
        source_smiles = [line.split()[0].strip() for line in f if line.strip()]
        
    initial_population = list(source_smiles)  # 初始种群
    
    # 初始化评估器
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    print('*****************************初始评估*******************************')
    
    # 评估初始种群
    initial_metrics = evaluate_population(initial_population, div_eval, nov_eval, 
                                        qed_eval, sa_eval, source_smiles)
    print(f"初始种群评估结果:\n{initial_metrics}")
    print('*****************************初始评估完成*******************************')
    print('*******************************开始交叉*********************************')
    
    # 执行交叉操作
    crossed_population = []   # 交叉后新种群
    vars = {
        'min_atom_match_mcs': 4,
        'max_time_mcs_prescreen': 1, # MCS预筛选最大时间（秒）
        'max_time_mcs_thorough': 1,  # MCS详细计算阶段最大时间（秒）
        'protanate_step': False,     # 是否执行质子化步骤
        'number_of_crossovers': args.crossover_attempts,
        'filter_object_dict': {},    # 过滤对象字典     
        'debug_mode': False,
        'gypsum_timeout_limit': 15.0, # 分子构象生成超时时间限制（秒）
        "--max_variants_per_compound": 3, # 每个配体的构象数量
        'min_ph': 6.4,  # 最小pH值（用于质子化状态）
        'max_ph': 8.4,  # 最大pH值
        'pka_precision': 1.0 # pKa精度    
    }

    with tqdm(total=args.crossover_attempts, desc="Performing crossovers") as pbar:
        while len(crossed_population) < args.crossover_attempts:
            # 随机选择两个父代分子
            parent1, parent2 = random.sample(initial_population, 2)
            
            try:
                # 转换SMILES为分子对象
                mol1 = execute_crossover.convert_mol_from_smiles(parent1)
                mol2 = execute_crossover.convert_mol_from_smiles(parent2)
                if mol1 is None or mol2 is None:
                    continue
                    
                # 检查MCS（最大公共子结构）
                mcs_result = execute_crossover.test_for_mcs(vars, mol1, mol2)
                if mcs_result is None:
                    continue  # 没有足够大的公共结构
            
                # 多次尝试合并
                ligand_new_smiles = None
                # 尝试3次交叉合并，直到成功为止
                for attempt in range(3):
                    ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
                    if ligand_new_smiles is not None:
                        break
                        
                if ligand_new_smiles is None:
                    continue
                    
                # 过滤新生成的分子 
                if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
                    crossed_population.append(ligand_new_smiles)
                    pbar.update(1)
            except Exception as e:
                # 捕获任何可能的异常并继续
                continue
    
    # 保存新生成的crossed_population到临时文件
    temp_crossed_file = "/data1/tgy/GA_llm/output/generation_0_crossed_new.smi"
    os.makedirs(os.path.dirname(temp_crossed_file), exist_ok=True)
    with open(temp_crossed_file, 'w') as f:
        for smi in crossed_population:
            f.write(f"{smi}\n")
            
    print('*******************************交叉完成*********************************')
    
    # 合并种群
    new_population = initial_population + crossed_population
    print('*******************************交叉后新种群评估*********************************')
    
    # 评估交叉后新种群
    crossed_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                        qed_eval, sa_eval, source_smiles)
    print(f"\n交叉后新种群(聚合初始与新生成分子群)评估结果:\n{crossed_metrics}")
    
    # 评估仅包含新生成分子的种群
    crossed_new_metrics = evaluate_population(crossed_population, div_eval, nov_eval,
                                        qed_eval, sa_eval, source_smiles)
    print(f"\n交叉后新生成分子群评估结果:\n{crossed_new_metrics}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")

if __name__ == "__main__":
    main()
