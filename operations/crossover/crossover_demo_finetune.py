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

PARSER = argparse.ArgumentParser()

PARSER = argparse.ArgumentParser(description='GA crossover parameters')
PARSER.add_argument("--seed_file", "-sf", type=str, required=True, 
                    help="种子分子文件路径")
PARSER.add_argument("--source_compound_file", "-s", type=str, required=True,
                    help="上一代种群文件路径")
PARSER.add_argument("--llm_generation_file", "-l", type=str, required=True,
                    help="LLM生成的分子片段文件")
PARSER.add_argument("--output_file", "-o", type=str, required=True,
                    help="输出文件路径")
PARSER.add_argument("--crossover_rate", type=float, default=0.8,
                    help="交叉率")
PARSER.add_argument("--crossover_attempts", type=int, default=10,
                    help="需要生成的交叉产物数量")

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
    if not smiles_list:
        return {
            'diversity': 0.0,
            'novelty': 0.0,
            'avg_qed': 0.0,
            'avg_sa': 0.0,
            'num_valid': 0
        }
    
    # 计算多样性时需要至少2个样本
    diversity = div_eval(smiles_list) if len(smiles_list) >= 2 else 0.0
    
    # 计算新颖性时处理分母为零的情况
    try:
        novelty = nov_eval(smiles_list, ref_smiles)
    except ZeroDivisionError:
        novelty = 0.0
    
    results = {
        'diversity': diversity,
        'novelty': novelty,
        'avg_qed': np.mean([qed_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'avg_sa': np.mean([sa_eval(s) for s in smiles_list]) if smiles_list else 0.0,
        'num_valid': len(smiles_list)
    }
    return results

# 主逻辑
def main():
    args = PARSER.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载种子分子
        seed_smiles = []
        with open(args.seed_file, 'r') as f:
            seed_smiles = [line.strip() for line in f if line.strip()]
        
        if not seed_smiles:
            raise ValueError("种子文件为空")
        
        print(f"从种子文件读取了 {len(seed_smiles)} 个种子分子")
        
        # 加载源种群
        source_smiles = []
        with open(args.source_compound_file, 'r') as f:
            source_smiles = [line.split()[0].strip() for line in f if line.strip()]
        
        if not source_smiles:
            raise ValueError("源种群文件为空")
        
        print(f"从源文件读取了 {len(source_smiles)} 个分子")
        
        # 加载LLM生成分子
        llm_smiles = []
        with open(args.llm_generation_file, 'r') as f:
            llm_smiles = [line.strip() for line in f if line.strip()]
        
        print(f"从LLM生成文件读取了 {len(llm_smiles)} 个分子")
        
        # 合并种群用于评估
        full_population = list(set(source_smiles + llm_smiles))
        
        # 初始化评估器
        div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
        
        print("*****************************开始交叉*****************************")
        crossed_population = []  # 交叉后新种群
        
        vars = {
            'min_atom_match_mcs': 4,
            'max_time_mcs_prescreen': 1,
            'max_time_mcs_thorough': 1,
            'protanate_step': False,
            'number_of_crossovers': args.crossover_attempts,
            'filter_object_dict': {},
            'debug_mode': False,
            'gypsum_timeout_limit': 15.0,
            'max_variants_per_compound': 3,
            'min_ph': 6.4,
            'max_ph': 8.4,
            'pka_precision': 1.0    
        } 

        # 基于种子分子进行交叉
        max_tries = args.crossover_attempts * 10  # 设置最大尝试次数，避免无限循环
        tries = 0
        
        with tqdm(total=args.crossover_attempts, desc="Performing crossovers") as pbar:
            while len(crossed_population) < args.crossover_attempts and tries < max_tries:
                tries += 1            
                
                parent1 = random.choice(seed_smiles)
                
                # 有50%的概率从LLM生成结果中选择第二个分子，50%的概率从种子中选择
                if random.random() < 0.5 and llm_smiles:
                    parent2 = random.choice(llm_smiles)
                else:
                    # 从种子中选择不同于parent1的分子
                    if len(seed_smiles) > 1:
                        available_seeds = [s for s in seed_smiles if s != parent1]
                        parent2 = random.choice(available_seeds)
                    else:
                        # 只有一个种子，从源种群中选择
                        parent2 = random.choice(source_smiles)
                
                # 转换SMILES为分子对象
                try:
                    mol1 = execute_crossover.convert_mol_from_smiles(parent1)
                    mol2 = execute_crossover.convert_mol_from_smiles(parent2)
                    if mol1 is None or mol2 is None:
                        continue
                    
                    # 检查MCS（最大公共子结构）
                    mcs_result = execute_crossover.test_for_mcs(vars, mol1, mol2)
                    if mcs_result is None:
                        continue  
                except Exception as e:
                    print(f"分子转换或MCS检查失败: {str(e)}")
                    continue
                
                ligand_new_smiles = None
                for attempt in range(3):
                    try:
                        ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
                        if ligand_new_smiles is not None:
                            break
                    except Exception as e:
                        print(f"交叉尝试失败: {str(e)}")
                        continue
                
                if ligand_new_smiles is None:
                    continue
                
                try:
                    if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
                        if ligand_new_smiles not in source_smiles and ligand_new_smiles not in crossed_population:
                            crossed_population.append(ligand_new_smiles)
                            pbar.update(1)
                except Exception as e:
                    print(f"过滤检查失败: {str(e)}")
                    continue
        
        print(f"成功生成了 {len(crossed_population)} 个交叉产物，总尝试次数: {tries}")
        
        # 如果交叉产生的分子不足，使用随机选择的源种群分子补充
        if len(crossed_population) < args.crossover_attempts:
            needed = args.crossover_attempts - len(crossed_population)
            print(f"交叉产生的分子不足，需要从源种群随机选择 {needed} 个分子补充")
            
            available_mols = [mol for mol in source_smiles if mol not in crossed_population]
            
            if available_mols:
                selected_mols = random.sample(available_mols, min(needed, len(available_mols)))
                crossed_population.extend(selected_mols)
                print(f"已补充 {len(selected_mols)} 个分子")
            else:
                print("没有可用的源种群分子用于补充")
        
        # 保存结果
        with open(args.output_file, 'w') as f:
            for smi in crossed_population:
                f.write(f"{smi}\n")
        
        # 评估结果
        eval_results = evaluate_population(crossed_population, div_eval, nov_eval, qed_eval, sa_eval, full_population)
        print(f"交叉结果评估:")
        print(f"多样性: {eval_results['diversity']:.4f}")
        print(f"新颖性: {eval_results['novelty']:.4f}")
        print(f"平均QED: {eval_results['avg_qed']:.4f}")
        print(f"平均SA: {eval_results['avg_sa']:.4f}")
        print(f"有效分子数: {eval_results['num_valid']}")
        
    except Exception as e:
        print(f"交叉过程发生错误: {str(e)}")
        # 创建空的输出文件
        with open(args.output_file, 'w') as f:
            pass
        raise

if __name__ == "__main__":
    main() 