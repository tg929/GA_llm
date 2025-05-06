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
import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter

PARSER = argparse.ArgumentParser()
PARSER = argparse.ArgumentParser(description='GA mutation parameters')
PARSER.add_argument("--seed_file", "-sf", type=str, required=True, 
                    help="种子分子文件路径")
PARSER.add_argument("--input_file", "-i", type=str, required=True,
                    help="上一代种群文件路径")
PARSER.add_argument("--llm_generation_file", "-l", type=str, required=True,
                    help="LLM生成的分子片段文件")
PARSER.add_argument("--output_file", "-o", type=str, required=True,
                    help="输出文件路径")     
PARSER.add_argument("--mutation_attempts", type=int, default=10,
                    help="需要生成的变异产物数量")
PARSER.add_argument("--max_mutations", type=int, default=2, 
                    help="每个父代最大变异尝试次数")

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


def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    
    if len(smiles_list) == 0:
        return {
            'diversity': 0.0,
            'novelty': 0.0,
            'avg_qed': 0.0,
            'avg_sa': 0.0,
            'num_valid': 0
        }        
    # 计算多样性时需要至少2个样本
    diversity = div_eval(smiles_list) if len(smiles_list)>=2 else 0.0    
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

def main():    
    args = PARSER.parse_args()
    
    # 加载种子分子
    seed_smiles = []
    with open(args.seed_file, 'r') as f:
        seed_smiles = [line.strip() for line in f if line.strip()]
    
    print(f"从种子文件读取了 {len(seed_smiles)} 个种子分子")
    
    # 加载源种群（上一代）
    source_smiles = []
    with open(args.input_file, 'r') as f:
        source_smiles = [line.split()[0].strip() for line in f if line.strip()]
    
    print(f"从源文件读取了 {len(source_smiles)} 个分子")
    
    # 加载LLM生成分子
    llm_smiles = []
    with open(args.llm_generation_file, 'r') as f:
        llm_smiles = [line.strip() for line in f if line.strip()]
    
    print(f"从LLM生成文件读取了 {len(llm_smiles)} 个分子")
    
    # 合并GPT生成分子和源种群（用于评估）
    full_population = list(set(source_smiles + llm_smiles))
    
    # 初始化评估器
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    
    # 变异参数配置
    vars = {
        'rxn_library': 'all_rxns',
        'rxn_library_file': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json',
        'function_group_library': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json',
        'complementary_mol_directory':'/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir',
        'filter_object_dict': {
            # 使用 autogrow 的过滤器类
            'Structure_check': LipinskiStrictFilter()
        },
        'max_time_mcs_thorough': 1,
        'gypsum_thoroughness': 3
    }

    # 初始化变异器
    rxn_library_vars = [
        vars['rxn_library'],
        vars['rxn_library_file'],
        vars['function_group_library'],
        vars['complementary_mol_directory']
    ]
    
    # 执行变异，确保生成args.mutation_attempts个新分子
    mutation_results = []
    max_tries = args.mutation_attempts * 10  # 设置最大尝试次数，避免无限循环
    tries = 0
    
    print("*********************开始变异*********************")
    
    with tqdm(total=args.mutation_attempts, desc="Processing mutations") as pbar:
        while len(mutation_results) < args.mutation_attempts and tries < max_tries:
            tries += 1          
            
            parent = random.choice(seed_smiles)            
            
            click_chem = SmileClickClass.SmilesClickChem(rxn_library_vars, [], vars['filter_object_dict'])
            
            
            success = False
            for attempt in range(args.max_mutations):
                result = click_chem.run_smiles_click(parent)
                if not result:
                    continue               
                
                valid_results = []
                for smi in result:
                    try:
                        
                        if all([check(smi) for check in vars['filter_object_dict'].values()]):
                            
                            if smi not in source_smiles and smi not in mutation_results:
                                valid_results.append(smi)
                    except:
                        continue
                
                if valid_results:
                   
                    chosen_smi = valid_results[0]
                    mutation_results.append(chosen_smi)
                    success = True
                    pbar.update(1)
                    break  
            
            if len(mutation_results) == args.mutation_attempts:
                break
    
    print(f"成功生成了 {len(mutation_results)} 个变异产物，总尝试次数: {tries}")  
   
    if len(mutation_results) < args.mutation_attempts:
        needed = args.mutation_attempts - len(mutation_results)
        print(f"变异产生的分子不足，需要从源种群随机选择 {needed} 个分子补充")
        
        # 从源种群中选择未被使用的分子
        available_mols = [mol for mol in source_smiles if mol not in mutation_results]
        
        if available_mols:
            supplement = random.sample(
                available_mols, 
                min(needed, len(available_mols))
            )
            mutation_results.extend(supplement)
    
    # 保存变异结果
    with open(args.output_file, 'w') as f:
        for smi in mutation_results:
            f.write(f"{smi}\n")
    
    print(f"变异完成，结果保存至: {args.output_file}")
    
    # 评估变异结果
    mutation_metrics = evaluate_population(mutation_results, div_eval, nov_eval,
                                          qed_eval, sa_eval, source_smiles)
    print(f"\n变异产生的新分子群评估结果:\n{mutation_metrics}")

if __name__ == "__main__":
    main() 