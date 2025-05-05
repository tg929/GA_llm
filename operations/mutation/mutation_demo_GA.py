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
PARSER = argparse.ArgumentParser(description='GA mutation parameters for pure GA implementation')
PARSER.add_argument("--input_file", "-i", type=str, required=True, help="输入分子文件路径")
PARSER.add_argument("--output_file", "-o", type=str, default="/data1/tgy/GA_llm/output/generation_0_mutationed.smi", help="输出文件路径")     
PARSER.add_argument("--mutation_attempts", type=int, default=1, help="变异尝试次数")
PARSER.add_argument("--max_mutations", type=int, default=2, help="每个父代最大变异尝试次数")

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
        
# 评估函数
def evaluate_population(smiles_list, div_eval, nov_eval, qed_eval, sa_eval, ref_smiles):
    # 添加空列表保护
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
    
    # 加载输入数据集
    input_smiles = []
    with open(args.input_file, 'r') as f:
        input_smiles = [line.strip().split()[0] for line in f if line.strip()]
        print(f"输入分子数量: {len(input_smiles)}")
     
    # 初始种群
    initial_population = list(input_smiles)
    
    # 初始化评估器
    div_eval, nov_eval, qed_eval, sa_eval = init_evaluators()
    from tdc import Oracle
    qed_evaluator = Oracle(name='qed')
    sa_evaluator = Oracle(name='sa')
    
    # 评估初始种群
    print('''*********************初始评估*********************''')
    initial_metrics = evaluate_population(initial_population, div_eval, nov_eval, 
                                         qed_eval, sa_eval, initial_population)
    print(f"初始种群评估结果:\n{initial_metrics}")
    
    # 变异参数配置
    vars = {
        'rxn_library': 'all_rxns',
        'rxn_library_file': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json',
        'function_group_library': '/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json',
        'complementary_mol_directory':'/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir',
        'filter_object_dict': {
            # 不使用过滤器，将在结果处理时手动过滤
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
    mutation_results = []
    
    # 执行变异
    with tqdm(total=args.mutation_attempts, desc="Processing mutations") as pbar:
        attempts_count = 0
        
        while len(mutation_results) < args.mutation_attempts and attempts_count < len(initial_population) * 2:
            # 随机选择一个父代分子
            parent = random.choice(initial_population)
            attempts_count += 1
            
            new_mutations = []
            click_chem = SmileClickClass.SmilesClickChem(rxn_library_vars, new_mutations, vars['filter_object_dict'])
            
            # 尝试变异
            success = False
            for attempt in range(args.max_mutations):
                try:
                    # 直接使用原始API，不使用内部过滤器
                    result = click_chem.run_smiles_click_for_mutation_no_filter(parent)
                    
                    if not result:
                        continue
                    
                    # 在结果上手动执行过滤
                    valid_results = []
                    for smi in result:
                        try:
                            # 避免None造成错误
                            mol = Chem.MolFromSmiles(smi)
                            if mol is None:
                                continue
                            
                            # 只要分子有效即添加
                            valid_results.append(smi)
                        except:
                            continue
                    
                    if valid_results:
                        # 只取第一个有效结果
                        chosen_smi = valid_results[0]
                        # 严格去重检查
                        if chosen_smi not in initial_population and chosen_smi not in mutation_results:
                            mutation_results.append(chosen_smi)
                            success = True
                            pbar.update(1)
                            break  # 成功即停止尝试
                except Exception as e:
                    print(f"变异失败: {str(e)}")
                    continue
            
            if attempts_count % 10 == 0:
                pbar.set_postfix({'success_rate': f"{len(mutation_results)}/{attempts_count}"})

    # 合并种群
    new_population = initial_population + mutation_results
    
    # 保存变异产生的新分子群
    temp_mutation_file = "/data1/tgy/GA_llm/output/generation_0_mutation_new.smi"
    os.makedirs(os.path.dirname(temp_mutation_file), exist_ok=True)
    with open(temp_mutation_file, 'w') as f:
        for smi in mutation_results:
            f.write(f"{smi}\n")
    
    # 评估新种群
    print('''*********************评估新种群*********************''')

    final_metrics = evaluate_population(new_population, div_eval, nov_eval,
                                      qed_eval, sa_eval, initial_population)
    print(f"\n变异后整个种群评估结果:\n{final_metrics}")
    
    # 评估变异之后产生的新分子群性质
    mutation_metrics = evaluate_population(mutation_results, div_eval, nov_eval,
                                          qed_eval, sa_eval, initial_population)
    print(f"\n变异产生的新种群评估结果:\n{mutation_metrics}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for smi in new_population:
            f.write(f"{smi}\n")

if __name__ == "__main__":
    main()
