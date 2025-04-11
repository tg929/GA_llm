import sys
import os
import argparse
import numpy as np
from rdkit import Chem
from tdc import Oracle
PROJECT_ROOT = "/data1/ytg/GA_llm"
sys.path.insert(0, PROJECT_ROOT)

# 导入Autogrow过滤器
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter


class StructureCheckFilter:
    def run_filter(self, mol):
        return mol is not None

def init_filters():
    """初始化过滤器集合"""
    return {
        'Lipinski': LipinskiStrictFilter(),
        'Ghose': GhoseFilter(),
        'VandeWaterbeemd': VandeWaterbeemdFilter(),
        'Structure': StructureCheckFilter()
    }

def evaluate_population(smiles_list, qed_eval, sa_eval):
    """评估种群质量"""
    valid_smiles = []
    qed_scores = []
    sa_scores = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid_smiles.append(smi)
        qed_scores.append(qed_eval(smi))
        sa_scores.append(sa_eval(smi))
    
    return {
        'num_valid': len(valid_smiles),
        'avg_qed': np.mean(qed_scores) if qed_scores else 0,
        'avg_sa': np.mean(sa_scores) if sa_scores else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Population Filter Parameters')
    parser.add_argument("-i", "--input", required=True, help="输入文件路径")
    parser.add_argument("-o", "--output", default="/data1/ytg/GA_llm/output/generation_0_filtered.smi", help="输出文件路径")
    args = parser.parse_args()

    # 初始化评估器
    qed_evaluator = Oracle(name='qed')
    sa_evaluator = Oracle(name='sa')
    
    # 加载种群
    with open(args.input, 'r') as f:
        population = [line.strip() for line in f if line.strip()]
    
    # 过滤前评估
    print("过滤前评估:")
    initial_stats = evaluate_population(population, qed_evaluator, sa_evaluator)
    print(f"总分子数: {len(population)}")
    print(f"有效分子: {initial_stats['num_valid']}")
    print(f"平均QED: {initial_stats['avg_qed']:.3f}")
    print(f"平均SA: {initial_stats['avg_sa']:.3f}")

    # 执行过滤
    filters = init_filters()
    filtered = []
    
    for smi in population:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
            
        # 应用所有过滤器
        if all(check.run_filter(mol) for check in filters.values()):
            filtered.append(smi)

    # 过滤后评估
    print("\n过滤后评估:")
    filtered_stats = evaluate_population(filtered, qed_evaluator, sa_evaluator)
    print(f"保留分子数: {len(filtered)}")
    print(f"保留分子QED: {filtered_stats['avg_qed']:.3f}")
    print(f"保留分子SA: {filtered_stats['avg_sa']:.3f}")

    # 保存结果
    with open(args.output, 'w') as f:
        f.write("\n".join(filtered))

if __name__ == "__main__":
    main()