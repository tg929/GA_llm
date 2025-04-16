import sys
import os
import argparse
import numpy as np
from rdkit import Chem
from tdc import Oracle
PROJECT_ROOT = "/data1/tgy/GA_llm"
sys.path.insert(0, PROJECT_ROOT)

# 导入Autogrow过滤器
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_strict_filter import LipinskiStrictFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.lipinski_lenient_filter import LipinskiLenientFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_filter import GhoseFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.ghose_modified_filter import GhoseModifiedFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.vande_waterbeemd_filter import VandeWaterbeemdFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.mozziconacci_filter import MozziconacciFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.pains_filter import PAINSFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.nih_filter import NIHFilter
from autogrow.operators.filter.filter_classes.filter_children_classes.brenk_filter import BRENKFilter


class StructureCheckFilter:
    def run_filter(self, mol):
        return mol is not None

def init_filters(args):
    """初始化过滤器集合"""
    filters = {'Structure': StructureCheckFilter()}
    
    # 根据命令行参数添加过滤器
    if hasattr(args, 'LipinskiStrictFilter') and args.LipinskiStrictFilter:
        filters['LipinskiStrict'] = LipinskiStrictFilter()
    
    if hasattr(args, 'LipinskiLenientFilter') and args.LipinskiLenientFilter:
        filters['LipinskiLenient'] = LipinskiLenientFilter()
    
    if hasattr(args, 'GhoseFilter') and args.GhoseFilter:
        filters['Ghose'] = GhoseFilter()
    
    if hasattr(args, 'GhoseModifiedFilter') and args.GhoseModifiedFilter:
        filters['GhoseModified'] = GhoseModifiedFilter()
    
    if hasattr(args, 'MozziconacciFilter') and args.MozziconacciFilter:
        filters['Mozziconacci'] = MozziconacciFilter()
    
    if hasattr(args, 'VandeWaterbeemdFilter') and args.VandeWaterbeemdFilter:
        filters['VandeWaterbeemd'] = VandeWaterbeemdFilter()
    
    if hasattr(args, 'PAINSFilter') and args.PAINSFilter:
        filters['PAINS'] = PAINSFilter()
    
    if hasattr(args, 'NIHFilter') and args.NIHFilter:
        filters['NIH'] = NIHFilter()
    
    if hasattr(args, 'BRENKFilter') and args.BRENKFilter:
        filters['BRENK'] = BRENKFilter()
    
    # 处理自定义过滤器
    if hasattr(args, 'alternative_filter') and args.alternative_filter:
        # 预期格式: [[name1, path1], [name2, path2], ...]
        try:
            for filter_info in args.alternative_filter:
                if isinstance(filter_info, list) and len(filter_info) == 2:
                    name, path = filter_info
                    # 动态导入自定义过滤器
                    sys.path.append(os.path.dirname(path))
                    module_name = os.path.basename(path).replace('.py', '')
                    filter_module = __import__(module_name)
                    filter_class = getattr(filter_module, name)
                    filters[name] = filter_class()
        except Exception as e:
            print(f"加载自定义过滤器失败: {e}")
    
    # 如果No_Filters设置为True，或者没有指定其他过滤器，就只使用结构检查过滤器
    if (hasattr(args, 'No_Filters') and args.No_Filters) or len(filters) <= 1:
        print("不应用任何药物化学过滤器，仅进行基本结构检查")
        return {'Structure': StructureCheckFilter()}
    
    return filters

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
    parser.add_argument("-o", "--output", default="/data1/tgy/GA_llm/output/generation_0_filtered.smi", help="输出文件路径")
    
    # 添加过滤器参数
    parser.add_argument('--LipinskiStrictFilter', action='store_true', default=False,
                        help='严格版Lipinski五规则过滤器,筛选口服可用药物')
    parser.add_argument('--LipinskiLenientFilter', action='store_true', default=False,
                        help='宽松版Lipinski五规则过滤,筛选口服可用药物')
    parser.add_argument('--GhoseFilter', action='store_true', default=False,
                        help='Ghose药物相似性过滤器')
    parser.add_argument('--GhoseModifiedFilter', action='store_true', default=False,
                        help='修改版Ghose过滤器,将分子量上限从480Da放宽到500Da')
    parser.add_argument('--MozziconacciFilter', action='store_true', default=False,
                        help='Mozziconacci药物相似性过滤器')
    parser.add_argument('--VandeWaterbeemdFilter', action='store_true', default=False,
                        help='筛选可能透过血脑屏障的药物')
    parser.add_argument('--PAINSFilter', action='store_true', default=False,
                        help='PAINS过滤器,过滤泛测试干扰化合物')
    parser.add_argument('--NIHFilter', action='store_true', default=False,
                        help='NIH过滤器,过滤含不良功能团的分子')
    parser.add_argument('--BRENKFilter', action='store_true', default=False,
                        help='BRENK前导物相似性过滤器')
    parser.add_argument('--No_Filters', action='store_true', default=False,
                        help='不应用任何过滤器')
    parser.add_argument('--alternative_filter', action='append',
                        help='添加自定义过滤器')
    
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

    # 初始化过滤器
    filters = init_filters(args)
    print(f"\n使用的过滤器: {', '.join(filters.keys())}")
    
    # 执行过滤
    filtered = []
    filter_counters = {name: 0 for name in filters.keys()}
    
    for smi in population:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # 逐一应用过滤器并统计通过情况
        passed_all = True
        for name, filter_obj in filters.items():
            if filter_obj.run_filter(mol):
                filter_counters[name] += 1
            else:
                passed_all = False
                break
                
        if passed_all:
            filtered.append(smi)

    # 报告每个过滤器的通过率
    print("\n各过滤器通过率:")
    total_valid = sum(1 for smi in population if Chem.MolFromSmiles(smi) is not None)
    for name, counter in filter_counters.items():
        if total_valid > 0:
            print(f"{name}: {counter}/{total_valid} ({counter/total_valid*100:.1f}%)")
    
    # 过滤后评估
    print("\n过滤后评估:")
    filtered_stats = evaluate_population(filtered, qed_evaluator, sa_evaluator)
    print(f"保留分子数: {len(filtered)}/{len(population)} ({len(filtered)/len(population)*100:.1f}%)")
    print(f"保留分子QED: {filtered_stats['avg_qed']:.3f}")
    print(f"保留分子SA: {filtered_stats['avg_sa']:.3f}")

    # 保存结果
    with open(args.output, 'w') as f:
        f.write("\n".join(filtered))
    
    print(f"\n过滤后的分子已保存到: {args.output}")
    
    return filtered

if __name__ == "__main__":
    main()