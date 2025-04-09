from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
import random
import os
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data1/ytg/GA_llm/output/crossover.log'),
        logging.StreamHandler()
    ]
)

def validate_molecule(mol):
    """分子验证加强版"""
    if not mol or mol.GetNumAtoms() < 5:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

class CrossoverEngine:
    def __init__(self, min_mcs_atoms=3, max_retries=100, timeout=60):
        self.min_mcs_atoms = min_mcs_atoms
        self.max_retries = max_retries
        self.timeout = timeout

    def _load_initial_population(self):
        """初始种群加载优化"""
        # 加载source_0.smi前100个有效分子
        source0 = []
        with open('/data1/ytg/GA_llm/datasets/source_0.smi') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 新增空行检查
                    logging.warning(f"检测到空行: 第{line_num}行")
                    continue
                
                parts = line.split()
                if not parts:  # 新增分割结果检查
                    logging.warning(f"无效行格式: 第{line_num}行 -> {line}")
                    continue
                
                smi = parts[0]
                mol = Chem.MolFromSmiles(smi)
                if mol and validate_molecule(mol):
                    source0.append(smi)
                    if len(source0) >= 100:
                        break
        
        # 从LLM文件随机选择20个有效分子
        llm_mols = []
        with open('/data1/ytg/GA_llm/datasets/source_0LLM.smi') as f:
            for line in f:
                smi = line.strip().split()[0]
                mol = Chem.MolFromSmiles(smi)
                if mol and validate_molecule(mol) and '*' not in smi:
                    llm_mols.append(smi)
        llm_mols = random.sample(llm_mols, min(20, len(llm_mols)))

        # 合并种群并生成ID
        initial_population = []
        combined = source0 + llm_mols
        for idx, smi in enumerate(combined):
            initial_population.append((smi, f"mol{idx+1:03d}"))
        return initial_population[:120]  # 确保总数不超过120

    def _mcs_based_crossover(self, mol1, mol2):
        """MCS交叉逻辑改进"""
        try:
            mcs = rdFMCS.FindMCS(
                [mol1, mol2],
                timeout=self.timeout,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder
            )
            if mcs.numAtoms < self.min_mcs_atoms:
                return None
                
            combo = Chem.CombineMols(mol1, mol2)
            edcombo = Chem.EditableMol(combo)
            
            # 在非MCS区域建立连接
            non_mcs_atoms = [
                a.GetIdx() for a in combo.GetAtoms()
                if not mcs.queryMol.HasSubstructMatch(Chem.MolFromSmarts(f"[#{a.GetAtomicNum()}]"))
            ]
            if len(non_mcs_atoms) >= 2:
                a1, a2 = random.sample(non_mcs_atoms, 2)
                edcombo.AddBond(a1, a2, random.choice([
                    Chem.BondType.SINGLE, 
                    Chem.BondType.DOUBLE
                ]))
            
            new_mol = edcombo.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol) if validate_molecule(new_mol) else None
        except:
            return None

    def crossover(self, smi1, smi2):
        """交叉操作增强"""
        if not smi1 or not smi2:
            return None
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if not all([mol1, mol2]):
            return None
        
        for _ in range(self.max_retries):
            result = self._mcs_based_crossover(mol1, mol2)
            if result:
                return result
        return None

def run_crossover():
    """执行完整流程"""
    engine = CrossoverEngine()
    population = engine._load_initial_population()
    
    os.makedirs("/data1/ytg/GA_llm/output", exist_ok=True)
    
    # 生成父代对（带重试机制）
    parent_pairs = []
    for _ in range(200):  # 生成足够数量的候选对
        p1, p2 = random.sample(population, 2)
        if p1[0] != p2[0]:
            parent_pairs.append((p1, p2))
    
    results = []
    for pair in tqdm(parent_pairs[:150], desc="交叉进度"):  # 尝试150次
        p1, p2 = pair
        for attempt in range(3):  # 每个对尝试3次
            child = engine.crossover(p1[0], p2[0])
            if child:
                child_id = f"({p1[1]}+{p2[1]})_Cross_{random.randint(1000,9999)}"
                results.append(f"{child}\t{child_id}")
                break
    
    # 严格结果验证
    valid_results = []
    seen = set()
    for line in results:
        smi = line.split("\t")[0]
        if smi in seen:
            continue
        mol = Chem.MolFromSmiles(smi)
        if validate_molecule(mol):
            valid_results.append(line)
            seen.add(smi)
    
    with open("/data1/ytg/GA_llm/output/cross_results_testLLM.smi", "w") as f:
        f.write("\n".join(valid_results))
    
    # # 生成结构图
    # if valid_results:
    #     mols = [Chem.MolFromSmiles(line.split("\t")[0]) for line in valid_results]
    #     img = Draw.MolsToGridImage(
    #         mols, 
    #         molsPerRow=5, 
    #         subImgSize=(300,300),
    #         maxMols=100
    #     )
    #     img.save("/data1/ytg/GA_llm/output/crossover_structures_LLM.png")

if __name__ == "__main__":
    run_crossover()