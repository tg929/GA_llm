import os
import sys
PROJECT_ROOT = "/data1/ytg/GA_llm"
sys.path.insert(0, PROJECT_ROOT)
import logging
import argparse
import autogrow
from tqdm import tqdm
from rdkit import Chem
from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion

# 配置日志
def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "docking.log")),
            logging.StreamHandler()
        ]
    )

# 对接执行器类
class DockingExecutor:
    def __init__(self, receptor_pdb, output_dir, mgltools_path):
        self.receptor_pdb = receptor_pdb
        self.output_dir = os.path.abspath(output_dir)
        self.mgltools_path = mgltools_path
        self._validate_paths()
        
        # 初始化对接参数
        self.docking_params = {
            'center_x': -70.76,   # PARP1结合口袋坐标
            'center_y': 21.82,
            'center_z': 28.33,
            'size_x': 25.0,       # 对接盒尺寸
            'size_y': 20.0,
            'size_z': 25.0,
            'exhaustiveness': 8,
            'num_modes': 9,
            'timeout': 120         # 单次对接超时时间（秒）
        }
        
        # 初始化文件转换器
        self.converter = MGLToolsConversion(
            vars=self._prepare_vars(), 
            receptor_file=receptor_pdb,
            test_boot=False
        )
        
        # 初始化对接器
        self.docker = VinaDocking(self._prepare_vars())

    def _prepare_vars(self):
        """准备Autogrow所需的参数字典"""
        return {
            'filename_of_receptor': self.receptor_pdb,
            'mgl_python': os.path.join(self.mgltools_path, "bin/pythonsh"),
            'prepare_receptor4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
            'prepare_ligand4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"), 
            'docking_executable': "/data1/ytg/GA_llm/autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina",
            'number_of_processors': 1,
            'debug_mode': False,
            'timeout_vs_gtimeout': 'timeout',  # 新增超时参数
            'docking_timeout_limit': 120,
            'environment': {                    # 新增环境变量配置
                'MGLPY': os.path.join(self.mgltools_path, "bin/python"),
                'PYTHONPATH': os.path.join(self.mgltools_path, "MGLToolsPckgs")
            }
        }

    def _validate_paths(self):
        """验证必要路径"""
        required_files = {
            'prepare_receptor4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"),
            'prepare_ligand4.py': os.path.join(self.mgltools_path, "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"),
            'pythonsh': os.path.join(self.mgltools_path, "bin/pythonsh")
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file missing: {name} -> {path}")

    def process_ligand(self, smile):
        """处理单个配体的完整对接流程"""
        try:
            # 生成3D结构
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            # 添加氢原子并生成3D坐标
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            # 使用ETKDG方法生成构象
            params = AllChem.ETKDGv3()
            params.randomSeed = 42  # 确保可重复性
            if AllChem.EmbedMolecule(mol, params) == -1:
                # 尝试基本方法作为后备
                if AllChem.EmbedMolecule(mol) == -1:
                    raise RuntimeError("3D coordinate generation failed")
            # 更严格的力场优化
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            
            # 验证3D坐标存在
            conf = mol.GetConformer()
            if not conf.Is3D():
                raise RuntimeError("Failed to generate valid 3D coordinates")
            
            # 转换为PDB格式
            pdb_path = os.path.join(self.output_dir, f"temp_{hash(smile)}.pdb")
            Chem.MolToPDBFile(mol, pdb_path)
            
            # 验证PDB文件包含3D坐标
            with open(pdb_path) as f:
                if not any(line.startswith('ATOM') for line in f):
                    raise ValueError("Generated PDB lacks 3D coordinates")
            
            # 转换为PDBQT格式
            try:
                self.converter.convert_ligand_pdb_file_to_pdbqt(pdb_path)
            except Exception as e:
                logging.error(f"PDBQT conversion failed: {str(e)}")
                # 保留失败文件供调试
                os.rename(pdb_path, f"{pdb_path}.error")
                raise RuntimeError(f"PDBQT conversion failed: {str(e)}")
                
            pdbqt_path = pdb_path + "qt"
            if not os.path.exists(pdbqt_path):
                raise RuntimeError("Ligand conversion failed - no output file")
        
            # 执行对接
            results = self.docker.dock_molecule(pdbqt_path, self.docking_params)
            if not results:
                return None
                
            # 提取最佳分数
            best_score = min(float(r[0]) for r in results)
            return best_score
            
        except Exception as e:
            logging.error(f"Docking failed for {smile}: {str(e)}")
            return None
        finally:
            # 清理临时文件
            for ext in ['', 'qt']:
                path = f"{pdb_path}{ext}"
                if os.path.exists(path):
                    os.remove(path)

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Molecular Docking Pipeline')
    parser.add_argument('-i', '--input', default="/data1/ytg/GA_llm/output/generation_0_filtered.smi", help='Input SMILES file')#/data1/ytg/GA_llm/output/generation_0_filtered.smi
    parser.add_argument('-r', '--receptor', default="/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb", help='Receptor PDB file path')#/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb
    parser.add_argument('-o', '--output', default="/data1/ytg/GA_llm/output/docking_results/generation_0_docked.smi", help='Output file path')#/data1/ytg/GA_llm/output/docking_results/generation_o_docked.smi
    parser.add_argument('-m', '--mgltools', default="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6", help='MGLTools installation path')
    
    args = parser.parse_args()
    
    # 准备输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    setup_logging(os.path.dirname(args.output))
    
    # 初始化对接执行器
    executor = DockingExecutor(
        receptor_pdb=args.receptor,
        output_dir=os.path.dirname(args.output),
        mgltools_path=args.mgltools
    )
    
    # 读取输入文件
    with open(args.input) as f:
        smiles_list = [line.strip().split()[0] for line in f if line.strip()]
    
    # 并行处理对接
    logging.info(f"Starting docking for {len(smiles_list)} molecules...")
    results = []
    for smile in tqdm(smiles_list, desc="Docking progress"):
        results.append(executor.process_ligand(smile))
    
    # 写入结果文件
    with open(args.output, 'w') as f:
        for smile, score in zip(smiles_list, results):
            if score is not None:
                f.write(f"{smile}\t{score:.2f}\n")
                
    logging.info(f"Docking completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
