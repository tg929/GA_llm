import os
import sys
PROJECT_ROOT = "/data1/ytg/GA_llm"
sys.path.insert(0, PROJECT_ROOT)
import logging
import argparse
import autogrow
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
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
        
        # 准备VINA需要的变量
        self.vars = self._prepare_vars()
        
        # 初始化文件转换器
        self.converter = MGLToolsConversion(
            vars=self.vars, 
            receptor_file=receptor_pdb,
            test_boot=False
        )
        
        # 初始化对接器
        self.docker = VinaDocking(
            vars=self.vars,
            receptor_file=receptor_pdb,
            file_conversion_class_object=self.converter,
            test_boot=False
        )

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
            'timeout_vs_gtimeout': 'timeout',  
            'docking_timeout_limit': 120,
            'center_x': -70.76,   # PARP1结合口袋坐标
            'center_y': 21.82,
            'center_z': 28.33,
            'size_x': 25.0,       # 对接盒尺寸
            'size_y': 20.0,
            'size_z': 25.0,
            'docking_exhaustiveness': 8,  # 添加这个参数
            'docking_num_modes': 9,       # 添加这个参数
            'environment': {                   
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

    def generate_3d_conformer(self, mol, max_attempts=5):
        """使用多种方法生成3D构象，提高成功率"""
        if mol is None:
            return None
            
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 方法1: ETKDG v3 (更现代的方法)
        for attempt in range(max_attempts):
            seed = 42 + attempt  # 每次尝试使用不同的随机种子
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.numThreads = 4  # 利用多线程
            params.useSmallRingTorsions = True
            params.useBasicKnowledge = True
            params.enforceChirality = True
            
            if AllChem.EmbedMolecule(mol, params) == 0:  # 0表示成功
                # 力场优化
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)  # 使用MMFF力场
                    return mol
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)  # 备选UFF力场
                        return mol
                    except:
                        continue  # 继续尝试下一种方法
        
        # 方法2: 基础ETKDG
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) == 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                return mol
            except:
                pass
                
        # 方法3: 距离几何法
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, useBasicKnowledge=True) == 0:
            return mol
            
        # 所有方法都失败
        return None

    def check_valid_3d_coords(self, pdb_path):
        """验证PDB文件包含有效的3D坐标"""
        atom_count = 0
        nonzero_coords = False
        
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
                    # 获取坐标 (x, y, z位于第7-9列)
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        # 检查坐标是否全为0或接近0
                        if abs(x) > 0.01 or abs(y) > 0.01 or abs(z) > 0.01:
                            nonzero_coords = True
                    except:
                        continue
        
        # 有足够多的原子且至少有一组非零坐标
        return atom_count > 3 and nonzero_coords
        
    def parse_vina_output(self, output_file):
        """解析Vina输出文件获取对接分数"""
        try:
            if not os.path.exists(output_file):
                return None
                
            results = []
            with open(output_file, 'r') as f:
                for line in f:
                    if line.startswith('   ') and not line.startswith('      '):
                        # Vina输出格式为: "   1     -8.7      0.000      0.000"
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                mode = parts[0]
                                score = float(parts[1])
                                # 每行存储为 [分数, 模式]
                                results.append([score, mode])
                            except:
                                continue
            
            return results if results else None
        except Exception as e:
            logging.error(f"解析Vina输出失败: {str(e)}")
            return None

    def process_ligand(self, smile):
        """处理单个配体的完整对接流程"""
        try:
            # 生成分子对象
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                logging.warning(f"无法从SMILES生成分子: {smile}")
                return None
            
            # 处理特殊情况：分子过大或过于复杂
            if mol.GetNumAtoms() > 100:
                logging.warning(f"分子过大，跳过: {smile} (原子数: {mol.GetNumAtoms()})")
                return None
                
            # 改进的3D构象生成
            mol_3d = self.generate_3d_conformer(mol)
            if mol_3d is None:
                logging.warning(f"无法生成3D构象: {smile}")
                return None
                
            # 转换为PDB格式
            pdb_path = os.path.join(self.output_dir, f"temp_{hash(smile)}.pdb")
            Chem.MolToPDBFile(mol_3d, pdb_path)
            
            # 验证PDB文件包含有效的3D坐标
            if not self.check_valid_3d_coords(pdb_path):
                logging.error(f"生成的PDB缺少有效的3D坐标: {smile}")
                
                # 尝试保存分子为MOL2格式，有时这种格式转换更可靠
                try:
                    mol2_path = os.path.join(self.output_dir, f"temp_{hash(smile)}.mol2")
                    
                    # 创建一个PDB文件作为中间体
                    writer = Chem.SDWriter(f"{pdb_path}.sdf")
                    writer.write(mol_3d)
                    writer.close()
                    
                    # 使用Open Babel从SDF转换为MOL2 (如果已安装)
                    import subprocess
                    try:
                        cmd = f"obabel {pdb_path}.sdf -O {mol2_path}"
                        subprocess.run(cmd, shell=True, check=True)
                        
                        # 使用默认的MGLTools方法从mol2转换
                        self.converter.convert_ligand_to_pdbqt_file(mol2_path)
                        pdbqt_path = mol2_path + "qt"
                        if os.path.exists(pdbqt_path):
                            # 运行对接
                            self.docker.run_dock(pdbqt_path)
                            
                            # 解析对接结果
                            docking_output = pdbqt_path + ".vina_docking_output.txt"
                            results = self.parse_vina_output(docking_output)
                            
                            if results:
                                # 获取最佳得分
                                best_score = min(float(r[0]) for r in results)
                                
                                # 清理额外文件
                                for temp_path in [f"{pdb_path}.sdf", mol2_path, pdbqt_path, pdbqt_path + ".vina"]:
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                        
                                return best_score
                    except Exception as e:
                        logging.error(f"Open Babel转换失败: {smile}, 错误: {str(e)}")
                except Exception as e:
                    logging.error(f"备选转换路径失败: {smile}, 错误: {str(e)}")
                
                return None
            
            # 转换为PDBQT格式
            try:
                self.converter.convert_ligand_pdb_file_to_pdbqt(pdb_path)
            except Exception as e:
                logging.error(f"PDBQT转换失败: {str(e)}")
                os.rename(pdb_path, f"{pdb_path}.error")
                return None
                
            pdbqt_path = pdb_path + "qt"
            if not os.path.exists(pdbqt_path):
                logging.error(f"配体转换失败 - 没有输出文件: {smile}")
                return None
        
            # 执行对接 - 使用正确的方法
            failed_smile = self.docker.run_dock(pdbqt_path)
            
            # 检查对接是否成功
            if failed_smile is not None:
                logging.warning(f"对接失败: {smile}")
                return None
                
            # 解析对接结果
            vina_output = pdbqt_path + ".vina"
            if not os.path.exists(vina_output):
                logging.warning(f"对接输出文件不存在: {vina_output}")
                return None
                
            # 从Vina输出文件中读取结果
            results = []
            with open(vina_output, 'r') as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        # Vina结果格式: "REMARK VINA RESULT:    -10.1     0.000     0.000"
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                score = float(parts[3])
                                results.append([score, ""])
                            except:
                                continue
            
            # 如果没有结果，尝试解析日志文件
            if not results:
                docking_log = pdbqt_path + "_docking_output.txt"
                results = self.parse_vina_output(docking_log)
                
            if not results:
                logging.warning(f"无法从对接输出获取分数: {smile}")
                return None
                
            # 提取最佳分数
            best_score = min(float(r[0]) for r in results)
            return best_score
            
        except Exception as e:
            logging.error(f"对接失败，分子: {smile}，错误: {str(e)}")
            return None
        finally:
            # 清理临时文件
            for ext in ['', 'qt', '.error', '.sdf', 'qt.vina', 'qt_docking_output.txt']:
                path = f"{pdb_path}{ext}"
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Molecular Docking Pipeline')
    parser.add_argument('-i', '--input', default="/data1/ytg/GA_llm/output/generation_0_filtered.smi", help='Input SMILES file')#/data1/ytg/GA_llm/output/generation_0_filtered.smi
    parser.add_argument('-r', '--receptor', default="/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb", help='Receptor PDB file path')#/data1/ytg/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb
    parser.add_argument('-o', '--output', default="/data1/ytg/GA_llm/output/docking_results/generation_0_docked.smi", help='Output file path')#/data1/ytg/GA_llm/output/docking_results/generation_o_docked.smi
    parser.add_argument('-m', '--mgltools', default="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6", help='MGLTools installation path')
    parser.add_argument('--max_failures', type=int, default=5, help='最大连续失败次数，超过此数将暂停并提示')
    
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
    logging.info(f"开始对接 {len(smiles_list)} 个分子...")
    results = []
    consecutive_failures = 0
    
    for i, smile in enumerate(tqdm(smiles_list, desc="对接进度")):
        result = executor.process_ligand(smile)
        results.append(result)
        
        # 检查是否连续失败
        if result is None:
            consecutive_failures += 1
            if consecutive_failures >= args.max_failures:
                logging.warning(f"连续失败 {consecutive_failures} 次，请检查对接配置")
                consecutive_failures = 0  # 重置计数器
        else:
            consecutive_failures = 0
            
        # 每处理50个分子保存一次中间结果
        if (i + 1) % 50 == 0:
            with open(f"{args.output}.partial", 'w') as f:
                for s, r in zip(smiles_list[:i+1], results):
                    if r is not None:
                        f.write(f"{s}\t{r:.2f}\n")
            logging.info(f"已完成 {i+1}/{len(smiles_list)} 分子对接，中间结果已保存")
    
    # 写入结果文件
    success_count = 0
    with open(args.output, 'w') as f:
        for smile, score in zip(smiles_list, results):
            if score is not None:
                success_count += 1
                f.write(f"{smile}\t{score:.2f}\n")
                
    logging.info(f"对接完成。成功率: {success_count}/{len(smiles_list)} ({success_count/len(smiles_list)*100:.1f}%)。结果保存至 {args.output}")

if __name__ == "__main__":
    main()
