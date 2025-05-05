#!/usr/bin/env python
import os
import tempfile
import subprocess
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial

def convert_smile_to_3d(smile):
    """将SMILES转换为3D分子"""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"无法从SMILES创建分子: {smile}")
            return None
            
        # 确保分子是合法的
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        
        # 使用ETKDG算法获得更好的3D构象
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        success = AllChem.EmbedMolecule(mol, params)
        
        # 检查嵌入是否成功
        if success == -1:
            print(f"分子嵌入失败，尝试基础嵌入方法: {smile}")
            AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
        
        # 尝试MMFF优化
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=2000) == -1:
            print("MMFF优化失败,尝试UFF优化")
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            
        return mol
    except Exception as e:
        print(f"转换SMILES到3D结构失败: {e}")
        return None

def parse_vina_output(output_text):
    """解析Vina输出,提取对接分数"""
    try:
        # 首先尝试从log文件中寻找标准输出格式
        lines = output_text.strip().split('\n')
        for line in lines:
            # Vina结果格式: "   1  -10.4      0.000      0.000"
            if line.strip() and not line.startswith('-----'):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        mode_num = int(parts[0])
                        if mode_num == 1:  # 只取第一个构象
                            score = float(parts[1])
                            if score < 0:  # 合理的对接分数应该是负数
                                return score
                    except ValueError:
                        continue
        
        # 如果上面的方法未找到，尝试查找REMARK VINA RESULT格式
        for line in lines:
            if "REMARK VINA RESULT" in line:
                parts = line.strip().split()
                try:
                    score = float(parts[3])
                    if score < 0:  # 合理的对接分数应该是负数
                        return score
                except (ValueError, IndexError):
                    continue
        
        # 如果没有找到符合格式的行，输出日志内容帮助调试
        print("未能解析有效对接分数，输出内容示例:")
        print(output_text[:200] + "..." if len(output_text) > 200 else output_text)
        return 0.0
    except Exception as e:
        print(f"解析Vina输出失败: {e}")
        return 0.0

def dock_single_molecule(smile, mgltools_path, receptor_file, vina_path, temp_dir, center_x=-70.76, center_y=21.82, center_z=28.33, size_x=25.0, size_y=16.0, size_z=25.0, exhaustiveness=8):
    """对单个分子进行对接"""
    try:
        # 创建唯一的分子ID，避免使用含特殊字符的SMILES作为文件名
        import hashlib
        mol_id = hashlib.md5(smile.encode()).hexdigest()[:10]
        
        pdb_file = os.path.join(temp_dir, f"ligand_{mol_id}.pdb")
        pdbqt_file = os.path.join(temp_dir, f"ligand_{mol_id}.pdbqt")
        output_pdbqt = os.path.join(temp_dir, f"output_{mol_id}.pdbqt")
        log_file = os.path.join(temp_dir, f"log_{mol_id}.txt")
        
        # 转换SMILES到PDB
        mol = convert_smile_to_3d(smile)
        if mol is None:
            return smile, None
            
        Chem.MolToPDBFile(mol, pdb_file)
        
        # 准备配体PDBQT
        pythonsh = os.path.join(mgltools_path, 'bin/pythonsh')
        prepare_ligand = os.path.join(mgltools_path, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
        
        cmd = f"{pythonsh} {prepare_ligand} -l {pdb_file} -o {pdbqt_file} -A hydrogens"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if process.returncode != 0 or not os.path.exists(pdbqt_file):
            print(f"配体准备失败: {smile}")
            return smile, None
        
        # 使用共享的受体文件
        receptor_pdbqt = os.path.join(temp_dir, "receptor_base.pdbqt")
        
        # 创建配置文件
        config_file = os.path.join(temp_dir, f"conf_{mol_id}.txt")
        with open(config_file, 'w') as f:
            f.write(f"receptor = {receptor_pdbqt}\n")
            f.write(f"ligand = {pdbqt_file}\n")
            f.write(f"center_x = {center_x}\n")
            f.write(f"center_y = {center_y}\n")
            f.write(f"center_z = {center_z}\n")
            f.write(f"size_x = {size_x}\n")
            f.write(f"size_y = {size_y}\n")
            f.write(f"size_z = {size_z}\n")
            f.write(f"exhaustiveness = {exhaustiveness}\n")
            f.write(f"out = {output_pdbqt}\n")
            f.write(f"log = {log_file}\n")
        
        # 运行Vina
        cmd = f"{vina_path} --config {config_file}"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 检查Vina是否成功运行
        if process.returncode != 0:
            print(f"Vina运行失败，返回码: {process.returncode}")
            print(f"错误信息: {process.stderr}")
            return smile, None
        
        # 检查结果并解析分数
        if os.path.exists(output_pdbqt) and os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            score = parse_vina_output(log_content)
            if score == 0.0:
                # 尝试从输出PDBQT文件中提取得分
                if os.path.exists(output_pdbqt):
                    with open(output_pdbqt, 'r') as f:
                        pdbqt_content = f.read()
                    score = parse_vina_output(pdbqt_content)
                
                if score == 0.0:
                    print(f"\n分子 {smile} 的对接似乎未生成有效分数")
                    with open(log_file, 'r') as f:
                        log_content = f.readlines()
                    # 只输出前几行和后几行帮助调试
                    if log_content:
                        print("日志文件头部:")
                        for line in log_content[:5]:
                            print(line.strip())
                        print("日志文件尾部:")
                        for line in log_content[-5:]:
                            print(line.strip())
                    return smile, None
            return smile, score
        else:
            return smile, None
        
    except Exception as e:
        print(f"对接过程中出错 ({smile}): {e}")
        return smile, None

def batch_dock_molecules(input_file, output_file, receptor_file, mgltools_path, 
                         center_x=-70.76, center_y=21.82, center_z=28.33, 
                         size_x=25.0, size_y=16.0, size_z=25.0,
                         exhaustiveness=8, num_processors=1, cleanup=True):
    """批量对接分子文件"""
    print(f"从文件 {input_file} 读取分子...")
    
    # 读取SMILES
    molecules = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                # 处理可能的注释或额外信息
                parts = line.strip().split()
                if parts:
                    molecules.append(parts[0])
    
    if not molecules:
        print("未找到有效分子")
        return
    
    print(f"找到 {len(molecules)} 个分子，开始对接...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")
    
    # 确保vina有执行权限
    vina_path = "/data1/tgy/GA_llm/autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina"
    os.system(f"chmod +x {vina_path}")
    
    # 准备受体文件 - 只准备一次
    receptor_base = os.path.join(temp_dir, "receptor_base.pdbqt")
    pythonsh = os.path.join(mgltools_path, 'bin/pythonsh')
    prepare_receptor = os.path.join(mgltools_path, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py')
    cmd = f"{pythonsh} {prepare_receptor} -r {receptor_file} -o {receptor_base} -A hydrogens"
    print(f"准备受体: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if process.returncode != 0 or not os.path.exists(receptor_base):
        print(f"受体准备失败: {process.stderr}")
        if cleanup:
            import shutil
            shutil.rmtree(temp_dir)
        return
    
    results = []
    
    # 决定是否使用多进程
    if num_processors > 1:
        print(f"使用 {num_processors} 个进程进行并行对接...")
        
        # 创建部分函数
        dock_func = partial(
            dock_single_molecule,
            mgltools_path=mgltools_path,
            receptor_file=receptor_file,
            vina_path=vina_path,
            temp_dir=temp_dir,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            exhaustiveness=exhaustiveness
        )
        
        # 使用进程池进行并行计算
        with multiprocessing.Pool(processes=num_processors) as pool:
            results = list(tqdm(pool.imap(dock_func, molecules), total=len(molecules)))
    else:
        print("使用单进程对接...")
        # 单进程模式
        for smile in tqdm(molecules):
            result = dock_single_molecule(
                smile, 
                mgltools_path, 
                receptor_file, 
                vina_path, 
                temp_dir,
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                exhaustiveness=exhaustiveness
            )
            results.append(result)
    
    # 过滤有效结果并排序
    valid_results = [(smile, score) for smile, score in results if score is not None]
    if valid_results:
        valid_results.sort(key=lambda x: x[1])  # 按照分数从小到大排序
    
    # 保存结果
    success_count = len(valid_results)
    print(f"对接完成，共 {success_count}/{len(molecules)} 个分子成功对接")
    
    # 写入结果文件
    with open(output_file, 'w') as f:
        for smile, score in valid_results:
            f.write(f"{smile}\t{score:.4f}\n")
    
    # 计算统计信息
    if valid_results:
        scores = [score for _, score in valid_results]
        mean_score = np.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        stats_file = output_file.replace('.smi', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"总分子数: {len(valid_results)}\n")
            f.write(f"平均对接分数: {mean_score:.4f}\n")
            f.write(f"最佳对接分数: {min_score:.4f}\n")
            #f.write(f"最差对接分数: {max_score:.4f}\n")
            
            if len(valid_results) >= 10:
                top10_mean = np.mean([score for _, score in valid_results[:10]])
                f.write(f"Top 10对接分数均值: {top10_mean:.4f}\n")                
            if len(valid_results) >= 20:
                top20_mean = np.mean([score for _, score in valid_results[:20]])
                f.write(f"Top 20对接分数均值: {top20_mean:.4f}\n")
            if len(valid_results) >= 50:
                top50_mean = np.mean([score for _, score in valid_results[:50]])
                f.write(f"Top 50对接分数均值: {top50_mean:.4f}\n")
            if len(valid_results) >= 100:
                top100_mean = np.mean([score for _, score in valid_results[:100]])
                f.write(f"Top 100对接分数均值: {top100_mean:.4f}\n")
    
    # 清理临时文件
    if cleanup:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"清理临时目录: {temp_dir}")
    
    print(f"结果已保存至 {output_file}")
    return valid_results

def full_dock_test():
    """单分子对接测试"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")
    
    # 设置文件路径
    mgltools_path = "/data1/tgy/GA_llm/mgltools_x86_64Linux2_1.5.6"
    pythonsh = os.path.join(mgltools_path, 'bin/pythonsh')
    prepare_ligand = os.path.join(mgltools_path, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py')
    prepare_receptor = os.path.join(mgltools_path, 'MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py')
    vina_path = "/data1/tgy/GA_llm/autogrow/docking/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina"
    receptor_file = "/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb"
    
    # 确保vina有执行权限
    os.system(f"chmod +x {vina_path}")
    
    # 生成测试分子
    test_smile = "c1ccccc1"  # 苯
    pdb_file = os.path.join(temp_dir, "ligand.pdb")
    pdbqt_file = os.path.join(temp_dir, "ligand.pdbqt")
    receptor_pdbqt = os.path.join(temp_dir, "receptor.pdbqt")
    output_pdbqt = os.path.join(temp_dir, "output.pdbqt")
    log_file = os.path.join(temp_dir, "docking.log")  # 添加log文件
    
    # 转换SMILES到PDB
    mol = convert_smile_to_3d(test_smile)
    if mol:
        Chem.MolToPDBFile(mol, pdb_file)
        print(f"成功创建PDB文件: {pdb_file}")
    else:
        print("分子转换失败")
        return
    
    # 准备配体PDBQT
    cmd = f"{pythonsh} {prepare_ligand} -l {pdb_file} -o {pdbqt_file} -A hydrogens"
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"配体准备结果: {result.returncode}")
    if result.stderr:
        print(f"错误: {result.stderr}")
    
    # 准备受体PDBQT
    cmd = f"{pythonsh} {prepare_receptor} -r {receptor_file} -o {receptor_pdbqt} -A hydrogens"
    print(f"执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"受体准备结果: {result.returncode}")
    
    # 检查文件存在
    if not os.path.exists(pdbqt_file):
        print(f"配体PDBQT文件未生成: {pdbqt_file}")
        return
    if not os.path.exists(receptor_pdbqt):
        print(f"受体PDBQT文件未生成: {receptor_pdbqt}")
        return
    
    # 创建配置文件
    config_file = os.path.join(temp_dir, "conf.txt")
    with open(config_file, 'w') as f:
        f.write(f"receptor = {receptor_pdbqt}\n")
        f.write(f"ligand = {pdbqt_file}\n")
        f.write("center_x = -70.76\n")
        f.write("center_y = 21.82\n")
        f.write("center_z = 28.33\n")
        f.write("size_x = 25.0\n")
        f.write("size_y = 16.0\n")
        f.write("size_z = 25.0\n")
        f.write("exhaustiveness = 8\n")
        f.write(f"out = {output_pdbqt}\n")
        f.write(f"log = {log_file}\n")
    
    # 运行Vina
    cmd = f"{vina_path} --config {config_file}"
    print(f"执行对接命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"对接结果: {result.returncode}")
    if result.stdout:
        print(f"输出: {result.stdout}")
    if result.stderr:
        print(f"错误: {result.stderr}")
    
    # 检查结果
    if os.path.exists(output_pdbqt):
        print("对接成功，生成了输出文件")
        
        # 解析log文件获取对接分数
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            score = parse_vina_output(log_content)
            print(f"对接分数: {score}")
    else:
        print("对接失败，没有输出文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分子对接工具')
    parser.add_argument('--input', type=str, help='输入的SMILES分子文件')
    parser.add_argument('--output', type=str, help='输出文件（包含对接分数的排序结果）')
    parser.add_argument('--receptor', type=str, default='/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb', 
                        help='蛋白质受体文件路径')
    parser.add_argument('--center_x', type=float, default=-70.76, help='对接盒子中心X坐标')
    parser.add_argument('--center_y', type=float, default=21.82, help='对接盒子中心Y坐标')
    parser.add_argument('--center_z', type=float, default=28.33, help='对接盒子中心Z坐标')
    parser.add_argument('--size_x', type=float, default=25.0, help='对接盒子X大小')
    parser.add_argument('--size_y', type=float, default=16.0, help='对接盒子Y大小')
    parser.add_argument('--size_z', type=float, default=25.0, help='对接盒子Z大小')
    parser.add_argument('--exhaustiveness', type=int, default=8, help='搜索彻底性参数')
    parser.add_argument('--processors', type=int, default=1, help='使用的处理器数量')
    parser.add_argument('--test', action='store_true', help='运行测试用例')
    
    args = parser.parse_args()
    
    # MGLTools路径
    mgltools_path = "/data1/tgy/GA_llm/mgltools_x86_64Linux2_1.5.6"
    
    if args.test:
        # 运行测试用例
        full_dock_test()
    elif args.input and args.output:
        # 批量对接
        batch_dock_molecules(
            args.input, 
            args.output, 
            args.receptor, 
            mgltools_path,
            center_x=args.center_x,
            center_y=args.center_y,
            center_z=args.center_z,
            size_x=args.size_x,
            size_y=args.size_y,
            size_z=args.size_z,
            exhaustiveness=args.exhaustiveness,
            num_processors=args.processors
        )
    else:
        parser.print_help()
