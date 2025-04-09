# import os
# os.chdir("./datasets/decompose")
import argparse
from tqdm import tqdm  # 用于显示进度条
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Lipinski
from collections import defaultdict

# from molecules.fragmentation import reconstruct
from rdkit import Chem
import numpy as np
from rdkit.Chem import BRICS
from copy import deepcopy

dummy = Chem.MolFromSmiles('[*]')

def mol_from_smiles(smi):
    smi = canonicalize(smi)
    return Chem.MolFromSmiles(smi)

def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol

def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res

def get_size(frag):
    dummies = count_dummies(frag)
    total_atoms = frag.GetNumAtoms()
    real_atoms = total_atoms - dummies
    return real_atoms


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count

def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)


def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]
    #return [Chem.MolToSmiles(m, isomericSmiles=True, allBondsExplicit=True) for m in mols]


def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))

        if bonds == []:
            frags.append(mol)
            return frags

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]

        # 只会断开一根键，也就是说，如果某个片段可以切割两个断点，但是只会切割其中一个，另一个会跟该变短视作一个整体
        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]], 
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        # print(mol_to_smiles(head), mol_to_smiles(tail))
        frags.append(head)
        return fragment_recursive(tail, frags)
    except Exception:
        pass

def join_molecules(molA, molB):
    marked, neigh = None, None
    for atom in molA.GetAtoms():
        if atom.GetAtomicNum() == 0:
            marked = atom.GetIdx()
            neigh = atom.GetNeighbors()[0]
            break
    neigh = 0 if neigh is None else neigh.GetIdx()

    if marked is not None:
        ed = Chem.EditableMol(molA)
        if neigh > marked:
            neigh = neigh - 1
        ed.RemoveAtom(marked)
        molA = ed.GetMol()

    joined = Chem.ReplaceSubstructs(
        molB, dummy, molA,
        replacementConnectionPoint=neigh,
        useChirality=False)[0]

    Chem.Kekulize(joined)
    return joined

def reconstruct(frags, reverse=False):
    if len(frags) == 1:
        return strip_dummy_atoms(frags[0]), frags

    if count_dummies(frags[0]) != 1:
        return None, None

    if count_dummies(frags[-1]) != 1:
        return None, None

    for frag in frags[1:-1]:
        if count_dummies(frag) != 2:
            return None, None
    
    mol = join_molecules(frags[0], frags[1])
    for i, frag in enumerate(frags[2:]):
        #print(i, mol_to_smiles(frag), mol_to_smiles(mol))
        mol = join_molecules(mol, frag)
        #print(i, mol_to_smiles(mol))

    # see if there are kekulization/valence errors
    mol_to_smiles(mol)

    return mol, frags
        
def break_into_fragments(mol, smi):
    frags = []
    frags = fragment_recursive(mol, frags)

    if len(frags) == 0:
        return smi, np.nan, 0,[]

    if len(frags) == 1:
        return smi, smi, 1,frags

    rec, frags = reconstruct(frags)
    if rec and mol_to_smiles(rec) == smi:
        fragments = mols_to_smiles(frags)
        return smi, fragments, len(frags), frags  # 直接返回列表

    return smi, [smi], 0, []  # 返回原始分子作为列表
def batch_process(input_file, output_file,output_file2,output_file3,output_file4):
    """批量处理文件的核心函数"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out,open(output_file2,"w") as f_out2,open(output_file3,"w") as f_out3,open(output_file4, "w") as f_out4:
        for line in tqdm(f_in, desc='Processing molecules'):
            smi = line.strip().split()[0]  # 读取每行第一个SMILES
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    f_out.write(f"{smi}\tInvalid\n")
                    f_out2.write(f"Invalid\n")
                    continue
                
                # 使用现有分解逻辑
                _, fragments_list, num_frag, _ = break_into_fragments(mol, smi)
                f_out.write(f"{smi}\t{str(fragments_list)}\n")            
                if isinstance(fragments_list, list) and num_frag > 1:                       
                    sep_joined = '[SEP]'.join(fragments_list)
                    f_out2.write(f"[BOS]{sep_joined}[EOS]\n")
                    f_out4.write(f"{smi}\n")  # 写入原始SMILES
                    if len(fragments_list) > 1:
                        truncated_list = fragments_list[:-1]
                        truncated_joined = '[SEP]'.join(truncated_list)
                        f_out3.write(f"[BOS]{truncated_joined}[SEP]\n")           

            except Exception as e:
                error_msg = f"{smi}\tError\n"
                f_out.write(error_msg)
                f_out2.write("Error\n")
                print(f"处理失败: {smi} | 错误: {str(e)}")

#入口函数---主函数
if __name__ == "__main__":

    # smiles = "O=C(C[C@H]1C2=NN=C(N2C3=C(C(C4=CC=C(C=C4)C5=CC=C(NC(C)=O)C=C5)=N1)C(C)=C(C)S3)C)OC(C)(C)C"
    # smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))  # 标准化SMILES
    # mol = Chem.MolFromSmiles(smiles)
    # result = break_into_fragments(mol, smiles)
    # print(result[1].split(" "))#分割结果：以列表形式存储了  ['*C(C)=O', '*N*', '*c1ccc(*)cc1', '*C(=O)C[C@@H]1N=C(c2ccc(*)cc2)c2c(sc(C)c2C)-n2c(C)nnc21', '*O*', '*C(C)(C)C']
    # print(result[0])#原始SMILES  CC(=O)Nc1ccc(-c2ccc(C3=N[C@@H](CC(=O)OC(C)(C)C)c4nnc(C)n4-c4sc(C)c(C)c43)cc2)cc1
    # print('[BOS]' + '[SEP]'.join(result[1].split()) + '[EOS]')#[BOS]*C(C)=O[SEP]*N*[SEP]*c1ccc(*)cc1[SEP]*C(=O)C[C@@H]1N=C(c2ccc(*)cc2)c2c(sc(C)c2C)-n2c(C)nnc21[SEP]*O*[SEP]*C(C)(C)C[EOS]

   
    parser = argparse.ArgumentParser(description='分子碎片分解工具')
    parser.add_argument('-i', '--input', help='输入文件路径(每行一个SMILES)')
    parser.add_argument('-o', '--output', help='输出文件路径',default='./decompose_results_0/frags_result_crossover0.smi')
    parser.add_argument('-o2', '--output2', help='BOS/EOS格式输出路径', default='./decompose_results_0/frags_seq_crossover0.smi')
    parser.add_argument('-o3', '--output3', help='新格式输出路径', default='./decompose_results_0/truncated_frags_crossover0.smi')
    parser.add_argument('-o4', '--output4', help='可分解分子记录路径',default='./decompose_results_0/decomposable_mols_crossover0.smi')
    
    args = parser.parse_args()
    batch_process(args.input, args.output,args.output2,args.output3,args.output4)


