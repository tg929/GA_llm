import torch
from utils.train_utils import seed_all
import os
import argparse
import subprocess
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import random
from rdkit import Chem
from utils.train_utils import get_mol
from utils.chem_utils import reconstruct
from tqdm import tqdm

# 当前地址：/data1/tgy/GA_llm/fragment_GPT
# vocab.txt地址：/data1/tgy/GA_llm/fragment_GPT/vocabs/vocab.txt

def decompose_molecules(input_file, output_dir):
    """
    对输入文件中的分子进行分解，生成片段序列文件
    
    Args:
        input_file: 输入的分子SMILES文件
        output_dir: 输出目录
        
    Returns:
        分解结果文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件路径
    output_file = os.path.join(output_dir, "frags_result.smi")
    output_file2 = os.path.join(output_dir, "frags_seq.smi")
    output_file3 = os.path.join(output_dir, "truncated_frags.smi")
    output_file4 = os.path.join(output_dir, "decomposable_mols.smi")
    
    # 调用分解脚本
    decompose_script = "/data1/tgy/GA_llm/datasets/decompose/demo_frags.py"
    cmd = [
        "python", decompose_script,
        "-i", input_file,
        "-o", output_file,
        "-o2", output_file2,
        "-o3", output_file3,
        "-o4", output_file4
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"分子分解失败: {process.stderr}")
        raise Exception("分子分解失败")
    
    print(f"分子分解完成，生成文件: {output_file3}")
    
    # 读取分解结果，去除最后一个片段作为前缀条件
    prefixes = []
    molecules = []
    with open(output_file2, 'r') as f:
        for line in f:
            if line.strip():
                molecules.append(line.strip())
    
    # 创建前缀文件（去除最后一个片段）
    with open(os.path.join(output_dir, "input_prefixes.smi"), 'w') as f:
        for mol in molecules:
            # 分离片段
            fragments = mol.split('[SEP]')
            if len(fragments) > 1:  # 至少有两个片段才能去掉最后一个
                prefix = '[SEP]'.join(fragments[:-1])
                f.write(f"{prefix}\n")
                prefixes.append(prefix)
    
    return os.path.join(output_dir, "input_prefixes.smi"), prefixes

def Test(model, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device,
         output_file_path, prefixes, generations_per_input=3):
    """
    使用GPT模型生成分子
    
    Args:
        model: GPT模型
        tokenizer: 分词器
        max_seq_len: 最大序列长度
        temperature: 温度参数
        top_k: top_k采样参数
        stream: 是否流式生成
        rp: 重复惩罚参数
        kv_cache: 是否使用KV缓存
        is_simulation: 是否为模拟
        device: 设备
        output_file_path: 输出文件路径
        prefixes: 输入前缀条件列表
        generations_per_input: 每个输入条件生成的分子数量
    
    Returns:
        有效分子SMILES列表
    """
    complete_answer_list = []
    valid_answer_list = []
    model.eval()
    
    if not prefixes:
        prefixes = [None]  # 保持无输入时生成功能
    
    # 修改循环结构：每个条件生成多个分子
    for input_prefix in tqdm(prefixes, desc='Processing molecules'):
        for _ in range(generations_per_input):  # 每个条件生成多个分子
            # 生成条件输入的token序列
            if input_prefix:
                prefix_tokens = tokenizer.encode(input_prefix, add_special_tokens=False)
                x = torch.tensor([prefix_tokens], dtype=torch.int64).to(device)
            else:
                x = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int64).to(device)

            with torch.no_grad():
                res_y = model.generate(x, tokenizer, max_new_tokens=max_seq_len,
                                    temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
                                    is_simulation=is_simulation)
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            complete_answer = f"{tokenizer.decode(x[0])}"  # 用于保存整个生成的句子

            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                # 保存生成的片段到完整回答中
                complete_answer += answer[history_idx:]

                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            complete_answer = complete_answer.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "")
            frag_list = complete_answer.replace(" ", "").split('[SEP]')
            try:
                frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
                mol = reconstruct(frag_mol)[0]
                if mol:
                    generate_smiles = Chem.MolToSmiles(mol)
                    valid_answer_list.append(generate_smiles)
                    answer = frag_list
                else:
                    answer = frag_list
            except:
                answer = frag_list
            complete_answer_list.append(answer)

    # 计算有效率
    valid_ratio = len(valid_answer_list) / len(complete_answer_list) if complete_answer_list else 0
    print(f"有效分子比例: {len(valid_answer_list)}/{len(complete_answer_list)} = {valid_ratio:.2f}")
    
    # 确保输出目录存在
    os.makedirs(output_file_path, exist_ok=True)
    
    # 保存完整片段结果
    with open(os.path.join(output_file_path, f'generated_fragments.smi'), "w") as w:
        for j in complete_answer_list:
            if not isinstance(j, str):
                j = str(j)
            w.write(j)
            w.write("\n")
    
    # 保存有效分子SMILES
    with open(os.path.join(output_file_path, f'generated_molecules.smi'), "w") as w:
        for j in valid_answer_list:
            w.write(j)
            w.write("\n")
    
    return valid_answer_list

def main_test(args):
    # 设置随机种子
    seed_value = int(args.seed)
    seed_all(seed_value)
    random.seed(seed_value)
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    # 输出目录
    output_dir = "/data1/tgy/GA_llm/fragment_GPT/test_output/generated_output_all_finetune"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 分解分子
    print(f"步骤1: 分解输入文件中的分子: {args.input_file}")
    prefix_file, prefixes = decompose_molecules(args.input_file, output_dir)
    
    # 步骤2: 初始化GPT模型
    print("步骤2: 初始化GPT模型")
    tokenizer = SmilesTokenizer('/data1/tgy/GA_llm/fragment_GPT/vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    
    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'/data1/tgy/GA_llm/fragment_GPT/weights/fragpt.pt')
    model.load_state_dict(checkpoint)
    
    # 步骤3: 使用GPT生成分子
    print(f"步骤3: 生成分子，每个前缀条件生成 {args.generations_per_input} 个分子")
    start_time = time.time()
    
    # 设置温度参数，增加多样性
    temperature = args.temperature
    
    valid_molecules = Test(
        model, tokenizer, 
        max_seq_len=1024, 
        temperature=temperature, 
        top_k=args.top_k, 
        stream=False, 
        rp=1., 
        kv_cache=True,
        is_simulation=True, 
        device=device, 
        output_file_path=output_dir, 
        prefixes=prefixes,
        generations_per_input=args.generations_per_input
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"生成完成，生成了 {len(valid_molecules)} 个有效分子")
    print(f"运行时间: {elapsed_time:.4f} 秒")
    
    return os.path.join(output_dir, "generated_molecules.smi")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分子生成脚本 - 用于生成多个分子')
    parser.add_argument('--input_file', required=True, help='输入的分子SMILES文件路径')
    parser.add_argument('--device', default='0', help='设备ID,例如 0 或 0,1 或 cpu')
    parser.add_argument('--seed', default='42', help='随机种子')
    parser.add_argument('--generations_per_input', type=int, default=3, help='每个输入条件生成的分子数量')
    parser.add_argument('--temperature', type=float, default=1.2, help='温度参数 (1.0=标准, >1.0=更高随机性)')
    parser.add_argument('--top_k', type=int, default=40, help='top_k采样参数,控制多样性')
    
    opt = parser.parse_args()
    
    main_test(opt)
