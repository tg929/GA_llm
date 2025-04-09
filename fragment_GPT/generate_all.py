import torch
from utils.train_utils import seed_all
import os
import argparse
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import datasets
from rdkit import Chem
from utils.train_utils import get_mol
from utils.chem_utils import reconstruct
from tqdm import tqdm

#当前地址：/data1/ytg/GA_llm/fragment_GPT
#vocab.txt地址：/data1/ytg/GA_llm/fragment_GPT/vocabs/vocab.txt


def Test(model, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device,
         output_file_path, seed,input_file=None):
    complete_answer_list = []
    valid_answer_list = []
    model.eval()
    if input_file:
        with open(input_file, 'r') as f:
            prefixes = [line.strip() for line in f if line.strip()]
    else:
        prefixes = [None]  # 保持无输入时生成功能

    # 修改循环结构：每个条件生成1个分子（不加第二个内部循环）
    for input_prefix in tqdm(prefixes, desc='Processing molecules'):
        #修改：每个输入条件下生成对应的两个新分子：
        #for _ in range(3): 
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

            # print(answer[history_idx:], end='', flush=True)
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

    print(
        f"valid ratio:{len(valid_answer_list)}/{len(complete_answer_list)}={len(valid_answer_list) / len(complete_answer_list)}")
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)
    with open(os.path.join(output_file_path, f'nap_fragsCom_new_{seed}.smi'), "w") as w:
        for j in complete_answer_list:
            if not isinstance(j, str):
                j = str(j)
            w.write(j)
            w.write("\n")
    w.close()
    with open(os.path.join(output_file_path, f'nap_frags_new_{seed}.smi'), "w") as w:
        for j in valid_answer_list:
            w.write(j)
            w.write("\n")
    w.close()


def main_test(args):
    # 设置随机种子的值
    seed_value = int(args.seed)
    seed_all(seed_value)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    batch_size = 1
    device = torch.device(f'cuda:{0}')

    test_names = "test"

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    collator = SmileCollator(tokenizer)

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'./weights/fragpt.pt', weights_only=True)   
    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, tokenizer, max_seq_len=1024, temperature=1.0, top_k=None, stream=False, rp=1., kv_cache=True,
         is_simulation=True, device=device, output_file_path="./output/test0", seed=seed_value, input_file=args.input_file)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"运行时间: {elapsed_time:.4f} 秒")


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--device', default='1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default='0', help='seed')
    parser.add_argument('--input_file', help='批量条件文件路径') 
    #parser.add_argument('--input_prefix', default=None, help='初始条件片段，如 "*CC(=[N+]=[N+]=N)[N+](=O)[O-]"')

    opt = parser.parse_args()

    main_test(opt)