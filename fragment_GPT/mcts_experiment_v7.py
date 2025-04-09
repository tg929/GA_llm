import torch
from utils.train_utils import seed_all
import os
import argparse
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
from mcts_v7_best_leaf import MCTSConfig, MolecularProblemState, MCTS, print_best
from utils.chem_utils import sentence2mol
from rdkit import rdBase
from utils.docking.docking_utils import DockingVina
import pandas as pd
from tqdm import tqdm

# 禁用所有日志信息
rdBase.DisableLog('rdApp.warning')


def Test(model, tokenizer, device, output_file_path):
    model.eval()
    predictor = DockingVina('5ht1b')
    results = []
    # 找到第一个分隔符
    # indices = torch.nonzero(x.squeeze(0) == 13, as_tuple=True)[0]
    # first_index = indices[0].item()
    # x = x[:, :first_index + 1]   # 取第一个片段作为输入
    x = torch.tensor([1], dtype=torch.int64).unsqueeze(0)
    x = x.to(device)
    for i in range(100):
        print('sample:', i+1)
        initial_state = MolecularProblemState(model, tokenizer, predictor, x)
        mcts_config = MCTSConfig()
        mcts = MCTS(initial_state, mcts_config)
        with torch.no_grad():
            rv, rq, rs, smi, cur_sentence = mcts.run()
            results.append([rv, rq, rs, smi, cur_sentence])
        if i % 5 == 0:
            # 将结果转换为 DataFrame 并保存为 CSV 文件
            df = pd.DataFrame(results, columns=['rv', 'rq', 'rs', 'smi', 'cur_sentence'])
            df.to_csv(output_file_path, index=False)

    # 将结果转换为 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(results, columns=['rv', 'rq', 'rs', 'smi', 'cur_sentence'])
    df.to_csv(output_file_path, index=False)


def main_test(args):
    # 设置随机种子的值
    seed_value = int(opt.seed)
    seed_all(seed_value)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = torch.device(f'cuda:{0}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'./weights/linkergpt.pt', weights_only=True)

    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, tokenizer, device, output_file_path=args.output_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"运行时间: {elapsed_time:.4f} 秒")


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--output_file_path', default='./output/cts', help='name of .pt file')
    parser.add_argument('--device', default='7', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default='42', help='seed')

    opt = parser.parse_args()

    main_test(opt)

