import torch
from utils.train_utils import seed_all
import os
import argparse
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
from mcts_mpo import MCTSConfig, MolecularProblemState, MCTS, print_best
from utils.chem_utils import sentence2mol
from rdkit import rdBase
from tdc import Oracle
from tqdm import tqdm
from fragpt_ppo import FRAGPT_Optimizer
import yaml

# 禁用所有日志信息
rdBase.DisableLog('rdApp.warning')

def main_test(args):
    # 设置随机种子的值
    seed_value = int(opt.seed)
    # seed_all(seed_value)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = torch.device(f'cuda:{0}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    # checkpoint = torch.load(f'./weights/linkergpt.pt', weights_only=True)

    config_default = yaml.safe_load(open('./hparams_default.yaml'))

    start_time = time.time()
    optimizer = FRAGPT_Optimizer(model)
    MPO = ['QED', 'DRD2', 'GSK3b', 'JNK3', 'Albuterol_Similarity', 'Amlodipine_MPO', 'Celecoxib_Rediscovery',
            'Deco_Hop',
            'Fexofenadine_MPO', 'Isomers_C7H8N2O2', 'Isomers_C9H10N2O2PF2Cl', 'Median1', 'Median2',
            'Mestranol_Similarity',
            'Osimertinib_MPO', 'Perindopril_MPO', 'Ranolazine_MPO', 'Scaffold_Hop', 'Sitagliptin_MPO',
            'Thiothixene_Rediscovery',
            'Troglitazone_Rediscovery', 'Valsartan_SMARTS', 'Zaleplon_MPO']
    for mpo_name in MPO:
        oracle = Oracle(name=mpo_name)
        optimizer.optimize(oracle=oracle, config=config_default, mconf=mconf, tokenizer=tokenizer, seed=seed_value)
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
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default='42', help='seed')

    opt = parser.parse_args()

    main_test(opt)

