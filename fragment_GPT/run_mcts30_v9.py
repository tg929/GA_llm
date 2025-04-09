import os
import time
import subprocess

start_time = time.time()  # 记录开始时间

os.chdir('/data1/ytg/GA_llm/fragment_GPT')

for i in range(30):
    if i < 15:
        device = 7
    else:
        device = 6
    cmd = 'python3 mcts_experiment_v9.py' + ' ' + '--output_file_path' + ' ' + f'./output/mcts_v9/mcts_5ht1b_v9_{i}.csv' + ' '\
                                          + f'--device {device}' + ' ' + f'--seed {i}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")