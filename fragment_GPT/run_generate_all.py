import os
import time
import subprocess

start_time = time.time()  # 记录开始时间

os.chdir('/data1/ytg/GA_llm/fragment_GPT')

for i in range(10):
    cmd = 'python3 generate_all.py' + ' ' + f'--seed {i}'
    subprocess.Popen(cmd, shell=True)  # 异步执行命令，不会阻塞
    time.sleep(2)  # 等待10秒

end_time = time.time()  # 记录结束时间

total_time = end_time - start_time  # 计算总时间
print(f"Total execution time: {total_time:.2f} seconds")