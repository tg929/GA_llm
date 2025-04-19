#!/bin/bash

# 运行继续脚本从第30代到第50代
# 使用多处理器和多线程模式提高性能

echo "开始从第30代继续运行到第50代..."

python GA_llm_continue.py \
  --start_from 30 \
  --generations 50 \
  --number_of_processors -1 \
  --multithread_mode multithreading \
  --LipinskiLenientFilter \
  --PAINSFilter \
  --num_crossovers 50 \
  --num_mutations 50

echo "运行完成!" 