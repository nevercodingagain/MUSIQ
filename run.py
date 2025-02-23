# run_train.py
import os
from utils.gpu_util import GPUGet

gpu_getter = GPUGet(
        min_gpu_number=1,
        max_gpu_number=4,
        required_mem=1000,
        max_power=200
    )
available_gpus = gpu_getter.get_available_gpus()
if not available_gpus:
        raise RuntimeError("没有找到符合条件的可用GPU")

gpu_list_str = ','.join(map(str, available_gpus))

# 设置环境变量并启动训练
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_str
print(f"CUDA_VISIBLE_DEVICES = %s" % gpu_list_str)
os.system(f'torchrun --nproc_per_node={len(available_gpus)} --nnodes=1 --master_port=29500 train.py')
