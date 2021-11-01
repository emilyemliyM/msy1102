import os
import numpy as np
import torch
# 实现指定 cuda ：0
"""
gpu_id = 0
gpu_str = "cuda:{}".format(gpu_id)
device = torch.device(gpu_str if torch.cuda.is_available() else "cpu")
print( device)
"""
# gpu的数量统计
"""
device_count = torch.cuda.device_count()
print("device_count: {}".format(device_count))

device_name = torch.cuda.get_device_name(0)
print("device_name: {}".format(device_name))
"""

# 设置 指定gpu
"""
gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print( device)
"""
# 设置多个gpu
def get_gpu_memory():
    import platform
    if 'Windows' != platform.system():
        import os
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')
    else:
        memory_gpu = False
        print("显存计算功能暂不支持windows操作系统")
    return memory_gpu


gpu_memory = get_gpu_memory()
if not gpu_memory:
    print("gpu free memory: {}".format(gpu_memory))
    gpu_list = np.argsort(gpu_memory)[::-1]

    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
