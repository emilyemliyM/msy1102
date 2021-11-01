import numpy as np
import torch

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength = n ** 2)
    torch.cuda.empty_cache()
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist

#
# # 统计参数量
# total = sum([param.nelement() for param in my_model.parameters()])
# print("Number of parameter: % .2fM" % (total / 1e6))

# 计算gpu显存
def get_gpu_memory():
    import platform
    if 'Windows' != platform.system():
        import os
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')
    else:
        memory_gpu = False
        print("显存计算不支持windows")
    return memory_gpu