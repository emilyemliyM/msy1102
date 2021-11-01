import os
import time
import argparse
from pathlib import Path
from torch.utils.data import random_split
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader_nuscenes.nusenes_dataset import collate_fn_BEV, My_nuscenes, get_scene_name, spherical_dataset
from network.lovasz_losses import lovasz_softmax
import warnings

warnings.filterwarnings("ignore")


def SemKITTI2train(label):  # label torch.Size([1, 480, 360, 32])
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    return label - 1  # uint8 trick


def parse_args():
    parser = argparse.ArgumentParser(description = 'PolarNuscenes')
    parser.add_argument('-d', '--data_dir', default = 'data//nuscenes_moving_obstacle_detection')
    parser.add_argument('-p', '--model_save_path', default = './SemKITTI_PolarSeg.pt')
    parser.add_argument('-s', '--grid_size', nargs = '+', type = int, default = [480, 360, 32])
    parser.add_argument('--train_batch_size', type = int, default = 2, help = 'batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type = int, default = 2, help = 'batch size for validation (default: 2)')
    parser.add_argument('--check_iter', type = int, default = 2)
    parser.add_argument('--log_dir', type = str, default = "runs//", help = 'Log path [default: None]')
    return parser.parse_args()


SemKITTI_label_name = {0: 'background', 1: 'moving_vehicles', 2: 'moving_pedestrains'}

BASE_dir = Path(__file__).resolve().parent  # E://xiaomeng
SCENE_dir = os.path.join(BASE_dir, 'data', 'nuscenes_moving_obstacle_detection')
scene_list = get_scene_name(SCENE_dir)


def main():
    args = parse_args()
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    check_iter = args.check_iter
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]  # 32 GRID SIZE 的最后一维
    grid_size = args.grid_size  # [480, 360, 32]
    # pytorch_device = torch.device('cuda:0')
    # pytorch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = 'polar'  # 'polar'
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[
                   1:] - 1  # 2 list [ 0  1   ]
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]  # 取出除了背景以外的 类别名字

    # max_pt_per_encode 每个体素最大包含点云数
    my_BEV_model = BEV_Unet(n_class = len(unique_label), n_height = compression_model, input_batch_norm = True,
                            dropout = 0.5, circular_padding = True)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size = grid_size, fea_dim = 8,
                        max_pt_per_encode = 256,
                        out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path))
    # my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters())
    # loss_fun = torch.nn.CrossEntropyLoss(ignore_index = 0)
    loss_fun = torch.nn.CrossEntropyLoss()

    # prepare dataset  1当前帧
    pt_dataset = My_nuscenes(data_path, scene_list)
    # print( pt_dataset)
    all_dataset = spherical_dataset(pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,
                                    rotate_aug = True, fixed_volume_space = True)

    train_size = int(len(all_dataset) * 0.6)
    test_size = int(len(all_dataset) * 0.2)
    val_size = len(all_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size, val_size])



    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = train_batch_size,
                                                 collate_fn = collate_fn_BEV,
                                                 shuffle = True, num_workers = 0)

    """#  test和 val的 collate_fn_BEV 需要换，还没写，哈哈哈哈。。。 不要忘了！！"""
    test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = train_batch_size,
                                                       collate_fn = collate_fn_BEV,
                                                       shuffle = True, num_workers = 0)

    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = train_batch_size,
                                                       collate_fn = collate_fn_BEV,
                                                       shuffle = True, num_workers = 0)

    # training
    epoch = 0
    best_val_miou = 0
    start_training = False
    my_model.train()
    global_iter = 0
    exce_counter = 0

    while (epoch <2 ):
        loss_list = []
        pbar = tqdm(total = len(train_dataset_loader))
        for i_iter, data in enumerate(train_dataset_loader):
            # 是data2stack_1, label2stack_1, grid_ind_stack_1, point_label_1, xyz_1(8wei),后面3都是list
            data_0=data[0] # 前一帧
            data_1=data[1] # 当前帧
            _, train_vox_label_0, train_grid_0, _, train_pt_fea_0 = data_0
            _, train_vox_label_1, train_grid_1, _, train_pt_fea_1 = data_1

            # print( train_pt_fea_0[0].shape)

            train_vox_label_0 = SemKITTI2train(train_vox_label_0)  # 全255
            # 转换数据类型
            train_pt_fea_ten_0 = [torch.from_numpy(i).type(torch.FloatTensor) for i in
                                train_pt_fea_0]
            # train_grid: [nponint,3]  ,trian_grid_ten: 不要z了
            # 放到cuda gpu里
            train_grid_ten_0 = [torch.from_numpy(i[:, :2]) for i in train_grid_0]
            train_vox_ten_0 = [torch.from_numpy(i) for i in train_grid_0]
            point_label_tensor_0 = train_vox_label_0.type(torch.LongTensor)

            # forward + backward + optimize
            # train_pt_fea_ten (npoints，9),  train_grid_ten ( torch.Size([113316, 2])), 这俩均被[]list包了一层
            outputs_0 = my_model(train_pt_fea_ten_0, train_grid_ten_0)
            # loss_fun:是交叉熵
            # loss_0 = lovasz_softmax(torch.nn.functional.softmax(outputs_0), point_label_tensor_0) + loss_fun(outputs_0, point_label_tensor_0)
            loss_0 = lovasz_softmax(torch.nn.functional.softmax(outputs_0), point_label_tensor_0, ignore = 0)
            print('--l0ss---', loss_0.item())
            loss_0.backward()
            optimizer.step()
            loss_list.append(loss_0.item())
            optimizer.zero_grad()
            pbar.update(1)
            start_training = True
            global_iter += 1
        pbar.close()
        epoch += 1




if __name__ == '__main__':
    main()
