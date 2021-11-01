from torch.utils.data import Dataset
from pathlib import Path
import torch
import os
import time
import random
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from data_utils import farthest_point_sample,index_points
import provider
from provider import normalize_data, shuffle_data, rotate_point_cloud

# 获得scene_name:
from tqdm import tqdm

BASE_dir = Path(__file__).resolve().parent.parent  # E://项目名
SCENE_dir = os.path.join(BASE_dir, 'data', 'nuscenes_moving_obstacle_detection')  # E://xiaomeng//data//moving_obstacle


# print( SCENE_dir_name)
def get_scene_name(dataroot):
    scene_name_all = os.listdir(dataroot)
    # print( scene_name_all)
    scene_name = []
    for name in scene_name_all:
        if os.path.isdir(os.path.join(dataroot, name)):
            scene_name.append(name)
    scene_name.sort()
    # print( scene_name)
    return scene_name


def nb_process_label(processed_label, sorted_label_voxel_pair):  # （480，360，32），(npoint,4)
    # print(processed_label.shape )
    # print(type( sorted_label_voxel_pair))
    label_size = 3
    counter = np.zeros((label_size,), dtype = np.uint16)
    # print(sorted_label_voxel_pair[0, 3]) 0
    counter[sorted_label_voxel_pair[0, 3]] = 1  # 获取第一个label, 统计对应标签有多少个点，出现的次数
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 取出体素索引 [478,23,0]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


# dataloader中获取可变序列自定义
def collate_fn_BEV(data):
    """
    # print( data) # 数据形式为[((5个属性),(5个属性))]，前一帧，后一帧  (type(data) )  #list
    # print('------',len(data[0]))  #tuple 2    # print('--0--',len(data[0][0]))  #tuple 5
    :param data:
    :return: 2
    """
    data_0_list = []
    data_1_list = []
    for d in data:
        data_0_list.append(d[0])
        data_1_list.append(d[1])
    # 前一帧
    data2stack_0 = torch.from_numpy(np.stack([d[0] for d in data_0_list]).astype(np.float32)  ) # (1,32,480,360)
    # print("data2stack ----------", data2stack_0.shape)  # torch.Size([8, 32, 480, 480])
    label2stack_0 =torch.from_numpy( np.stack([d[1] for d in data_0_list]) ) # (1,480,360,32)
    torch.cuda.empty_cache()

    grid_ind_stack_0 = [d[2] for d in data_0_list]  # list[[478 312 30],[],...]
    point_label_0 = [d[3] for d in data_0_list]  # list[[0], [0], [0],[19],....]
    xyz_0 = [d[4] for d in data_0_list]
    torch.cuda.empty_cache()

    # 当前帧
    data2stack_1 = torch.from_numpy(np.stack([d[0] for d in data_1_list]).astype(np.float32))  # (1,32,480,360)
    label2stack_1 = torch.from_numpy(np.stack([d[1] for d in data_1_list]))  # (1,480,360,32)
    torch.cuda.empty_cache()
    grid_ind_stack_1 = [d[2] for d in data_1_list]  # list[[478 312 30],[],...]
    point_label_1 = [d[3] for d in data_1_list]  # list[[0], [0], [0],[19],....]
    xyz_1 = [d[4] for d in data_1_list]
    torch.cuda.empty_cache()

    data_collate_0= (data2stack_0, label2stack_0, grid_ind_stack_0, point_label_0, xyz_0)
    torch.cuda.empty_cache()
    data_collate_1=( data2stack_1, label2stack_1, grid_ind_stack_1, point_label_1, xyz_1)
    return data_collate_0,data_collate_1



def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def cart2polar(input_xyz):  # (91943,3)
    # print( input_xyz.shape)
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)  # （91943，）
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    ffff = np.stack((rho, phi, input_xyz[:, 2]), axis = 1)
    torch.cuda.empty_cache()
    return np.stack((rho, phi, input_xyz[:, 2]), axis = 1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis = 0)


class FILEException(Exception):
    def __init__(self, label_file_path):
        super(FILEException, self).__init__()
        self.label_file_path = label_file_path


class My_nuscenes(Dataset):
    def __init__(self, dataroot, scene_list, npoints = 2048, centering = False, return_ref = False):
        """
        :param dataroot:
        :param centering: 归一化
        """
        super(My_nuscenes, self).__init__()
        self.dataroot = dataroot
        self.scene_list = scene_list
        self.centering = centering
        self.npoints = npoints
        self.sample_id_list = self.get_sample_id_list()  # 'scene-0001,000000,000001
        self.classes_original = {'background': 0, 'moving_vehicles': 1, 'moving_pedestrians': 2}
        self.return_ref = return_ref  # 加不加intensity，。

    def get_sample_id_list(self):
        """
        :return:
        ['scene-0001,000000,000001',
        'scene-0001,000001,000002'
        .....每个scene处理成了39对。
        ]
        """
        sample_id_list = []
        for scene_id, scene_name in enumerate(self.scene_list):
            file_list = os.listdir(os.path.join(os.path.join(self.dataroot, scene_name), 'lidar'))
            file_list.sort()
            for idx in range(len(file_list) - 1):  # 场景名，前一帧，当前一帧
                sample_id_list.append(
                    scene_name + ',' + file_list[idx].replace('.bin', '') + ',' + file_list[idx + 1].replace('.bin',
                                                                                                             ''))
        # print('sample_id_list', sample_id_list)
        return sample_id_list  #

    def get_lidar(self, scene_name, sample_id):
        lidar_file_path = os.path.join(self.dataroot, scene_name, 'lidar', sample_id + '.bin')
        assert os.path.exists(lidar_file_path)
        return np.fromfile(lidar_file_path, dtype = np.float32).reshape(-1, 3)  # shape: n*3

    def get_label(self, scene_name, sample_id):
        label_file_path = os.path.join(self.dataroot, scene_name, 'gt', sample_id + '.txt')
        # assert os.path.exists(label_file_path)
        # label_file = open(label_file_path)
        # lines = []
        # for line in label_file:
        #     lines.append(int(line.replace("\n", '')))
        # label_file.close()
        # return np.asarray(lines).reshape(-1, 2)
        try:
            os.path.exists(label_file_path)
        except FILEException as FE:
            print(FE.label_file_path)

        return np.loadtxt(label_file_path)  # n*2 matrix, n is the number of points

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        scene_name = self.sample_id_list[index].split(',')[0]
        sample_id0 = self.sample_id_list[index].split(',')[1]  # 前一帧
        sample_id1 = self.sample_id_list[index].split(',')[2]  # 当前帧

        pts_lidar0 = self.get_lidar(scene_name, sample_id0)
        pts_label0 = self.get_label(scene_name, sample_id0)

        pts_lidar1 = self.get_lidar(scene_name, sample_id1)
        pts_label1 = self.get_label(scene_name, sample_id1)  # (34688, 2)

        # pts_label0 =self.get_label( scene_name,sample_id0)
        # print( pts_label0  )
        # 开始选点和label
        # choice0 = np.random.choice(len(pts_label0 ), self.npoints, replace = True)
        #
        # choice1 = np.random.choice(len(pts_label1), self.npoints, replace = True)

        if self.centering == True:
            pts_lidar0 = pts_lidar0 - pts_lidar0.mean(0)
            pts_lidar1 = pts_lidar1 - pts_lidar1.mean(0)

        # 不要后面的insance标签,
        pts_label1 = pts_label1[:, 0]
        pts_label0 = pts_label0[:, 0]
        # 用于后面nb_process_label用，因为拿类别值当索引了，必须int32一下
        pts_label1 = pts_label1.astype(np.int32)
        pts_label0 = pts_label0.astype(np.int32)

        # pts_lidar0 = pts_lidar0[choice0, :]  # 返回选了2048个点的点云数据
        # pts_label0 = pts_label0[choice0]
        #
        # pts_lidar1 = pts_lidar1[choice1, :]  # 返回选了2048个点的点云数据
        # pts_label1 = pts_label1[choice1 ]
        # print( pts_label1.shape)
        # torch.Size([39, 2048, 3])
        data_tuple0 = (scene_name, sample_id0, pts_lidar0, pts_label0)
        data_tuple1 = (scene_name, sample_id1, pts_lidar1, pts_label1)
        data_tuple_all = (scene_name, sample_id0, pts_lidar0, pts_label0, sample_id1, pts_lidar1, pts_label1)
        return data_tuple_all


class spherical_dataset(Dataset):
    # max_volume_space 说的是distance，ignore_label=0, flip_aug= True rotate_aug =True
    def __init__(self, in_dataset, grid_size, rotate_aug = True, flip_aug = False, ignore_label = 0,
                 return_test = False, max_volume_space = [50, np.pi, 3], min_volume_space = [0, -np.pi, -5],
                 fixed_volume_space = True):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotate_aug_fun(self, xyz):
        rotate_rad = np.deg2rad(np.random.random() * 360)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        xyz[:, :2] = np.dot(xyz[:, :2], j)
        return xyz

    def flip_aug_fun(self, xyz):
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            xyz[:, 0] = -xyz[:, 0]
        elif flip_type == 2:
            xyz[:, 1] = -xyz[:, 1]
        elif flip_type == 3:
            xyz[:, :2] = -xyz[:, :2]
        return xyz

    def __getitem__(self, index):
        'Generates one sample of data'
        # (scene_name, sample_id0, pts_lidar0, pts_label0, sample_id1, pts_lidar1, pts_label1)
        # data = self.point_cloud_dataset[index]  # （xyz，label，intensity）
        scene_name, sample_id0, pts_lidar0, pts_label0, sample_id1, pts_lidar1, pts_label1 = self.point_cloud_dataset[
            index]  # （scene_name,binid,xyz，label）,（scene,binid,xyz，label）,（scene,binid,xyz，label）

        data0 = (pts_lidar0, pts_label0)  # (前一帧  的  xyz,label)
        data1 = (pts_lidar1, pts_label1)  # (当前帧  的  xyz,label)
        torch.cuda.empty_cache()

        xyz_before, labels_before = data0
        xyz_now, labels_now = data1
        torch.cuda.empty_cache()

        labels_before = labels_before.reshape((-1, 1))
        labels_now = labels_now.reshape((-1, 1))
        torch.cuda.empty_cache()
        # 数据旋转
        if self.rotate_aug:
            xyz_now = self.rotate_aug_fun(xyz_now)  # 只做x，y的变换
            xyz_before = self.rotate_aug_fun(xyz_before)  # 只做x，y的变换

        # 数据翻转-- flip x , y or x+y  # 1 对x翻转，2 y，3 x+y
        if self.flip_aug:
            xyz_now = self.flip_aug_fun(xyz_now)  # 只做x，y的变换
            xyz_before = self.flip_aug_fun(xyz_before)

        # convert coordinate into polar coordinates，--笛卡尔变polar
        xyz_pol_before = cart2polar(xyz_before)  # （91943，3）
        xyz_pol_now = cart2polar(xyz_now)  # （91943，3）

        # 计算体素边界---
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        crop_range = max_bound - min_bound  # [47.          6.28318531  4.5       ]
        cur_grid_size = self.grid_size  # [480,360,32]
        intervals = crop_range / (cur_grid_size - 1)  # 多少个间隙，所以-1， [0.09812109 0.01750191 0.14516129]
        if (intervals == 0).any(): print("Zero interval!")

        # 获得体素索引  --------get grid index, 当前点和前一帧，还是各存个的吧，
        # （11943，3）[[478,13,30] ,[ 478,26,30],....]
        grid_ind_before = (np.floor((np.clip(xyz_pol_before, min_bound, max_bound) - min_bound) / intervals)).astype(
            np.int)

        grid_ind_now = (np.floor((np.clip(xyz_pol_now, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
        torch.cuda.empty_cache()

        # process voxel position   计算体素位置 ---,tuple 3(480, 360, 32)
        voxel_position = np.zeros(self.grid_size, dtype = np.float32)  #
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1  # tuple [-1  1  1  1]
        # voxel_position: (3,480,360,32), min_bound.reshape()---> [3,1,1,1],之前减去min现在加上
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        voxel_position_before = voxel_position
        voxel_position_now = voxel_position

        # ------------before帧  ----------的处理
        processed_label_before = np.ones(self.grid_size, dtype = np.uint8) * self.ignore_label  # 全0 （480，360，32）
        label_voxel_pair_before = np.concatenate([grid_ind_before, labels_before],
                                                 axis = 1)  # (92074, 4)[171 223 26 15]
        label_voxel_pair_before = label_voxel_pair_before[
                                  np.lexsort((grid_ind_before[:, 0], grid_ind_before[:, 1], grid_ind_before[:, 2])),
                                  :]  # 对于label_voxle_pair排序，基于z
        processed_label_before = nb_process_label(np.copy(processed_label_before), label_voxel_pair_before)
        # data_tuple = (voxel_position,processed_label)
        # prepare visiblity feature, find max distance index in each angle,height pair
        valid_label_before = np.zeros_like(processed_label_before, dtype = bool)  # (480,360, 32)
        valid_label_before[
            grid_ind_before[:, 0], grid_ind_before[:, 1], grid_ind_before[:, 2]] = True  # 对应体素位置为True，其余false
        valid_label_before = valid_label_before[::-1]  # 倒排
        max_distance_index_before = np.argmax(valid_label_before, axis = 0)  # (360, 32)
        max_distance_before = max_bound[0] - intervals[0] * (max_distance_index_before)  # # (360, 32)
        distance_feature_before = np.expand_dims(max_distance_before, axis = 2) - np.transpose(voxel_position_before[0],
                                                                                               (1, 2, 0))  # 360，32，480
        distance_feature_before = np.transpose(distance_feature_before, (1, 2, 0))  # 32,480,360
        # convert to boolean feature ,>0 的地方为-1，小于0的地方为0
        distance_feature_before = (distance_feature_before > 0) * -1.
        # (32,480,360)
        distance_feature_before[grid_ind_before[:, 2], grid_ind_before[:, 0], grid_ind_before[:, 1]] = 1.
        # （（32,480,360），(480,360,32,)）
        data_tuple_before = (distance_feature_before, processed_label_before)

        # center data on each voxel for PTnet
        voxel_centers_before = (grid_ind_before.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz_before = xyz_pol_before - voxel_centers_before
        # return_xyz (npoint,3),xyz_pol （npoint,3），
        return_xyz_before = np.concatenate((return_xyz_before, xyz_pol_before, xyz_before[:, :2]),
                                           axis = 1)  # 8维 （npoint， 8）
        return_fea_before = return_xyz_before

        # -------------------now的处理
        processed_label_now = np.ones(self.grid_size, dtype = np.uint8) * self.ignore_label  # 全0 （480，360，32）
        label_voxel_pair_now = np.concatenate([grid_ind_now, labels_now],
                                              axis = 1)  # (92074, 4)[171 223 26 15]
        label_voxel_pair_now = label_voxel_pair_now[
                               np.lexsort((grid_ind_now[:, 0], grid_ind_now[:, 1], grid_ind_now[:, 2])),
                               :]  # 对于label_voxle_pair排序，基于z
        processed_label_now = nb_process_label(np.copy(processed_label_now), label_voxel_pair_now)
        # data_tuple = (voxel_position,processed_label)
        # prepare visiblity feature, find max distance index in each angle,height pair
        valid_label_now = np.zeros_like(processed_label_now, dtype = bool)  # (480,360, 32)
        valid_label_now[
            grid_ind_now[:, 0], grid_ind_now[:, 1], grid_ind_now[:, 2]] = True  # 对应体素位置为True，其余false
        valid_label_now = valid_label_now[::-1]  # 倒排
        max_distance_index_now = np.argmax(valid_label_now, axis = 0)  # (360, 32)
        max_distance_now = max_bound[0] - intervals[0] * (max_distance_index_now)  # # (360, 32)
        distance_feature_now = np.expand_dims(max_distance_now, axis = 2) - np.transpose(voxel_position_now[0],
                                                                                         (1, 2, 0))  # 360，32，480
        distance_feature_now = np.transpose(distance_feature_now, (1, 2, 0))  # 32,480,360
        # convert to boolean feature ,>0 的地方为-1，小于0的地方为0
        distance_feature_now = (distance_feature_now > 0) * -1.
        # (32,480,360)
        distance_feature_now[grid_ind_now[:, 2], grid_ind_now[:, 0], grid_ind_now[:, 1]] = 1.
        # （（32,480,360），(480,360,32,)）
        data_tuple_now = (distance_feature_now, processed_label_now)

        # center data on each voxel for PTnet
        voxel_centers_now = (grid_ind_now.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz_now = xyz_pol_now - voxel_centers_now
        # return_xyz (npoint,3),xyz_pol （npoint,3），
        return_xyz_now = np.concatenate((return_xyz_now, xyz_pol_now, xyz_now[:, :2]),
                                        axis = 1)  # 8维 （npoint， 8）
        return_fea_now = return_xyz_now
        torch.cuda.empty_cache()
        # return
        data_tuple_now += (grid_ind_now, labels_now, return_fea_now)
        data_tuple_before += (grid_ind_before, labels_before, return_fea_before)
        # print(data_tuple_before[0][0].shape)  --(480,360)
        # (distance_feature,processed_label,索引，label，特征)   （（32,480,360），(480,360,32,),suoyin,）
        data_tuple = (data_tuple_before, data_tuple_now)
        return data_tuple


if __name__ == '__main__':
    BASE_dir = Path(__file__).resolve().parent.parent  # E://xiaomeng
    SCENE_dir = os.path.join(BASE_dir, 'data', 'nuscenes_moving_obstacle_detection')  # E://xiaomeng//data
    scene_list = get_scene_name(SCENE_dir)
    # print( scene_list)
    nuse = My_nuscenes(SCENE_dir, scene_list)
    polarSpData = spherical_dataset(nuse)
    # datalo = NuscenesDataset(SCENE_dir)
