#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter

class ptBEVnet(nn.Module):
    # fea_compre 32, fea_dim 9 ,grid_size[480,360,32] ,out_pt_fea_dim 512 ,max_pt_per_encode 256 ， fea_compre: 表示的是维度压缩，就是512维换个小点儿的32
    def __init__(self, BEV_net, grid_size, pt_model = 'pointnet', fea_dim = 3, pt_pooling = 'max', kernal_size = 3,
                 out_pt_fea_dim = 64, max_pt_per_encode = 64, cluster_num = 4, pt_selection = 'farthest', fea_compre = None):
        super(ptBEVnet, self).__init__()
        assert pt_pooling in ['max']
        assert pt_selection in ['random','farthest']
        
        if pt_model == 'pointnet':
            self.PPmodel = nn.Sequential(
                nn.BatchNorm1d(fea_dim),
                
                nn.Linear(fea_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                
                nn.Linear(256, out_pt_fea_dim)
            )  # 输出512维特征
        
        self.pt_model = pt_model
        self.BEV_model = BEV_net  # unet
        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.pt_selection = pt_selection
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        
        # NN stuff
        if kernal_size != 1:
            if self.pt_pooling == 'max':
                self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size-1)//2, dilation=1)
            else: raise NotImplementedError
        else: self.local_pool_op = None
        
        # parametric pooling        
        if self.pt_pooling == 'max':
            self.pool_dim = out_pt_fea_dim
        
        # point feature compression
        # 512,32
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim
        
    def forward(self, pt_fea, xy_ind, voxel_fea=None):
        # pt_
        cur_dev = pt_fea[0].get_device()  # -1，， why?因为list包了一层，所以[0]后才是全部的点特征 9维
        # xy_ind （npoint,2）
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)): # len 为1 因为list[]包裹，就是补0的操作了相当于
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch))  # 前面补充一维

        cat_pt_fea = torch.cat(pt_fea,dim = 0) # 就是把list外壳去掉，恢复原来的点特征格式，torch.Size([113316, 9])
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)  # 同理，torch.Size([113316, 3]) ，由于之前的pad操作现在的第一维都是0，其余为，xy
        pt_num = cat_pt_ind.shape[0] # 就是 点数

        # shuffle the data
        # torch.randperm 返回一个0-n-1的数组
        # shuffled_ind = torch.randperm(pt_num,device = cur_dev)
        shuffled_ind = torch.randperm(pt_num)

        cat_pt_fea = cat_pt_fea[shuffled_ind,:]
        cat_pt_ind = cat_pt_ind[shuffled_ind,:]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        
        # subsample pts
        if self.pt_selection == 'random':
            # unq_cnt: 旧在新出现的次数
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt  # max_pt 256 就是 取256个点
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv.detach().cpu().numpy()), np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]))
            remain_ind = np.zeros((pt_num,),dtype = np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:,:3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds,:],self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1
            
        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)
        
        # process feature
        if self.pt_model == 'pointnet':  # preocessed_cat_pt_fea torch.Size([91606, 512])
            processed_cat_pt_fea = self.PPmodel(cat_pt_fea) # cat_pt_fea: 点数，9
        
        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]  # 按照inv找出max  torch.Size([20703, 512])
        else: raise NotImplementedError
        
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data) # torch.Size([20703, 32])
        else:
            processed_pooled_data = pooled_data
        
        # stuff pooled data into 4D tensor
        # 1，480，360，32  ，len(list)->1
        out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        # out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data = torch.zeros(out_data_dim, dtype=torch.float32)

        out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data
        out_data = out_data.permute(0,3,1,2)
        if self.local_pool_op != None:  # 没用
            out_data = self.local_pool_op(out_data)
        if voxel_fea is not None: # 没用
            out_data = torch.cat((out_data, voxel_fea), 1)
        
        # run through network  out_data: 1,32,480,360, net_return_data:  torch.Size([1, 19, 480, 360, 32])
        net_return_data = self.BEV_model(out_data)
        
        return net_return_data
    
def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)  # npoint的tensor tensor([    5,     7,     8,  ..., 91443, 91452, 91455])
    # id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr = torch.ones(idx[-1],dtype = torch.int64)

    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FPS(np_cat_fea,K)

# 加速， xyz二维数组，k： int，  返回 布尔值一维数组
@nb.jit('b1[:](f4[:,:],i4)',nopython=True,cache=True)
def nb_greedy_FPS(xyz,K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num,1),dtype = np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j,0] = np.sum(xyz_sq[j,:])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2*np.dot(xyz, np.transpose(xyz))
    
    candidates_ind = np.zeros((sample_num,),dtype = np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,),dtype = np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)
    
    for i in range(1,K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:,start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind,:]
            cur_dis = cur_dis[:,candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],),dtype = np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j,:])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False
        
    return candidates_ind