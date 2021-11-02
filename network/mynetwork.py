import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing
# pointnet encode
import torch_scatter
from dropblock import DropBlock2D
# from network.pvt_msy_v1 import PyramidVisionTransformer


class PPmodel(nn.Module):
    def __init__(self, fea_dim, out_pt_fea_dim):
        """

        :param fea_dim: 8特征维度的npoint
        :param out_pt_fea_dim:转为512
        :return shape :(npoints out_pt_fea_dim)
        """
        super(PPmodel, self).__init__()
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),

            nn.Linear(64, 128, bias =  False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),

            nn.Linear(128, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),

            nn.Linear(256, out_pt_fea_dim, bias = False)
        )  #
        self.initialize()

    def forward(self, x):
        x = self.PPmodel(x)
        # print( "vhatch size=  sffg", x.shape)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PPmodel_all_preprocess(nn.Module):
    def __init__(self, grid_size, fea_dim = 8, pt_pooling = 'max', kernal_size = 3,
                 out_pt_fea_dim = 512, max_pt_per_encode = 64, cluster_num = 4, fea_compre = None):
        super(PPmodel_all_preprocess, self).__init__()
        self.ppmodel = PPmodel(fea_dim, out_pt_fea_dim)
        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.grid_size = grid_size
        self.fea_compre = fea_compre

        if self.pt_pooling == 'max':
            self.pool_dim = out_pt_fea_dim

        # point feature compression
        # 512,32
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre,bias =  False),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):
        # pt_
        cur_dev = pt_fea[0].get_device()  # -1，， why?因为list包了一层，所以[0]后才是全部的点特征 9维，当时debug的时候gpu不可用，所以返回-1了，正常是0，1，。。。
        # xy_ind （npoint,2）
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):  # len 为1 因为list[]包裹，就是补0的操作了相当于
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value = i_batch))  # 前面补充一维

        cat_pt_fea = torch.cat(pt_fea, dim = 0)  # 就是把list外壳去掉，恢复原来的点特征格式，torch.Size([113316, 9])
        cat_pt_ind = torch.cat(cat_pt_ind, dim = 0)  # 同理，torch.Size([113316, 3]) ，由于之前的pad操作现在的第一维都是0，其余为，xy
        pt_num = cat_pt_ind.shape[0]  # 就是 点数

        # shuffle the data
        # torch.randperm 返回一个0-n-1的数组
        shuffled_ind = torch.randperm(pt_num,device = cur_dev)
        # shuffled_ind = torch.randperm(pt_num)

        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse = True, return_counts = True, dim = 0)
        unq = unq.type(torch.int64)

        # subsample pts随机采样
        # unq_cnt: 旧在新出现的次数
        grp_ind = grp_range_torch(unq_cnt, cur_dev)[torch.argsort(torch.argsort(unq_inv))]
        remain_ind = grp_ind < self.max_pt  # max_pt 256 就是 取256个点

        cat_pt_fea = cat_pt_fea[remain_ind, :]
        cat_pt_ind = cat_pt_ind[remain_ind, :]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt, max = self.max_pt)

        # process feature
        # if self.pt_model == 'pointnet':  # preocessed_cat_pt_fea torch.Size([91606, 512])
        processed_cat_pt_fea = self.ppmodel(cat_pt_fea)  # cat_pt_fea: 点数，9
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim = 0)[0]
            # 按照inv找出max  torch.Size([20703, 512])
        else:
            raise NotImplementedError

        processed_pooled_data = self.fea_compression(pooled_data)  # torch.Size([20703, 32])
        torch.cuda.empty_cache()
        # stuff pooled data into 4D tensor
        # 1，480，360，32  ，len(list)->1
        out_data_dim = [len(pt_fea), self.grid_size[0], self.grid_size[1], self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        # out_data = torch.zeros(out_data_dim, dtype = torch.float32)

        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data
        out_data = out_data.permute(0, 3, 1, 2)

        return out_data


# 随机采样
def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)  # npoint的tensor tensor([    5,     7,     8,  ..., 91443, 91452, 91455])
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    # id_arr = torch.ones(idx[-1], dtype = torch.int64)

    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)


# #  最远点采样
# def parallel_FPS(np_cat_fea,K):
#     return  nb_greedy_FPS(np_cat_fea,K)


# 一个polar的conv 块
class conv_circular(nn.Module):
    '''conv => BN => ReLU'''  # in 32, out_ch 64 ,group_conv False

    def __init__(self, in_ch = 32, out_ch = 64,kernel_size=3 , group_conv = True, dilation = 1):
        super(conv_circular, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding = (1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, x):  # torch.Size([1, 32, 480, 360])
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode = 'circular')  # 1,32,480,362
        x = self.conv1(x)  # torch.Size([1, 64, 480, 360])
        return x


# 时序头的定义2个帧的输入  2 conv + 时序头SElayer+ 1个 conv，输入输出通道一致
class Template_class(nn.Module):
    def __init__(self, in_ch = 512, out_ch = 512):
        super(Template_class, self).__init__()
        self.conv_circular = double_conv_circular(in_ch, out_ch)
        self.conv1d = conv_circular(in_ch * 2, in_ch // 8)
        # self.con1d=  torch.max( data,1,keepdim= True)[0]
        self.se = SELayer(out_ch * 2)

    def forward(self, x1, x2):  # x1 前一帧，x2当前帧
        _, C, _, _ = x2.size()
        # print("--C", C)  # 512
        x2 = self.conv_circular(x2)
        # print("x2.shape", x2.shape)  # 512
        x1 = self.conv_circular(x1)
        x1 = torch.cat((x1, x2), dim = 1)
        # 时序头
        x1 = self.se(x1)
        x1 = torch.max(x1, 1, keepdim = True)[0]
        x1 = x1.repeat(1, C, 1, 1)  # 512
        # x= torch.cat( (x1,x2),dim = 1)
        torch.cuda.empty_cache()
        x = x1 + x2
        # print("-repeat-", x.shape)
        return x


class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''  # in 32, out_ch 64 ,group_conv False

    def __init__(self, in_ch = 32, out_ch = 64, group_conv = True, dilation = 1):
        super(double_conv_circular, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch//2, 3, padding = (1, 0), groups = out_ch//2),
            nn.Conv2d(in_ch, out_ch // 2, 3, padding = (1, 0)),

            nn.BatchNorm2d(out_ch // 2),
            nn.LeakyReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch // 2, out_ch, 3, padding = (1, 0)),
            # nn.Conv2d(out_ch//2, out_ch, 3, padding = (1, 0), groups = out_ch//2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, x):  # torch.Size([1, 32, 480, 360])
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode = 'circular')  # 1,32,480,362
        x = self.conv1(x)  # torch.Size([1, 64, 480, 360])
        torch.cuda.empty_cache()
        x = F.pad(x, (1, 1, 0, 0), mode = 'circular')  # 1,64,480,362
        x = self.conv2(x)  # 1，64，480，360   这里面啊32 是特征
        return x


# self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")上采样的一种, 用来定义时序头
class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
        """
        We found empirically that on ResNet
        architectures, removing the biases of the FC layers in the
        excitation operation facilitates the modelling of channel
        dependencies
        """

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class down(nn.Module):
    # in:64, out:128, ciucular_padding: True,对 后两维下采样
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv = group_conv, dilation = dilation)
            )

    def forward(self, x):  # torch.Size([1, 64, 480, 360])
        x = self.mpconv(x)  # torch.Size([1, 128, 240, 180])
        return x


# BEV_Unet包着 UNetEncoder编码，+ 上采样的类，还没写
class BEV_Unet_encoder(nn.Module):

    def __init__(self,  n_height=32, dilation = 1, group_conv = False, input_batch_norm = False, dropout = 0.,
                 circular_padding = False, dropblock = True, ):
        super(BEV_Unet_encoder, self).__init__()
        # self.n_class = n_class  # 19
        self.n_height = n_height  # 32

        self.network = UNetEncoder(n_height, dilation, group_conv, input_batch_norm, dropout,
                                       circular_padding, dropblock)

    def forward(self, x):  # 1,32,480,360,
        x = self.network(x)  # torch.Size([1, 256, 480, 360])
        # x = x.permute(0, 2, 3, 1)  # 1，480，360， 608
        # new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        # x = x.view(new_shape)  # 1,480,360,32,19
        # x = x.permute(0, 4, 1, 2, 3)  # 1,19,480,360,32
        return x


class UNetEncoder(nn.Module):
    #  n_class :608, group_conv=False, n_height =32  # 因为压缩后的特征是32
    def __init__(self, n_height, dilation, group_conv, input_batch_norm, dropout, circular_padding, dropblock):
        super(UNetEncoder, self).__init__()
        # self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        # self.down3 = down(256, 256, dilation, group_conv, circular_padding)

    def forward(self, x):  # torch.Size([1, 32, 480, 480])
        # x1 = self.inc(x)  # torch.Size([1, 64, 480, 480])
        x2 = self.down1(x)  # torch.Size([1, 128, 240, 240])
        x3 = self.down2(x2)  # torch.Size([1, 256, 120, 120])  x4是经过conv下采样4倍的 后面用于concat
        # x4 = self.down3(x3)  # x4 torch.Size([1, 256, 60, 60]) x4是经过conv下采样8倍的 后面用于concat
        return x3


class up(nn.Module):
    # bilinear True ,circular_padding True, in 1024, out_256
    def __init__(self, in_ch, out_ch, circular_padding= True, scale_factor=2, group_conv = False, use_dropblock = False,
                 drop_p = 0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        # self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride = 2, groups = in_ch // 2)
        # self.up = nn.ConvTranspose2d(512, 256, 2, stride = 2)
        self.up = nn.Upsample(scale_factor = scale_factor, mode = 'bilinear', align_corners = True)

        if circular_padding:  # 1024,256
            self.conv = double_conv_circular(in_ch, out_ch, group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size = 7, drop_prob = drop_p)

    def forward(self, x1, x2):  # x1 torch.Size([1, 512, 30, 22]),x2 torch.Size([1, 512, 60, 45])
        x1 = self.up(x1)
        # 双线性插值后 ，x1 torch.Size([1, 512, 60, 44])
        # input is CHW 因为除不开45/2 所以 并不能直接x2 ，还得补一个 用pad补回来
        diffY = x2.size()[2] - x1.size()[2]  # 0
        diffX = x2.size()[3] - x1.size()[3]  # 1

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))  # 45 了就，

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim = 1)  # torch.Size([1, 1024, 60, 45])
        x = self.conv(x)  # torch.Size([1, 256, 60, 45])
        if self.use_dropblock:
            x = self.dropblock(x)
        return x  # up1 torch.Size([1, 256, 60, 45])


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

#  先将 特征输成64维，本来经过特征压缩是32维了
class inconv(nn.Module):  # 一层卷积, pointnet后的第一层卷积
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv = False, dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv = False, dilation = dilation)

    def forward(self, x):  # torch.Size([1, 32, 480, 360])
        x = self.conv(x)  # torch.Size([1, 64, 480, 360])
        return x








# if __name__ == '__main__':
#     data1 = torch.randn(100, 8)  # dangqianzhen
#     data = torch.randn(102, 8)
#     # model = PPmodel(8, 512)
#     # fea = model(data1)
#     # fea2 = model(data)
#     #
#     # print(fea.shape)
#     # print(fea2.shape)
#     model = PPmodel_all_preprocess(grid_size = [480, 360, 32], max_pt_per_encode = 256, kernal_size = 1,
#                                    fea_compre = 32)
#
# if __name__ == '__main__':
#     model =PolarSegFormer(grid_size= [480,480,32])
#     data= [torch.randn( 100,2)]
#     data_fea=[ torch.randn( 100,8)]
#     output=model( data_fea, data,data_fea, data)
#     print(output.shape)
if __name__ == '__main__':
    data = torch.randn(1, 64, 480, 360)
    model= BEV_Unet_encoder(n_height=32, dilation=1, group_conv=False, input_batch_norm=True, dropout=0, circular_padding=True, dropblock=True )
    out= model( data)
    print( out.shape)