import torch
import torch.nn as nn
from functools import partial
from network.mynetwork import PPmodel_all_preprocess, Template_class, BEV_Unet_encoder, conv_circular, up,outconv
# from network.pvt_msy_v1 import PyramidVisionTransformer
from network.pvt_v2 import PyramidVisionTransformerV2
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import warnings

warnings.filterwarnings("ignore")

@register_model
def pvt_v2_b0(pretrained = False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size = 4, embed_dims = [128, 512], num_heads = [2, 8], mlp_ratios = [8, 4], qkv_bias = True,
        norm_layer = partial(nn.LayerNorm, eps = 1e-5), depths = [2, 2], sr_ratios = [8, 4], linear = True,
        **kwargs)
    model.default_cfg = _cfg()

    return model


# total model

class PolarSegFormer(nn.Module):
    def __init__(self, n_class, grid_size, fea_dim = 8, n_height = 32, pt_pooling = 'max', kernal_size = 3,
                 out_pt_fea_dim = 512, max_pt_per_encode = 64, cluster_num = 4, fea_compre = None):
        super(PolarSegFormer, self).__init__()

        self.bev_Unet_encoder = BEV_Unet_encoder(n_height = 32, dilation = 1, group_conv = False,
                                                 input_batch_norm = True, dropout = 0, circular_padding = True,
                                                 dropblock = True)

        # self.ppmodel = PPmodel(fea_dim, out_pt_fea_dim)
        self.ppmodel = PPmodel_all_preprocess(grid_size = [480, 480, 32], max_pt_per_encode = 256, kernal_size = 1,
                                              fea_compre = 32)
        # self.pyramidformer = PyramidVisionTransformerV2(patch_size = 4, embed_dims = [128, 320], num_heads = [2, 5],
        #                                                 mlp_ratios = [8, 4], qkv_bias = True,
        #                                                 norm_layer = partial(nn.LayerNorm, eps = 1e-5), depths = [2, 2],
        #                                                 sr_ratios = [8, 4, ])

        self.pyramidformer = pvt_v2_b0(pretrained = True)
        self.template_class = Template_class(512, 512)
        self.trans_dim_conv = conv_circular(256, 64)

        self.up1= up( 640, 256,circular_padding = True)  # in + x2.channel 512+128
        self.up2= up( 512, 256,scale_factor =  4)  # 恢复至transformer的输入尺寸
        # self.up3= up( 256, 128,scale_factor =  2)  # 恢复至1/2的输入尺寸
        self.up3= up( 288, 64,scale_factor =  4)  # 恢复至raw的输入尺寸  32+256
        self.dropout = nn.Dropout(p = 0. )

         # 因为所以合计降低维度一下，就64吧
        self.n_class = n_class *32
        self.outc = outconv(64, self.n_class)

    def forward(self, pt_fea_0, xy_ind_0, pt_fea_1, xy_ind_1):
        """
train_pt_fea_ten_0---- torch.Size([34720, 8])
train_grid_ten_0---- torch.Size([34720, 2])
        :param pt_fea_0:
        :param xy_ind_0:
        :param pt_fea_1: 1当前帧
        :param xy_ind_1:
        :return:
        """
        outdata_0 = self.ppmodel(pt_fea_0, xy_ind_0)  # Pointnet b,32,480,480
        outdata_1 = self.ppmodel(pt_fea_1, xy_ind_1)  # [1,64, 480, 480]
        raw_size= outdata_1

        torch.cuda.empty_cache()


        outdata_0 = self.bev_Unet_encoder(outdata_0)
        outdata_1 = self.bev_Unet_encoder(outdata_1)  # torch.Size([1, 256, 120, 120])
        res2 = outdata_1  # 保留用于后面concat 这是cnn2个阶段的意思，先down2次吧

        # 转换维度 ->64  要不然 transformer参数好多
        outdata_1 = self.trans_dim_conv(outdata_1)  # ([1, 64, 120, 120])
        outdata_0 = self.trans_dim_conv(outdata_0)
        torch.cuda.empty_cache()

        outdata_0= self.pyramidformer(outdata_0)  # 2个stage ，返回最后一个就是最后下采用的东西torch.Size([1, 320, 15, 15])
        outdata_1 = self.pyramidformer(outdata_1)  # outdata_1[0] 留着up的时候当恢复尺寸模板
        data_16_size=outdata_1[0]

        outdata_1 = self.template_class(outdata_0[1], outdata_1[1])  # 时序头 结合后的
        # 至此 前一帧 退出舞台
        # decoder
        out_data =self.up1( outdata_1,data_16_size)
        out_data =self.up2( out_data,res2 )
        out_data =self.up3( out_data,raw_size )
        out_data = self.outc(self.dropout( out_data))

        out_data = out_data.permute(0, 2, 3, 1)  # 1，480，360， 608
        new_shape = list(out_data.size())[:3] + [32, 2]
        out_data = out_data.view(new_shape)  # 1,480,360,32,19
        x = out_data.permute(0, 4, 1, 2, 3)

        return x


if __name__ == '__main__':
    model = PolarSegFormer(n_class = 2, grid_size = [480, 480, 32])
    data = [torch.randn(100, 2)]
    data_fea = [torch.randn(100, 8)]
    output = model(data_fea, data, data_fea, data)
    print(output.shape)
