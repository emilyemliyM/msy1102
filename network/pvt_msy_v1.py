import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'pvt_tiny',
]
# 单纯的一层conv
class conv_circular_layer(nn.Module):
    '''conv => BN => ReLU'''  # in 32, out_ch 64 ,group_conv False

    def __init__(self, in_ch = 32, out_ch = 64,kernel_size=3 ,stride=1, groups=1, dilation = 1):
        super(conv_circular_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride,padding = (1, 0), groups = 1),
        )

    def forward(self, x):  # torch.Size([1, 32, 480, 360])
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode = 'circular')  # 1,32,480,362
        x = self.conv1(x)  # torch.Size([1, 64, 480, 360])
        return x
# 版本1代码，容易理解

class DWConv(nn.Module):
    def __init__(self, dim=768, groups= True):
        super(DWConv, self).__init__()

        # self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv= conv_circular_layer( in_ch = dim,out_ch = dim,groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#         self.linear = linear
#         if self.linear:
#             self.relu = nn.ReLU(inplace=True)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         if self.linear:
#             x = self.relu(x)
#         x = self.dwconv(x, H, W) # 它的创新是深度可分离卷积
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

#
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = conv_circular_layer(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)   # 根据sr_ratio[8 4 2 1]再次切分重排
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        # x [B ,3136,64]
        B, N, C = x.shape #N 3136 C 64
        # B,N, head个数，head的维度 -> B,head个数,N, head 维度
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, 1, 3136,64

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # torch.Size([2, 1, 1, 49, 64])
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # torch.Size([1, 1, 49, 64])  # V * softmax(q .* k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # torch.Size([1, 1, 3136, 49])
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# attention+ norm+ mlp
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # 跳连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 跳连

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    能够这样写的原因是MLP和Conv2d的计算过程很像。
具体而言，MLP一般处理一维vector，方式为矩阵乘法，也就是n个点积，点积是先乘再加。Conv2d处理二维，方式为卷积，也就是随着窗口滑动的先乘再加。
MLP和Conv2d最重要的两点区别为局部连接、权重共享。
可以发现在图像切分重排这个过程中，所有的操作都在每个patch内(局部链接)，而且对于每个patch的操作是相同的(权重共享)，因此可以用conv替代。

    """
    def __init__(self, img_size=120, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # ，而且对于每个patch的操作是相同的(权重共享)，因此可以用conv替代 没有重叠的情况下就是mlp，所以这么设置卷积核和步长
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape # C :3 H:56, W:56 patch_size(4, 4) ,num_patch:3136

        x = self.proj(x).flatten(2).transpose(1, 2)  # 切分重排
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=120, patch_size=16, in_chans=32, num_classes=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
        #                                embed_dim=embed_dims[2])
        # self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
        #                                embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        self.norm = norm_layer(embed_dims[1])
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        # stage 1
        #图像切分重排
        x, (H, W) = self.patch_embed1(x)
        #给每个patch，安排上patch embeding
        x = x + self.pos_embed1
        #dropout
        x = self.pos_drop1(x)
        #进transformer encoder
        for blk in self.block1:
            x = blk(x, H, W)
        #[1, 900, 64]) reshape成2d的样子进入下一个stage
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(" stae1: x----",x.shape)

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2  # torch.Size([1, 64, 56, 56])
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)  # torch.Size([1, 784, 128])  H 28  ,W 28
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # torch.Size([1, 128, 28, 28])
        # print(" stae2: x----", x.shape)

        # # stage 3
        # x, (H, W) = self.patch_embed3(x)
        # x = x + self.pos_embed3
        # x = self.pos_drop3(x)
        # for blk in self.block3:
        #     x = blk(x, H, W)  # torch.Size([1, 196, 320])
        # print(" block输出 stae3: x----", x.shape)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # torch.Size([1, 320, 14, 14])
        # # print(" stae3: x----", x.shape)

        # # stage 4
        # x, (H, W) = self.patch_embed4(x)
        # cls_tokens = self.cls_token.expand(B, -1, -1)   # torch.Size([1, 1, 512]) # patch 0
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed4
        # x = self.pos_drop4(x)
        # for blk in self.block4:
        #     x = blk(x, H, W)
        # print(" block输出 stae4: x----", x.shape)
        # x = self.norm(x)  # torch.Size([1, 50, 512])
        # # print(" stae4: x----", x.shape)
        return x # 只要token 0

    def forward(self, x):
        x = self.forward_features(x)  # torch.Size([1, 512])
        # x = self.head(x)  # mlp torch.Size([1, 1000])

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

#
# @register_model
# def pvt_tiny(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#

@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128,], num_heads=[1, 2], mlp_ratios=[8, 8], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-5), depths=[2, 2], sr_ratios=[8, 4],
        **kwargs)
    model.default_cfg = _cfg()

    return model


"""
    stae1: x---- torch.Size([1, 64, 56, 56])  224 224      
    stae2: x---- torch.Size([1, 128, 28, 28])
    stae3: x---- torch.Size([1, 320, 14, 14])
    stae4: x---- torch.Size([1, 50, 512])

 没有reshape encoder的block输出 stae1: x---- torch.Size([1, 3136, 64])
 block输出 stae2: x---- torch.Size([1, 784, 128])
 block输出 stae3: x---- torch.Size([1, 196, 320])
 block输出 stae4: x---- torch.Size([1, 50, 512])
"""

if __name__== "__main__":
    model = pvt_tiny()
    img = torch.randn(1, 32, 120, 120)
    pred = model(img)
    print( pred.shape)
