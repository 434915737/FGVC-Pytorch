import torch
import torch.nn as nn
import torch.nn.functional as F
# tensor=torch.ones(size=(2,1280,32,32))
# print(tensor)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

class SE_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(SE_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, 768, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(768, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.senet = SE_Block(in_planes=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # print('feature:',feature_cat.shape)
        seaspp1=self.senet(feature_cat)             #加入通道注意力机制
        # print('seaspp1:',seaspp1.shape)
        se_feature_cat=seaspp1*feature_cat
        result = self.conv_cat(se_feature_cat)
        # print('result:',result.shape)
        return result

# class SE_ASPP(nn.Module):
#     def __init__(self, dim_in=768, dim_out=768, rate=1, bn_mom=0.1):
#         super(SE_ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, 768, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(768, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#         self.senet = SE_Block(in_planes=dim_out * 5)
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         se_feature_cat = self.senet(feature_cat) * feature_cat
#         result = self.conv_cat(se_feature_cat)
#         return result