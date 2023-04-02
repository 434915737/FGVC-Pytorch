"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import convnext
import models.vgg as vgg
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d
#from timm.models.layers import CbamBlock
from models.baseline import SE_ASPP
from models.baseline_2 import CBAM_ASPP
from models.FasterNet import FasterNet
import models.iResnet as iResnet
from models.xception import AlignedXception
from models.conv_res import *
from models.cal import CALoss, cal_attention
from models.FcaNet import *
import config
from models.case_res import caresnet50,CaSE
import timm
__all__ = ['WSDAN_CAL']
EPSILON = 1e-12






#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, reduction_ratio)
#         self.sa = SpatialAttention()
#
#     def forward(self, x):
#         out = self.ca(x) * x
#         out = self.sa(out) * out
#         return out

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature

class ActionDecisionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ActionDecisionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)
        return x * y.expand_as(x)


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN_CAL(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN_CAL, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net



        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'convnext' in net:
            self.features = getattr(convnext, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'fasternet' in net:  # Add a condition for FasterNet
            self.features = FasterNet(pretrained=None, fork_feat=False, num_classes=200).get_features()
            self.num_features = 1280
        elif 'iResnet' in net:
            self.features = getattr(iResnet, net)(pretrained=pretrained).get_features()
            #self.num_features = 512 * self.features[-1][-1].expansion
            self.num_features=2048
        elif 'xception' in net:
            self.features = AlignedXception(pretrained=pretrained,BatchNorm=nn.BatchNorm2d,output_stride=16).get_features()
            self.num_features = 2048
        elif 'conv_res' in net:
            self.features = resnet50(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        elif 'fcanet' in net:
            self.features = fcanet50(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'case' in net:
            self.features = caresnet50(pretrained=pretrained,adaptive_layer=CaSE).get_features()
            #self.num_features = 512 * self.features[-1][-1].expansion
            self.num_features=2048
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        #self.ad_block = ActionDecisionBlock(self.num_features, self.num_features)
        #self.cbam = CBAM(self.num_features)
        # Adding the SE-ASPP module
        #self.se_aspp = SE_ASPP(self.num_features, 256)
        #self.cbam_aspp=CBAM_ASPP(self.num_features,256)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        self.cal_lambda = config.cal_lambda
        self.cal_loss = CALoss()

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return p, p - self.fc(feature_matrix_hat * 100.), feature_matrix, attention_map




    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN_CAL, self).load_state_dict(model_dict)
