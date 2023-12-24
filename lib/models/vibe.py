# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y

class singe_view_AGGRENET(nn.Module):
    def __init__(
            self,
            batch_size=128,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(singe_view_AGGRENET, self).__init__()

        layer1 = nn.Sequential()  # 时序容器
        layer1.add_module('conv1', nn.Conv1d(3, 32, 3, 1, padding=1))
        layer1.add_module('BN1', nn.BatchNorm1d(32))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool1d(2, 2))
        self.layer1 = layer1
        layer2 = nn.Sequential()  # 时序容器
        layer2.add_module('conv2', nn.Conv1d(32, 64, 3, 1, padding=1))
        layer2.add_module('BN2', nn.BatchNorm1d(64))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool1d(2, 2))
        self.layer2 = layer2
        layer3 = nn.Sequential()  # 时序容器
        layer3.add_module('conv3', nn.Conv1d(64, 128, 3, 1, padding=1))
        layer3.add_module('BN3', nn.BatchNorm1d(128))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool1d(2, 2))
        self.layer3 = layer3
        layer4 = nn.Sequential()  # 时序容器
        layer4.add_module('conv4', nn.Conv1d(128, 256, 3, 1, padding=1))
        layer4.add_module('BN4', nn.BatchNorm1d(256))
        layer4.add_module('relu4', nn.ReLU(True))
        layer4.add_module('pool4', nn.MaxPool1d(2, 2))
        self.layer4 = layer4
        layer5 = nn.Sequential()  # 时序容器
        layer5.add_module('conv5', nn.Conv1d(256, 512, 3, 1, padding=1))
        layer5.add_module('BN5', nn.BatchNorm1d(512))
        layer5.add_module('relu5', nn.ReLU(True))
        layer5.add_module('pool5', nn.MaxPool1d(2, 2))
        self.layer5 = layer5
        layer6 = nn.Sequential()  # 时序容器
        layer6.add_module('conv6', nn.Conv1d(512, 1024, 3, 1, padding=1))
        layer6.add_module('BN6', nn.BatchNorm1d(1024))
        layer6.add_module('relu6', nn.ReLU(True))
        layer6.add_module('pool6', nn.MaxPool1d(2, 2))
        self.layer6 = layer6
        layer7 = nn.Sequential()  # 时序容器
        layer7.add_module('conv7', nn.Conv1d(1024, 1024, 3, 1, padding=1))
        layer7.add_module('BN7', nn.BatchNorm1d(1024))
        layer7.add_module('relu7', nn.ReLU(True))
        layer7.add_module('pool7', nn.MaxPool1d(2, 2))
        self.layer7 = layer7
        layer8 = nn.Sequential()  # 时序容器
        layer8.add_module('conv8', nn.Conv1d(1024, 2048, 3, 1, padding=1))
        layer8.add_module('BN8', nn.BatchNorm1d(2048))
        layer8.add_module('relu8', nn.ReLU(True))
        layer8.add_module('pool8', nn.MaxPool1d(4, 4))
        self.layer8 = layer8
        layer9 = nn.Sequential()  # 时序容器
        layer9.add_module('conv9', nn.Conv1d(2048, 2048, 3, 1, padding=1))
        layer9.add_module('BN9', nn.BatchNorm1d(2048))
        layer9.add_module('relu9', nn.ReLU(True))
        layer9.add_module('pool9', nn.MaxPool1d(4, 4))
        self.layer9 = layer9

    def forward(self, x, input_left, input_right):
        input = torch.cat((input_left, x), dim=1)
        input = torch.cat((input, input_right), dim=1)
        conv = self.layer1(input)
        conv = self.layer2(conv)
        conv = self.layer3(conv)
        conv = self.layer4(conv)
        conv = self.layer5(conv)
        conv = self.layer6(conv)
        conv = self.layer7(conv)
        conv = self.layer8(conv)
        conv1 = self.layer9(conv)
        conv1 = conv1.reshape(conv1.size(0), conv1.size(1))
        return conv1


class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE, self).__init__()
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        # feature = self.encoder(input)
        feature = input
        feature = feature.reshape(-1, feature.size(-1))#

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
        return smpl_output


class VIBE_cross_eye(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE_cross_eye, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = singe_view_AGGRENET(
            batch_size=self.batch_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, input_left, input_right, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        input=input.reshape(input.shape[0]*input.shape[1],1,input.shape[2])
        input_left = input_left.reshape(input_left.shape[0] * input_left.shape[1],1, input_left.shape[2])
        input_right = input_right.reshape(input_right.shape[0] * input_right.shape[1],1, input_right.shape[2])
        feature = self.encoder(input, input_left, input_right)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
        return smpl_output


class VIBE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        feature = feature.reshape(batch_size, seqlen, -1)
        feature = self.encoder(feature)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output
