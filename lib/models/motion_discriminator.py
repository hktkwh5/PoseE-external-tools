# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from lib.models.attention import SelfAttention

class MotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size,
                 input_size,
                 num_layers,
                 output_size=2,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        # layer1 = nn.Sequential()  # 时序容器
        # layer1.add_module('conv1', nn.Conv1d(1, 8, 3, 1, padding=1))
        # layer1.add_module('BN1', nn.BatchNorm1d(8))
        # layer1.add_module('relu1', nn.ReLU(True))
        # layer1.add_module('pool1', nn.MaxPool1d(2, 2))
        #
        # layer1.add_module('conv2', nn.Conv1d(8, 64, 3, 1, padding=1))
        # layer1.add_module('BN2', nn.BatchNorm1d(64))
        # layer1.add_module('relu2', nn.ReLU(True))
        # layer1.add_module('pool2', nn.MaxPool1d(2, 2))
        #
        # layer1.add_module('conv3', nn.Conv1d(64, 256, 3, 1, padding=1))
        # layer1.add_module('BN3', nn.BatchNorm1d(256))
        # layer1.add_module('relu3', nn.ReLU(True))
        # layer1.add_module('pool3', nn.MaxPool1d(2, 2))
        #
        # layer1.add_module('conv4', nn.Conv1d(256, 256, 3, 1, padding=1))
        # layer1.add_module('BN4', nn.BatchNorm1d(256))
        # layer1.add_module('relu4', nn.ReLU(True))
        # layer1.add_module('pool4', nn.MaxPool1d(2, 2))
        #
        # layer1.add_module('conv5', nn.Conv1d(256, 32, 3, 1, padding=1))
        # layer1.add_module('BN5', nn.BatchNorm1d(32))
        # layer1.add_module('relu5', nn.ReLU(True))
        # layer1.add_module('pool5', nn.MaxPool1d(2, 2))
        #
        # layer1.add_module('conv6', nn.Conv1d(32, 1, 3, 1, padding=1))
        # layer1.add_module('BN6', nn.BatchNorm1d(1))
        # layer1.add_module('relu6', nn.ReLU(True))
        # layer1.add_module('pool6', nn.MaxPool1d(2, 2))
        # self.layer1 = layer1

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        # sequence = torch.transpose(sequence, 0, 1)
        # sequence = sequence.reshape(sequence.shape[0] * sequence.shape[1],1, sequence.shape[2])

        outputs, state = self.gru(sequence)
        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])
        return output



class MotionDiscriminator_cross_eye(nn.Module):

    def __init__(self,
                 rnn_size,
                 input_size,
                 num_layers,
                 output_size=2,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator_cross_eye, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        # self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        layer1 = nn.Sequential()  # 时序容器
        layer1.add_module('conv1', nn.Conv1d(1, 8, 3, 1, padding=1))
        layer1.add_module('BN1', nn.BatchNorm1d(8))
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool1d(2, 2))

        layer1.add_module('conv2', nn.Conv1d(8, 64, 3, 1, padding=1))
        layer1.add_module('BN2', nn.BatchNorm1d(64))
        layer1.add_module('relu2', nn.ReLU(True))
        layer1.add_module('pool2', nn.MaxPool1d(2, 2))

        layer1.add_module('conv3', nn.Conv1d(64, 256, 3, 1, padding=1))
        layer1.add_module('BN3', nn.BatchNorm1d(256))
        layer1.add_module('relu3', nn.ReLU(True))
        layer1.add_module('pool3', nn.MaxPool1d(2, 2))

        layer1.add_module('conv4', nn.Conv1d(256, 512, 3, 1, padding=1))
        layer1.add_module('BN4', nn.BatchNorm1d(512))
        layer1.add_module('relu4', nn.ReLU(True))
        layer1.add_module('pool4', nn.MaxPool1d(2, 2))

        layer1.add_module('conv5', nn.Conv1d(512, 1024, 3, 1, padding=1))
        layer1.add_module('BN5', nn.BatchNorm1d(1024))
        layer1.add_module('relu5', nn.ReLU(True))
        layer1.add_module('pool5', nn.MaxPool1d(2, 2))

        layer1.add_module('conv6', nn.Conv1d(1024, 2048, 3, 1, padding=1))
        layer1.add_module('BN6', nn.BatchNorm1d(2048))
        layer1.add_module('relu6', nn.ReLU(True))
        layer1.add_module('pool6', nn.MaxPool1d(2, 2))

        self.layer1 = layer1
        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        # sequence = torch.transpose(sequence, 0, 1)
        sequence = sequence.reshape(sequence.shape[0] * sequence.shape[1], 1, sequence.shape[2])
        # outputs, state = self.gru(sequence)
        outputs = self.layer1(sequence)
        outputs = outputs.reshape(outputs.size(0), outputs.size(1))
        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            # outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = y.reshape(y.size(0),1)
            # output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return output