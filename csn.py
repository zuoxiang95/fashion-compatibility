#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: csn.py
@time: 2020/4/28 17:24
@desc:
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.num_conditions = n_conditions

        # mask branch
        self.downsample = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False),
                                        nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False),
                                        nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False),
                                        nn.Conv2d(1, 1, kernel_size=1, stride=2, bias=False),
                                        nn.Flatten(),
                                        nn.Linear(in_features=576, out_features=128)
                                        )

        # create the mask
        if learnedmask:
            if prein:
                # define masks
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7)  # 0.1, 0.005
        else:
            # define masks
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    @staticmethod
    def norm_feature(feature):
        norm = torch.norm(feature, p=2, dim=1) + 1e-10
        norm = torch.unsqueeze(norm, 1)
        norm_feature = feature / norm.expand_as(feature)
        return norm_feature

    def condition_forward(self, embedded_x, c):
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask
        norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
        norm = torch.unsqueeze(norm, 1)
        masked_embedding = masked_embedding / norm.expand_as(masked_embedding)
        return masked_embedding

    def forward(self, x):
        embedded_x = None
        general_x = self.embeddingnet(x[:, :3, :, :])

        for idx in range(self.num_conditions):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx = concept_idx + idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)
            tmp_embedded_x = self.condition_forward(general_x, concept_idx)

            if embedded_x is None:
                embedded_x = tmp_embedded_x
            else:
                embedded_x = embedded_x + tmp_embedded_x

        # masked mask
        masked_x = self.downsample(x[:, 3:, :, :])
        masked_x = self.norm_feature(masked_x)       # shape [1, 128]

        # add masked feature into embedding feature
        #embedded_x = embedded_x + masked_x
        #feature_x = embedded_x / (self.num_conditions+1)

        feature_x = torch.cat([embedded_x, masked_x], dim=1)    # shape [1, 512]

        return feature_x
