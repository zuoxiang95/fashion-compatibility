#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: tripletnet_outfit.py
@time: 2020/4/28 17:25
@desc:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def accuracy(pos_samples, neg_samples):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    is_cuda = pos_samples.is_cuda
    margin = 0.6
    pred = (pos_samples - neg_samples + margin).cpu().data
    acc = (pred < 0).sum().float() / float(pos_samples.size()[0])
    acc = torch.from_numpy(np.array([acc], np.float32))
    if is_cuda:
        acc = acc.cuda()

    return Variable(acc)


class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet, margin):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.margin = margin
        self.criterion_triplet = torch.nn.TripletMarginLoss(margin=margin)
        self.criterion_ranking = torch.nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def calculate_multi_dist(a, b):
        dist = 0
        for i in a:
            dist = dist + F.pairwise_distance(i, b, 2)
        return dist/len(a)

    def calculate_ranking_loss(self, anchors, positive, negatives, method='avg'):
        dist_p = self.calculate_multi_dist(anchors, positive)
        dists_n = []
        for negative in negatives:
            distance = self.calculate_multi_dist(anchors, negative)
            dists_n.append(distance)
        if method == 'avg':
            dist_n = sum(dists_n) / len(dists_n)
        else:
            dist_n = min(dists_n)

        target = torch.FloatTensor(dist_p.size()).fill_(1)
        if dist_p.is_cuda:
            target = target.cuda()
        target = Variable(target)

        ranking_loss = self.criterion_ranking(dist_n, dist_p, target)
        accs = []
        for i in dists_n:
            acc = accuracy(dist_p, i)
            accs.append(acc)
        accs = torch.cat(accs)
        acc_avg = torch.mean(accs)
        return ranking_loss, acc_avg

    def forward(self, anchor_imgs, positive_img, negative_imgs):
        """ anchor_imgs: Anchors image,
            positive_img: Close (positive) image,
            negative_imgs: Distants (negative) image"""

        anchor_embeddings = []
        anchor_img_count = int(anchor_imgs.shape[1]/4)
        anchor_imgs = [anchor_imgs[:, i*4:(i+1)*4, :, :] for i in range(anchor_img_count)]
        for x in anchor_imgs:
            embedded_x = self.embeddingnet(x)
            anchor_embeddings.append(embedded_x)

        positive_embeddings = self.embeddingnet(positive_img)

        negative_embeddings = []
        negative_img_count = int(negative_imgs.shape[1]/4)
        negative_imgs = [negative_imgs[:, i*4:(i+1)*4, :, :] for i in range(negative_img_count)]
        for z in negative_imgs:
            embedded_z = self.embeddingnet(z)
            negative_embeddings.append(embedded_z)

        loss_triplet, acc = self.calculate_ranking_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        return acc, loss_triplet
