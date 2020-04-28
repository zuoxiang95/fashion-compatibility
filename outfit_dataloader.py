#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: outfit_dataloader.py
@time: 2020/4/28 17:25
@desc:
"""

import json
import torch
import numpy as np
from copy import deepcopy
from PIL import Image
from os.path import join


def default_image_loader(path):
    im = Image.open(path)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    return im


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, is_train, split=0.99, transform=None, loader=default_image_loader, neg_num=1):
        self.is_train = is_train
        self.loader = loader
        self.transform = transform
        self.neg_num = neg_num
        self.image_dir = join(data_dir, 'image')
        self.label_path = join(data_dir, '4and5_items_train.json')
        outfits_json, category2id, id2category = self.get_data(self.label_path)
        self.category2id = category2id
        self.id2category = id2category

        pos_pairs = []
        for outfit_id, items_list in outfits_json.items():
            for item_id in items_list:
                anchor_list = deepcopy(items_list)
                anchor_list.remove(item_id)
                pos_pairs.append([anchor_list, item_id])

        split_num = int(len(pos_pairs) * split)
        if is_train:
            self.pos_pairs_train = pos_pairs[:split_num]
        else:
            self.pos_pairs_valid = pos_pairs[split_num:]

    def sample_negative(self, item_id, item_type, neg_num):
        item_out = item_id
        candidate_sets = self.category2id[item_type]
        count = 0
        negative_items = []
        while count < neg_num:
            negative_index = np.random.choice(range(len(candidate_sets)))
            item_out = candidate_sets[negative_index]
            if item_out == item_id:
                continue
            negative_items.append(item_out)
            count += 1
        return negative_items

    @staticmethod
    def get_data(file_path):
        outfits_json = dict()
        category_id_dict = dict(tops=[],
                                bottoms=[],
                                shoes=[],
                                outerwear=[],
                                all_body=[],
                                bags=[],
                                jewellery=[],
                                accessories=[],
                                hats=[])
        id_category_dict = dict()
        with open(file_path, 'r') as f1:
            outfits_info = json.load(f1)
        for outfit_info in outfits_info:
            outfit_id = outfit_info["set_id"]
            outfits_json[outfit_id] = []
            for item_info in outfit_info["items"]:
                item_id = item_info["item_id"]
                item_type = item_info["category"]
                outfits_json[outfit_id].append(item_id)
                category_id_dict[item_type].append(item_id)
                id_category_dict[item_id] = item_type

        return outfits_json, category_id_dict, id_category_dict

    def load_image(self, image_id):
        img_path = join(self.image_dir, '{}.png'.format(image_id))
        img = self.loader(img_path)
        if len(img.split()) != 4:
            print(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        if self.is_train:
            anchors_id, positive_id = self.pos_pairs_train[index]
        else:
            anchors_id, positive_id = self.pos_pairs_valid[index]

        positive_type = self.id2category[positive_id]
        negatives_id = self.sample_negative(positive_id, positive_type, neg_num=self.neg_num)

        anchors_img = [self.load_image(anchor_id) for anchor_id in anchors_id]
        anchors_img = torch.cat(anchors_img)

        positive_img = self.load_image(positive_id)
        negatives_img = [self.load_image(negative_id) for negative_id in negatives_id]
        negatives_img = torch.cat(negatives_img)
        return anchors_img, positive_img, negatives_img

    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs_train)

        return len(self.pos_pairs_valid)
