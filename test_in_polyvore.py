#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: test_in_polyvore.py
@time: 2020/4/28 17:27
@desc:
"""

import json
import torch
import torch.nn.functional as F
from os.path import join
from util import init_model, load_image


def load_polyvore_data(polyvore_data_path):
    with open(polyvore_data_path, 'r') as f1:
        polyvore_json_data = json.load(f1)
    return polyvore_json_data


def load_image_feature(model, image_name):
    """

    :param model:
    :param image_name:
    :return:
    """
    image_path = join('/home/zuoxiang/Outfit-notext/data/dida_outfits/image', image_name)
    image = load_image(image_path)
    image = torch.unsqueeze(image, dim=0)
    with torch.no_grad():
        image_embedding = model.embeddingnet(image.cuda())[0]
    return image_embedding


def calculate_distance(question_imgs_feature, answer_img_feature):
    scores_list = []
    answer_img_feature = torch.unsqueeze(answer_img_feature, dim=0)
    for question_img_feature in question_imgs_feature:
        question_img_feature = torch.unsqueeze(question_img_feature, dim=0)

        score = F.pairwise_distance(answer_img_feature, question_img_feature)
        scores_list.append(score)
    final_score = float(sum(scores_list)) / float(len(scores_list))
    print(final_score)
    return final_score


def test_accuracy(data_path, checkpoint_path):
    model = init_model(checkpoint_path)
    polyvore_json_data = load_polyvore_data(data_path)
    true_count = 0
    error_count = 0
    for question_json in polyvore_json_data[:2000]:
        try:
            question_imgs = question_json["question"]
            question_imgs_feature = []
            for question_img_name in question_imgs:
                question_img_name = question_img_name.split('_')[0] + '.png'
                question_img_feature = load_image_feature(model, question_img_name)
                question_imgs_feature.append(question_img_feature)

            answer_imgs = question_json["answers"]
            answer_imgs_score = []
            for answer_img_name in answer_imgs:
                answer_img_name = answer_img_name.split('_')[0] + '.png'
                answer_imgs_feature = load_image_feature(model, answer_img_name)
                answer_img_score = calculate_distance(question_imgs_feature, answer_imgs_feature)
                answer_imgs_score.append(answer_img_score)
            if min(answer_imgs_score) == answer_imgs_score[0]:
                true_count += 1
            print(true_count)
        except Exception as e:
            error_count += 1
            continue
    # print("Test accuracy: {}".format(float(true_count)/float(len(polyvore_json_data))))
    print("Test accuracy: {}".format(float(true_count) / float(2000 - error_count)))


if __name__ == '__main__':
    test_accuracy(r'/home/zuoxiang/Outfit-notext/data/dida_outfits/fill_in_blank_test.json',
                  r'/home/zuoxiang/learning-Similarity-Conditions/runs/test_20/checkpoint_21.pth.tar')