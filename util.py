#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: util.py
@time: 2020/4/28 17:26
@desc:
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import Resnet_101
from PIL import Image
from main import parser
from csn import ConditionalSimNet
from torchvision import transforms
from tripletnet_outfit import CS_Tripletnet

args = parser.parse_args()


def init_model(checkpoint_path):
    """
       load model from checkpoint file.
    :param checkpoint_path: the path of checkpoint file. (String)
    :return: the model object.
    """
    criterion = torch.nn.TripletMarginLoss(margin=args.margin)
    model = Resnet_101.resnet34(pretrained=True, embedding_size=args.dim_embed)
    csn_model = ConditionalSimNet(model, n_conditions=args.num_concepts,
                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)

    tnet = CS_Tripletnet(csn_model, criterion)
    checkpoint = torch.load(checkpoint_path)
    tnet.load_state_dict(checkpoint['state_dict'])
    tnet.eval()
    tnet.cuda()

    return tnet


cloth_match_model_women = init_model("./runs/test_20/checkpoint_16.pth.tar")
cloth_match_model_men = init_model("./runs/test_20/checkpoint_16.pth.tar")
print("Load model and data successful!")


def load_image(image_path):
    """
       load image and transform image to pytorch Tensor.
    :param image_path: the path of image.(String)
    :return: Transformed image tensor.
    """
    image = Image.open(image_path)
    # RGBA to RGB
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # image normalization
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image
