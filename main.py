#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: ZuoXiang
@contact: zx_data@126.com
@file: main.py
@time: 2020/4/28 17:28
@desc:
"""

# -*- coding: utf-8 -*-

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
import sys
import shutil
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from outfit_dataloader import TripletImageLoader
from tripletnet_outfit import CS_Tripletnet
from torch.nn.parallel import DistributedDataParallel
from visdom import Visdom
import numpy as np
import Resnet_101
from csn import ConditionalSimNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=114, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--distribute', type=bool, default=True,
                    help='enables multi gpu training')
parser.add_argument('--gpus', type=str, default='0',
                    help='which gpu will be uesd in training')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.6, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='test_20', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--sim_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for similarity norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--dim_embed', type=int, default=384, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
parser.add_argument('--num_concepts', type=int, default=5, metavar='N',
                    help='number of random embeddings when rand_typespaces=True')
parser.add_argument('--local_rank', type=int, default=0)
parser.set_defaults(test=False)
parser.set_defaults(learned=True)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.distribute:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    global conditions
    if args.conditions is not None:
        conditions = args.conditions
    else:
        conditions = [0, 1, 2, 3]

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    print('Loading Train Dataset')
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('/home/zuoxiang/Outfit-notext/data/dida_outfits', True,
                           transform=transforms.Compose([
                               transforms.Scale(384),
                               transforms.CenterCrop(384),
                               transforms.ToTensor(),
                               # normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print('Loading Test Dataset')
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader('/home/zuoxiang/Outfit-notext/data/dida_outfits', False,
                           transform=transforms.Compose([
                               transforms.Scale(384),
                               transforms.CenterCrop(384),
                               transforms.ToTensor(),
                               # normalize,
                           ])),
        batch_size=128, shuffle=True, **kwargs)

    model = Resnet_101.resnet34(pretrained=True, embedding_size=args.dim_embed)
    csn_model = ConditionalSimNet(model, n_conditions=args.num_concepts,
                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)

    tnet = CS_Tripletnet(csn_model, args.margin)
    tnet.cuda()
    if args.distribute:
        tnet = DistributedDataParallel(tnet)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        checkpoint = torch.load('runs/%s/' % ('new_context_4/') + 'model_best.pth.tar')
        tnet.load_state_dict(checkpoint['state_dict'])
        test_acc = test(test_loader, tnet, 1)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, tnet, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, tnet, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if args.distribute:
            checkpoint_state = tnet.module.state_dict()
        else:
            checkpoint_state = tnet.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': checkpoint_state,
            'best_prec1': best_acc,
        }, epoch, is_best)

    checkpoint = torch.load('runs/%s/' % (args.name) + 'model_best.pth.tar')
    tnet.load_state_dict(checkpoint['state_dict'])
    test_acc = test(test_loader, tnet, 1)


def train(train_loader, tnet, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        acc, loss_triplet = tnet(data1, data2, data3)

        # loss
        loss = loss_triplet.mean()
        acc = acc.mean()

        # measure accuracy and record loss
        losses.update(loss.item(), data1.size(0))
        accs.update(acc.item(), data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * accs.val, 100. * accs.avg))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)


def test(test_loader, tnet, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = {}
    for condition in conditions:
        accs_cs[condition] = AverageMeter()

    # switch to evaluation mode
    tnet = tnet.module
    tnet.eval()
    tnet.embeddingnet.eval()
    tnet.embeddingnet.embeddingnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

            # compute output
            acc, loss_triplet = tnet(data1, data2, data3)
            test_loss = loss_triplet

            # measure accuracy and record loss
            accs.update(acc.item(), data1.size(0))
            losses.update(test_loss, data1.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    if args.visdom:
        for condition in conditions:
            plotter.plot('accs', 'acc_{}'.format(condition), epoch, accs_cs[condition].avg)
        plotter.plot(args.name, args.name, epoch, accs.avg, env='overview')
        plotter.plot('acc', 'test', epoch, accs.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg


def save_checkpoint(state, epoch, is_best):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'checkpoint_{}.pth.tar'.format(epoch)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % args.name + 'model_best.pth.tar')


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name],
                                 name=split_name)

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
