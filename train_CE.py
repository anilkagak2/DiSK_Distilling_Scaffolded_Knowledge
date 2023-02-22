

import os, sys, time, torch, random, argparse, json
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from model_dict import get_model_from_name
from utils import get_model_infos
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time
from starts import prepare_logger
from get_dataset_with_transform import get_datasets

from DiSK import obtain_accuracy, get_mlr, save_checkpoint, evaluate_model

def m__get_prefix( args ):
    prefix = 'disk-CE-' + args.dataset + '-' + args.model_name + '-' 
    return prefix

def get_model_prefix( args ):
    prefix = os.path.join(args.save_dir, m__get_prefix( args ) )
    return prefix

def cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, xloader, criterion, batch_size, mode='eval' ):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if mode == 'eval': 
        network.eval()
    else:
        network.train()

    progress = ProgressMeter(
            logger,
            len(xloader),
            [losses, top1, top5],
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets) in enumerate(xloader):
        if mode == 'train':
            optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        _, logits, _ = network(inputs)

        loss = criterion(logits, targets)
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        if mode == 'train':
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'train':
            scheduler.step(epoch)

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg

def main(args):
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    criterion = nn.CrossEntropyLoss()

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    args.class_num = class_num
    logger = prepare_logger(args)

    Arguments = namedtuple("Configure", ('class_num','dataset')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset }
    model_config = Arguments(**md_dict)

    base_model = get_model_from_name( model_config, args.model_name )
    model_name = args.model_name

    base_model = base_model.cuda()
    network = base_model

    optimizer = torch.optim.SGD(base_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    flop, param = get_model_infos(base_model, xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(valid_data))

    best_acc = 0.0
    for epoch in range(args.epochs):
        trn_loss, trn_acc1, trn_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, train_loader, criterion, args.batch_size, mode='train' )
        val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, valid_loader, criterion, args.eval_batch_size, mode='eval' )
        is_best = False 
        if val_acc1 > best_acc:
            best_acc = val_acc1
            is_best = True
        save_checkpoint({
                'epoch': epoch + 1,
                'base_state_dict': base_model.state_dict(),
                'best_acc': best_acc,
                'scheduler' : scheduler.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best, prefix=get_model_prefix( args ))


        logger.log('\t\t LR=' + str(get_mlr(scheduler)) + ' -- best acc so far ' + str( best_acc )  )


    valid_loss, valid_acc1, valid_acc5 = evaluate_model( network, valid_loader, criterion, args.eval_batch_size )
    logger.log(
        "***{:s}*** [Post-train] [Student] EVALUATION loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            valid_loss,
            valid_acc1,
            valid_acc5,
            100 - valid_acc1,
            100 - valid_acc5,
        )
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name", type=str, 
        default='ResNet32TwoPFiveM-NAS',
        help="The path to the model configuration"
    )
    
    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument(
        "--cutout_length", type=int, default=16, help="The cutout length, negative means not use."
    )

    # Printing
    parser.add_argument(
        "--print_freq", type=int, default=100, help="print frequency (default: 200)"
    )
    parser.add_argument(
        "--print_freq_eval",
        type=int,
        default=100,
        help="print frequency (default: 200)",
    )
    # Checkpoints
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1,
        help="evaluation frequency (default: 200)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log.",
        default='./logs/',
    )
    # Acceleration
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of data loading workers (default: 8)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=2007, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)

    # Optimization options
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=200, help="Batch size for training."
    )

    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint',
                        help='checkpoint file format')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"

    main(args)






