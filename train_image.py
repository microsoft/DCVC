# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import sys
import time
import torch
import torch.distributed as dist

from torch.nn.utils import clip_grad_norm_

from src.datasets.image_dataset import ImageFolder
from src.models.image_model import DMCI
from src.utils.common import start_train, save_ckpt, save_status, get_current_device, \
    get_training_lambdas, get_dataloader, init_train, cleanup_train, load_existing_weights, wrap_ddp


def get_training_strategy():
    # epoch is for referencing purpose
    # lr is the learning rate for the current epoch
    training_strategy = \
        [[0,   2e-4, 256, 256]] * 45 + \
        [[49,  5e-5, 256, 256]] * 25 + \
        [[69,  1e-5, 256, 256]] * 20 + \
        [[90,  2e-4, 512, 512]] * 5 + \
        [[95,  5e-5, 512, 512]] * 4 + \
        [[99,  1e-5, 512, 512]] * 4 + \
        [[103, 1e-6, 512, 512]] * 2 + \
        [[105, 1e-6, 512, 512]]  # noqa: E501 E221
        # epo, lr,   patch_size  # noqa: E116 E501

    return training_strategy


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epochs', default=104, type=int)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Dataloaders worker per trainer')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models')
    parser.add_argument('--train_dataset', type=str, required=True)

    args = parser.parse_args(argv)
    return args


def train_one_epoch(i_net, dataloader, optimizer, epoch, rank):
    training_strategy = get_training_strategy()
    i_net.train()
    device = next(i_net.parameters()).device

    idx = min(len(training_strategy) - 1, epoch)
    _, lr, patch_width, patch_height = training_strategy[idx]

    for g in optimizer.param_groups:
        g['lr'] = lr

    world_size = dist.get_world_size() if rank >= 0 else 1
    dataloader.dataset.set_patch_size(patch_width, patch_height)

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        batch = [t.to(device) for t in batch]
        batch_size = batch[0].size(0)
        qp = batch[-2]
        curr_lambdas = batch[-1]

        print_loss_info = rank <= 0 and i % 200 == 0

        loss, info = i_net(batch[0], qp, lambdas=curr_lambdas, get_loss_info=print_loss_info)
        optimizer.zero_grad()
        loss.backward()

        total_norm = clip_grad_norm_(i_net.parameters(), max_norm=0.1,
                                     error_if_nonfinite=False).item()
        if math.isnan(total_norm) or math.isinf(total_norm):
            if rank <= 0:
                print('non-finite norm, skip this batch')
            continue

        optimizer.step()

        if print_loss_info:
            numer = i * batch_size * world_size
            denom = len(dataloader.dataset)
            t1 = time.time()
            print(
                f'Time: {t1-t0:.3f} seconds, Train epoch {epoch}:'
                f' [{numer} / {denom} ({100. * numer / denom:.0f}%)]'
                f' MSE: {info['mse']} | Bpp y: {info['bpp_y']} | Bpp z: {info['bpp_z']} |'
                f' Losses: {info['losses']} | lr: {optimizer.param_groups[0]['lr']:.1e}'
            )
            t0 = t1


def train(rank, args):
    world_size = init_train(rank, args.save_dir)
    train_dataset = ImageFolder(args.train_dataset, 256, 256, DMCI.qp_num(),
                                get_training_lambdas(args.lambdas, DMCI.qp_num()))

    i_net = DMCI()
    begin_epoch, opt_status = load_existing_weights(args.save_dir, i_net, rank)

    device = get_current_device(rank)
    if rank >= 0:
        i_net = i_net.cuda(rank)
    else:
        i_net = i_net.to(device)
    i_net = wrap_ddp(i_net, rank)

    optimizer = torch.optim.AdamW(i_net.parameters(), lr=1e-4)
    if opt_status is not None:
        optimizer.load_state_dict(opt_status)

    dataloader = get_dataloader(train_dataset, rank, world_size, args.batch_size,
                                args.num_workers)
    for epoch in range(begin_epoch, args.epochs):
        if rank >= 0:
            dataloader.sampler.set_epoch(epoch)
        train_one_epoch(i_net, dataloader, optimizer, epoch, rank)

        if rank <= 0:
            save_status(args.save_dir, i_net, optimizer, epoch)

    if rank <= 0:
        save_ckpt(args.save_dir, i_net)
    cleanup_train(rank)


def main(argv):
    args = parse_args(argv)
    start_train(train, args)


if __name__ == '__main__':
    main(sys.argv[1:])
