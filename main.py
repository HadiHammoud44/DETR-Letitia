# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=40000, type=int)
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--clip_max_norm', default=100, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--drop_last', action='store_true',
                        help='Drop the last incomplete batch if its size is smaller than the batch size')
    parser.add_argument('--NO_schedule_sampling', action='store_false', dest='use_schedule_sampling',
                        help='Disable scheduled sampling during training')

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=64, type=int,
                        help="Number of query slots")
    parser.add_argument('--original_feature_size', default=69, type=int,
                        help="Size of the original radiomics feature vector per point (3D coords + radiomics + empty_pt + superclass)")
    parser.add_argument('--num_superclasses', default=2, type=int,
                        help="Number of superclasses (excluding no_object class)")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_superclass', default=1, type=float,
                        help="Superclass coefficient in the matching cost")
    parser.add_argument('--set_cost_coordinates', default=5, type=float,
                        help="L2 coordinates coefficient in the matching cost")
    parser.add_argument('--set_cost_radiomics', default=0.0, type=float,
                        help="L2 radiomics coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--superclass_loss_coef', default=1, type=float,
                        help="Superclass classification coefficient in the loss")
    parser.add_argument('--radiomics_loss_coef', default=3, type=float)
    parser.add_argument('--coordinates_loss_coef', default=5, type=float)
    parser.add_argument('--superclass_coef', default=[0.75, 2.0, 0.15], nargs='+', type=float,
                        help="Superclass weights to handle class imbalance (list of floats of size num_superclasses + 1 including no-object)")

    # dataset parameters
    parser.add_argument('--dataset_file', default='letitia')
    parser.add_argument('--data_root', type=str, help='Root directory containing TP0, TP1, TP2 folders')
    parser.add_argument('--split_ratio', default=(0.8, 0.15, 0.05), nargs=3, type=float, help='Train/val/test split ratios')
    parser.add_argument('--val_ids', default=[18, 50, 68, 78, 91, 95, 102, 106, 111, 133, 142, 150, 177, 210, 220, 238, 245], nargs='+', type=int,
                        help='List of integers specifying validation sample IDs for custom split (overrides split_ratio if provided)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for data splitting')
    
    # Additional parameters for fit dataset
    parser.add_argument('--train_data_root', type=str, default='/mnt/letitia/scratch/students/hhammoud/detr/synthetic_dataset_test500/',
                        help='Root directory for training data (used with fit dataset)')
    parser.add_argument('--val_data_root', type=str, default='/mnt/letitia/scratch/students/hhammoud/detr/synthetic_dataset_test/',
                        help='Root directory for validation data (used with fit dataset)')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # exclude some parameters from weight decay
    no_decay, with_decay = [], []
    no_decay_names = []  # Track names for debugging

    # Map module names -> modules (keys are case-sensitive original names)
    modules = dict(model_without_ddp.named_modules())

    for name, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue

        # keep original case for parent lookup; use lower for substring tests
        lname = name.lower()
        parent = name.rsplit('.', 1)[0]  # original case
        parent_mod = modules.get(parent, None)

        # 0D/1D params: biases, norm weights, temps etc.
        is_scalar_or_1d = (p.ndim <= 1)

        # true embedding tables by module type
        is_embedding_table = isinstance(parent_mod, torch.nn.Embedding)

        # positional embeddings (narrow patterns)
        is_pos_embed = any(t in lname for t in [
            "pos_embed", "position_embed", "temporal_embed", "temp_embed"
        ])

        # relative position bias tables (common in ViTs/Transformers)
        is_rel_pos_table = any(t in lname for t in [
            "relative_position_bias_table", "rel_pos_bias", "rel_pos_table"
        ])

        if (is_scalar_or_1d or is_embedding_table or is_pos_embed or is_rel_pos_table):
            no_decay.append(p)
            no_decay_names.append(name)
        else:
            with_decay.append(p)

    print(f"Parameters with weight decay: {sum(p.numel() for p in with_decay)}")
    print(f"Parameters without weight decay: {sum(p.numel() for p in no_decay)}")
    print(f"Layers excluded from weight decay: {len(no_decay_names)} parameters")
    print(f"Excluded layer names: {no_decay_names}")

    param_dicts = [
        {"params": with_decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_gamma)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=args.drop_last)

    # Set the appropriate collate function based on dataset
    if args.dataset_file == 'letitia':
        from datasets.letitia_ds import custom_collate_fn
        collate_fn = custom_collate_fn
    else:
        collate_fn = utils.collate_fn

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, data_loader_val, device, args)
        if args.output_dir:
            utils.save_on_master(test_stats, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.epochs, args.clip_max_norm, args.use_schedule_sampling)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(
            model, criterion, data_loader_val, device, args
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
