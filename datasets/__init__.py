# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .letitia_ds import build_letitia_dataset, custom_collate_fn


def build_dataset(image_set, args):
    # This also includes the Synthetic dataset
    if args.dataset_file == 'letitia':
        return build_letitia_dataset(
            data_root=args.data_root,
            split=image_set,
            split_ratio=args.split_ratio,
            transforms=None,  # Add transforms later if needed
            seed=args.seed
        )

    raise ValueError(f'dataset {args.dataset_file} not supported')
