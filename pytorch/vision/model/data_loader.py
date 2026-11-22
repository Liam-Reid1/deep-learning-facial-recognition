"""Extended Yale B loader. build_dataset.py writes split_manifest.json; each record is one face image.

Preprocessing (proposal steps 2-6): resize, [train: random brightness + rectangle occlusion],
[evaluation: one fixed robustness condition], repeat grayscale to 3 channels, ImageNet normalisation.
"""
import functools
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # pretrained ResNet expects ImageNet stats


def load_image(path, params):
    """uint8 tensor [H, W]. This Kaggle copy is landscape: 192 wide x 168 tall."""
    with Image.open(path) as image:
        resized = image.convert('L').resize((params.image_width, params.image_height))
    return torch.from_numpy(np.array(resized))


def to_float(x):
    return x.float().div(255).unsqueeze(0)  # [1, H, W] in [0, 1]


def random_brightness(x):
    return (x * torch.empty(1).uniform_(0.2, 1.5)).clamp(0, 1)


def repeat_rgb(x):
    return x.repeat(3, 1, 1)


def occlude(x, r0, r1, c0, c1):
    x = x.clone()
    h, w = x.shape[-2:]
    x[..., int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)] = 0
    return x


def identity(x):
    return x


def darken(x):
    return x * 0.3


def brighten(x):
    return (x * 1.8).clamp(0, 1)


# Step 6: fixed, reproducible robustness conditions (band positions approximate the cropped Yale faces).
# Named functions/partials (not lambdas) so DataLoader workers can pickle them on Windows.
CONDITIONS = {
    'clean': identity,
    'dark': darken,
    'bright': brighten,
    'occ_eyes': functools.partial(occlude, r0=0.30, r1=0.50, c0=0.0, c1=1.0),
    'occ_mouth': functools.partial(occlude, r0=0.68, r1=0.88, c0=0.15, c1=0.85),
    'occ_left_half': functools.partial(occlude, r0=0.0, r1=1.0, c0=0.0, c1=0.5),
}


def make_transform(params, training=False, condition='clean'):
    steps = [transforms.Lambda(to_float)]
    if training and getattr(params, 'augment', False):
        steps += [transforms.Lambda(random_brightness),
                  transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0)]
    steps += [transforms.Lambda(CONDITIONS[condition]), transforms.Lambda(repeat_rgb),
              transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def load_manifest(data_dir):
    path = Path(data_dir) / 'split_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found: run build_dataset.py first.')
    return json.loads(path.read_text(encoding='utf-8'))


class YaleBDataset(Dataset):
    def __init__(self, records, classes, params, transform):
        if not records:
            raise ValueError('Empty split.')
        self.records, self.classes, self.transform = records, classes, transform
        self.labels = [classes.index(r['subject']) for r in records]
        # All images held in RAM as uint8 (~0.5 GB for the full 16k-image dataset).
        self.images = torch.stack([load_image(r['path'], params) for r in records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def fetch_dataloader(types, data_dir, params, condition='clean'):
    manifest = load_manifest(data_dir)
    loaders = {}
    for split in types:
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown split: {split}')
        dataset = YaleBDataset(manifest['splits'][split], manifest['classes'], params,
                               make_transform(params, split == 'train', condition))
        generator = torch.Generator().manual_seed(params.seed)
        loaders[split] = DataLoader(dataset, batch_size=params.batch_size,
            shuffle=split == 'train', num_workers=params.num_workers,
            pin_memory=params.device.startswith('cuda'), drop_last=False,
            worker_init_fn=utils.seed_worker, generator=generator)
    return loaders
