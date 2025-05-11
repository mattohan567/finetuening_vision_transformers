"""
NIH Chest X-ray Dataset Loader
------------------------------
This module provides dataset classes and a data loader factory for the NIH Chest X-ray dataset.
It supports:
- Multi-label and binary classification setups
- Image caching for faster I/O
- Balanced and random sampling
- Configurable data augmentation
- Batch loading with optimized transforms
"""

import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image, ImageFile
from collections import OrderedDict
import multiprocessing
import warnings
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms

# Suppress warnings and handle truncated images
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------
# Dataset Implementations
# ---------------------------

class NIHChestXRay(Dataset):
    """Simple NIH Chest X-ray dataset for multi-label classification"""
    LABELS = [
        'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
        'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
        'Fibrosis','Pleural_Thickening','Hernia','No Finding'
    ]

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._map = {os.path.basename(p): p
                     for root in [img_dir]
                     for p in glob(os.path.join(root, '**', '*.png'), recursive=True)}
        # Precompute multi-hot labels
        self.labels = np.zeros((len(self.df), len(self.LABELS)), dtype=np.float32)
        for i, labs in enumerate(self.df['Finding Labels']):
            for lab in labs.split('|'):
                if lab in self.LABELS:
                    self.labels[i, self.LABELS.index(lab)] = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['Image Index']
        path = self._map.get(fname)
        if path and os.path.exists(path):
            img = Image.open(path).convert('RGB')
        else:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        return img, label


class NIHChestDataset(Dataset):
    """Advanced NIH Chest X-ray dataset with caching and error handling"""
    def __init__(self, df, img_dir, transform=None, cache_size=100, verbose=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_size = cache_size
        self.verbose = verbose
        self.image_cache = OrderedDict()
        # Build mapping
        self.image_map = self._build_image_map()
        # Filter to existing images
        exists = [f in self.image_map for f in self.df['Image Index']]
        self.df = self.df[exists].reset_index(drop=True)
        # Precompute multi-hot labels
        self.labels = np.zeros((len(self.df), len(self.all_labels)), dtype=np.float32)
        for i, labs in enumerate(self.df['Finding Labels']):
            for lab in labs.split('|'):
                if lab in self.all_labels:
                    self.labels[i, self.all_labels.index(lab)] = 1

    @property
    def all_labels(self):
        return NIHChestXRay.LABELS

    def _build_image_map(self):
        paths = {}
        roots = [self.img_dir, os.path.join(self.img_dir, 'images')]
        for i in range(1,13):
            roots.append(os.path.join(self.img_dir, f'images_{i:03d}', 'images'))
        for root in roots:
            if not os.path.isdir(root):
                continue
            for p in glob(os.path.join(root, '*.png')):
                paths[os.path.basename(p)] = p
        return paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['Image Index']
        try:
            if fname in self.image_cache:
                img = self.image_cache.pop(fname)
                self.image_cache[fname] = img
            else:
                path = self.image_map.get(fname)
                img = Image.open(path).convert('RGB') if path else Image.new('RGB',(224,224))
                if len(self.image_cache) >= self.cache_size:
                    self.image_cache.popitem(last=False)
                self.image_cache[fname] = img
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            if self.verbose:
                print(f"Error loading {fname}: {e}")
            img = Image.new('RGB',(224,224))
        label = torch.from_numpy(self.labels[idx])
        return img, label

# ---------------------------
# Collate Function
# ---------------------------

def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

# ---------------------------
# Class Weights Helper
# ---------------------------

def compute_class_weights(class_counts):
    counts = np.array(list(class_counts.values()))
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)

# ---------------------------
# Default Transforms & DataLoader Settings
# ---------------------------

DEFAULT_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

WORKER_SETTINGS = {
    'num_workers': min(8,multiprocessing.cpu_count()),
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
}

# ---------------------------
# Data Loader Factory
# ---------------------------

def create_data_loaders(
    data_dir, csv_file, batch_size=32,
    train_transform=None, val_transform=None,
    sample_size=0, test_size=0,
    balance=False, verbose=True
):
    """
    Builds train, val, test DataLoaders.
    Args:
        data_dir: root directory for images
        csv_file: path to Data_Entry_2017.csv
        batch_size: batch size
        sample_size: number of samples for training (0=all)
        test_size: number of samples for test (0=all)
        balance: whether to use balanced sampling
        verbose: print info
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    df = pd.read_csv(csv_file)
    # Binary label: any disease vs no finding
    df['Binary'] = df['Finding Labels'].apply(lambda x: 0 if x=='No Finding' else 1)
    # Optionally subsample
    if sample_size>0 and sample_size<len(df):
        if balance:
            pos = df[df.Binary==1]
            neg = df[df.Binary==0]
            n = sample_size//2
            df = pd.concat([pos.sample(min(n,len(pos))), neg.sample(min(n,len(neg)))])
        else:
            df = df.sample(sample_size)
        df = df.reset_index(drop=True)
    # Split
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df.Binary, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp.Binary, random_state=42)
    # Datasets
    train_ds = NIHChestDataset(train_df, data_dir, transform=train_transform or DEFAULT_TRANSFORMS['train'])
    val_ds   = NIHChestDataset(val_df,   data_dir, transform=val_transform   or DEFAULT_TRANSFORMS['val'])
    test_ds  = NIHChestDataset(test_df,  data_dir, transform=DEFAULT_TRANSFORMS['val'])
    # Subsample test
    if test_size>0 and test_size<len(test_ds):
        test_ds = Subset(test_ds, range(test_size))
    # Class weights
    class_counts = {'No Finding': (train_df.Binary==0).sum(), 'Disease': (train_df.Binary==1).sum()}
    weights = compute_class_weights(class_counts) if verbose else None
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate, **WORKER_SETTINGS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate, **WORKER_SETTINGS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=custom_collate, **WORKER_SETTINGS)
    if verbose:
        print(f"Loaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader, weights

# ---------------------------
# Convenience Function
# ---------------------------

def get_nih_data_loaders(
    data_dir, batch_size=32, sample_size=0, test_size=0,
    balance=False, verbose=True
):
    """Simplified interface: auto-locates CSV and applies defaults"""
    csv_file = os.path.join(data_dir, 'Data_Entry_2017.csv')
    return create_data_loaders(
        data_dir, csv_file, batch_size,
        sample_size=sample_size, test_size=test_size,
        balance=balance, verbose=verbose
    )
