import torch


def collate_fn(batch):
    images, captions = zip(*batch)
    return images, captions