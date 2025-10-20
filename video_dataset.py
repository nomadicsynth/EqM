import os
import csv
from typing import List, Optional
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
import warnings

# Silence the TorchCodec migration warning
# TODO: Migrate to TorchCodec in the future for better video decoding support
warnings.filterwarnings("ignore", category=UserWarning, message="The video decoding and encoding capabilities of torchvision are deprecated")

from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    """Simple Video dataset loader for UCF-101 style folders.

    Returns clips of shape (C, T, H, W) and a label index.
    """
    def __init__(self, root: str, split: str = 'train', clip_len: int = 16, transform=None, extensions=('.avi',)):
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.transform = transform
        self.extensions = extensions
        csv_path = os.path.join(root, f"{split}.csv")
        self.label_dict = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.label_dict[row['video_name']] = row['tag']
        self.video_paths = []
        self.labels = []
        # collect video files under root/train or root
        base = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if fn.lower().endswith(self.extensions) and fn in self.label_dict:
                    path = os.path.join(dirpath, fn)
                    label = self.label_dict[fn]
                    self.video_paths.append(path)
                    self.labels.append(label)
        # build label->idx mapping
        uniq = sorted(set(self.labels))
        self.label_to_idx = {l: i for i, l in enumerate(uniq)}
        self.labels_idx = [self.label_to_idx[l] for l in self.labels]

    def __len__(self):
        return len(self.video_paths)

    def _read_video_frames(self, path):
        frames, _, _ = read_video(path, pts_unit='sec')
        # frames: T,H,W,C as uint8 tensor
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        return frames  # numpy array

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels_idx[idx]
        frames = self._read_video_frames(path)  # (T, H, W, C)
        n_frames = frames.shape[0]
        if n_frames == 0:
            raise ValueError(f"Video {path} has no frames")
        if n_frames >= self.clip_len:
            # sample clip_len frames uniformly
            indices = np.linspace(0, n_frames - 1, num=self.clip_len, dtype=int)
            clip = frames[indices]
        else:
            # pad by repeating last frame
            reps = self.clip_len - n_frames
            last = frames[-1:]
            clip = np.concatenate([frames, np.repeat(last, reps, axis=0)], axis=0)

        # apply per-frame transform
        pil_frames = [Image.fromarray(f) for f in clip]
        if self.transform is not None:
            proc = [self.transform(p) for p in pil_frames]  # each is C,H,W
            # stack to (T, C, H, W) then permute to (C, T, H, W)
            proc = torch.stack(proc, dim=0)
            proc = proc.permute(1, 0, 2, 3)
        else:
            proc = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0

        return proc, label
