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
from curriculum_sampler import TemporalClipSampler


class VideoDataset(Dataset):
    """Simple Video dataset loader for UCF-101 style folders.

    Returns clips of shape (C, T, H, W) and a label index.
    
    Args:
        root: Root directory containing videos
        split: Dataset split ('train', 'test', etc.)
        num_frames: Maximum number of frames per clip
        transform: Transform to apply to each frame
        extensions: Video file extensions to load
        random_frames: If True, randomly sample num_frames between min_frames and num_frames
        min_frames: Minimum number of frames when random_frames=True
        single_frame_prob: Probability of returning a single frame (image) instead of video
        clip_sampler: Optional TemporalClipSampler for sampling temporal clips
    """
    def __init__(
        self, 
        root: str, 
        split: str = 'train', 
        num_frames: int = 16, 
        transform=None, 
        extensions=('.avi',),
        random_frames: bool = False,
        min_frames: int = 1,
        single_frame_prob: float = 0.0,
        clip_sampler: Optional[TemporalClipSampler] = None
    ):
        super().__init__()
        self.root = root
        self.num_frames = num_frames
        self.transform = transform
        self.extensions = extensions
        self.random_frames = random_frames
        self.min_frames = max(1, min_frames)  # Ensure at least 1 frame
        self.single_frame_prob = max(0.0, min(1.0, single_frame_prob))  # Clamp to [0, 1]
        self.clip_sampler = clip_sampler
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
        frames, _, info = read_video(path, pts_unit='sec')
        # frames: T,H,W,C as uint8 tensor
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        fps = info.get('video_fps', 30.0)  # Default to 30 fps if not available
        return frames, fps  # numpy array and fps

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels_idx[idx]
        frames, fps = self._read_video_frames(path)  # (T, H, W, C) and fps
        n_frames = frames.shape[0]
        if n_frames == 0:
            raise ValueError(f"Video {path} has no frames")
        
        # Determine target number of frames for this sample
        # Check if we should return a single frame (image)
        if self.single_frame_prob > 0 and random.random() < self.single_frame_prob:
            target_frames = 1
        elif self.random_frames and self.min_frames < self.num_frames:
            # Randomly sample between min_frames and num_frames
            target_frames = random.randint(self.min_frames, self.num_frames)
        else:
            target_frames = self.num_frames
        
        video_duration = n_frames / fps
        
        # Determine if we're sampling a clip or full video
        if self.clip_sampler is not None and self.clip_sampler.should_sample_clip():
            clip_start, clip_end, clip_duration = self.clip_sampler.sample_clip_params(
                video_duration, target_frames, fps
            )
            # Convert to frame indices
            start_frame = int(clip_start * fps)
            end_frame = int(clip_end * fps)
            clip_frames = frames[start_frame:end_frame]
            clip_n_frames = len(clip_frames)
            
            if clip_n_frames >= target_frames:
                # Sample uniformly from the clip
                indices = np.linspace(0, clip_n_frames - 1, target_frames, dtype=int)
                sampled_frames = clip_frames[indices]
                time_span = clip_duration
            else:
                # Clip too short: take target_frames consecutive frames centered on clip start
                desired_start = start_frame - (target_frames // 2)
                actual_start = max(0, min(desired_start, n_frames - target_frames))
                sampled_frames = frames[actual_start : actual_start + target_frames]
                # Adjust time_span to reflect consecutive frames at video fps
                time_span = (target_frames - 1) / fps if target_frames > 1 else 0.0
        else:
            # Original full-video sampling logic
            if n_frames >= target_frames:
                # sample target_frames frames uniformly
                if target_frames == 1:
                    # For single frame, pick a random frame from the video
                    idx_frame = random.randint(0, n_frames - 1)
                    indices = np.array([idx_frame])
                else:
                    indices = np.linspace(0, n_frames - 1, num=target_frames, dtype=int)
                sampled_frames = frames[indices]
                # Calculate actual time span of sampled frames
                if target_frames == 1:
                    time_span = 0.0  # Single frame = no temporal extent (image)
                else:
                    time_span = (indices[-1] - indices[0]) / fps  # in seconds
            else:
                # pad by repeating last frame
                indices = list(range(n_frames))
                reps = target_frames - n_frames
                last = frames[-1:]
                sampled_frames = np.concatenate([frames, np.repeat(last, reps, axis=0)], axis=0)
                # Time span is the full video duration
                time_span = (n_frames - 1) / fps if n_frames > 1 else 0.0

        # apply per-frame transform
        pil_frames = [Image.fromarray(f) for f in sampled_frames]
        if self.transform is not None:
            proc = [self.transform(p) for p in pil_frames]  # each is C,H,W
            # stack to (T, C, H, W) then permute to (C, T, H, W)
            proc = torch.stack(proc, dim=0)
            proc = proc.permute(1, 0, 2, 3)
        else:
            proc = torch.from_numpy(sampled_frames).permute(3, 0, 1, 2).float() / 255.0

        return proc, label, time_span
