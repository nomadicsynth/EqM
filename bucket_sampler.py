"""
Temporal bucketing sampler for efficient variable-length video training.
Groups videos by similar temporal length to minimize padding waste.
"""
import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Tuple, Iterator


class TemporalBucketSampler(Sampler):
    """
    Sampler that groups videos into temporal buckets to minimize padding.
    
    Each batch contains videos of similar length, reducing wasted computation
    on padding while maintaining GPU parallelism.
    
    Example:
        buckets = [(1, 2), (3, 4), (5, 8), (9, 16)]
        Videos with 1-2 frames go in first bucket (padded to 2)
        Videos with 3-4 frames go in second bucket (padded to 4)
        etc.
    
    Args:
        dataset: VideoDataset instance
        batch_size: Number of samples per batch
        buckets: List of (min_frames, max_frames) tuples defining buckets
        shuffle: Whether to shuffle within buckets
        drop_last: Whether to drop incomplete batches
        seed: Random seed for shuffling
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        buckets: List[Tuple[int, int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Default buckets if not provided
        if buckets is None:
            buckets = [(1, 2), (3, 4), (5, 8), (9, 12), (13, 16)]
        self.buckets = buckets
        
        # Pre-compute bucket assignments for all samples
        self.bucket_indices = self._assign_to_buckets()
        
    def _assign_to_buckets(self):
        """Assign each dataset index to appropriate bucket based on frame count."""
        bucket_indices = {i: [] for i in range(len(self.buckets))}
        
        print(f"Assigning {len(self.dataset)} videos to {len(self.buckets)} temporal buckets...")
        
        for idx in range(len(self.dataset)):
            # Get frame count for this video
            # This assumes dataset tracks frame counts somehow
            # You may need to adapt this based on your dataset structure
            video, _, time_span = self.dataset[idx]
            num_frames = video.shape[1]  # (C, T, H, W)
            
            # Find appropriate bucket
            for bucket_idx, (min_frames, max_frames) in enumerate(self.buckets):
                if min_frames <= num_frames <= max_frames:
                    bucket_indices[bucket_idx].append(idx)
                    break
            else:
                # No bucket found - assign to largest bucket
                bucket_indices[len(self.buckets) - 1].append(idx)
        
        # Print bucket statistics
        for bucket_idx, indices in bucket_indices.items():
            min_f, max_f = self.buckets[bucket_idx]
            print(f"  Bucket [{min_f:2d}-{max_f:2d} frames]: {len(indices):6d} videos")
        
        return bucket_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices grouped by temporal bucket."""
        # Set random seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch if hasattr(self, 'epoch') else self.seed)
        
        all_batches = []
        
        # Create batches within each bucket
        for bucket_idx, indices in self.bucket_indices.items():
            if len(indices) == 0:
                continue
            
            # Shuffle within bucket if requested
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            
            # Create batches for this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batches across buckets (optional - maintains some length grouping)
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]
        
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        """Total number of batches."""
        total_batches = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling with distributed training."""
        self.epoch = epoch


# Usage in train.py:
"""
from bucket_sampler import TemporalBucketSampler

# Replace DistributedSampler with:
sampler = TemporalBucketSampler(
    dataset,
    batch_size=local_batch_size,
    buckets=[(1, 2), (3, 4), (5, 8), (9, 16)],
    shuffle=True,
    drop_last=True,
    seed=args.global_seed
)

# Collate function pads to bucket max instead of batch max
def bucket_aware_collate_fn(batch):
    videos, labels, time_spans, num_real_frames = zip(*batch)
    
    # All videos in batch should be in same bucket, so max is close to actual
    max_frames = max(v.shape[1] for v in videos)
    
    # Pad to max (minimal padding due to bucketing!)
    padded_videos = []
    for video in videos:
        C, T, H, W = video.shape
        if T < max_frames:
            last_frame = video[:, -1:, :, :]
            padding = last_frame.repeat(1, max_frames - T, 1, 1)
            video = torch.cat([video, padding], dim=1)
        padded_videos.append(video)
    
    return (
        torch.stack(padded_videos),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(time_spans, dtype=torch.float32),
        torch.tensor(num_real_frames, dtype=torch.long)
    )
"""
