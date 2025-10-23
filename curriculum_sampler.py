"""
Curriculum-based temporal training with VRAM-aware batch scheduling.

Key Ideas:
1. Start with short clips (fine temporal detail) + large batch size
2. Gradually increase frame count while decreasing batch size (constant VRAM)
3. Model learns: fine motion → medium motion → long-term dynamics
4. Maintains consistent gradient statistics via batch size compensation
"""
import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Tuple, Iterator, Optional
import math


class CurriculumTemporalSampler(Sampler):
    """
    Curriculum learning sampler that gradually increases temporal complexity.
    
    Training Phases (example for 4 GPUs, base batch_size=64/gpu):
    
    Phase 1 (epochs 0-10):     1-4 frames,  batch_size=64/gpu  (256 total)
    Phase 2 (epochs 10-20):    3-8 frames,  batch_size=32/gpu  (128 total)
    Phase 3 (epochs 20-40):    5-16 frames, batch_size=16/gpu  (64 total)
    Phase 4 (epochs 40+):      1-16 frames, batch_size=16/gpu  (64 total) [full range]
    
    VRAM usage stays constant: batch_size × num_frames ≈ constant
    
    The curriculum_schedule specifies ABSOLUTE batch sizes per GPU, not multipliers.
    This makes it explicit and easy to understand from the CLI.
    
    Args:
        dataset: VideoDataset with variable-length support
        batch_size: Default/fallback batch size per GPU (typically from CLI --global-batch-size)
        max_frames: Maximum frames to sample
        curriculum_schedule: List of (epoch, min_frames, max_frames, batch_size_per_gpu)
                            If None, uses automatic schedule based on batch_size
        world_size: Number of GPUs for distributed training
        rank: Current GPU rank
        shuffle: Shuffle within buckets
        drop_last: Drop incomplete batches
        seed: Random seed
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_frames: int = 16,
        curriculum_schedule: Optional[List[Tuple[int, int, int, int]]] = None,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        epochs_per_phase: int = 1,
    ):
        self.dataset = dataset
        self.default_batch_size = batch_size
        self.max_frames = max_frames
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Automatic curriculum: Start at num_frames, then double frames while halving batch
        if curriculum_schedule is None:
            curriculum_schedule = []
            current_epoch = 0
            current_max_frames = max_frames  # Start at the specified max_frames
            current_batch = batch_size
            
            while current_batch >= 1:
                curriculum_schedule.append((current_epoch, 1, current_max_frames, current_batch))
                
                # Stop if batch size would go below 1
                if current_batch == 1:
                    break
                    
                # Move to next phase: double frames, halve batch
                current_epoch += epochs_per_phase
                current_max_frames *= 2  # Double the frame count
                current_batch = max(1, current_batch // 2)
            
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        
        # Create buckets for each phase
        self.phase_buckets = self._create_phase_buckets()
        
        # Do not print schedule here; caller (training script) will describe the final schedule
    
    def _create_phase_buckets(self):
        """Pre-create bucket structures for each curriculum phase."""
        phase_buckets = []
        
        for epoch_start, min_frames, max_frames, batch_size_per_gpu in self.curriculum_schedule:
            # Create fine-grained buckets within this phase's range
            # More buckets = less padding
            bucket_size = max(1, (max_frames - min_frames) // 4)  # ~4 buckets per phase
            buckets = []
            
            current = min_frames
            while current <= max_frames:
                bucket_max = min(current + bucket_size, max_frames)
                buckets.append((current, bucket_max))
                current = bucket_max + 1
            
            phase_buckets.append({
                'epoch_start': epoch_start,
                'min_frames': min_frames,
                'max_frames': max_frames,
                'batch_size': batch_size_per_gpu,
                'buckets': buckets,
                'bucket_indices': None,  # Computed lazily
            })
        
        return phase_buckets

    def describe(self, world_size: int = 1, rank: int = 0):
        """Print a human-readable description of the curriculum schedule.

        Call this after the sampler is final (e.g. after epochs_per_phase is computed).
        """
        # Only print from rank 0 in distributed runs to avoid duplicate logs
        if rank != 0:
            return
        print("=" * 80)
        print("Curriculum Temporal Sampler Initialized")
        print("=" * 80)
        if not self.curriculum_schedule:
            print("<empty curriculum schedule>")
            print("=" * 80)
            return

        base_phase = self.curriculum_schedule[0]
        base_max_frames = base_phase[2]
        base_batch = base_phase[3]

        for epoch, min_f, max_f, bs in self.curriculum_schedule:
            global_batch = bs * world_size
            print(f"Epoch {epoch:6d}+: frames=[{min_f:2d}, {max_f:2d}], "
                  f"batch_size={bs}/gpu ({global_batch} global)")
        print("=" * 80)
    
    def _get_current_phase(self):
        """Get current curriculum phase based on training epoch."""
        current_phase = self.phase_buckets[0]
        for phase in self.phase_buckets:
            if self.epoch >= phase['epoch_start']:
                current_phase = phase
            else:
                break
        return current_phase
    
    def _assign_to_buckets(self, phase):
        """Assign dataset indices to buckets for current phase."""
        if phase['bucket_indices'] is not None:
            return phase['bucket_indices']  # Already computed
        
        min_frames = phase['min_frames']
        max_frames = phase['max_frames']
        buckets = phase['buckets']
        
        bucket_indices = {i: [] for i in range(len(buckets))}
        
        # Only print assignment details on rank 0 to avoid duplicate logs in DDP
        if getattr(self, 'rank', 0) == 0:
            print(f"\nAssigning samples to phase [{min_frames}-{max_frames} frames]...")
        
        # For curriculum learning, all videos are eligible since we control sampling range
        # Bucket based on the number of frames we'll sample (random within phase range)
        for idx in range(len(self.dataset)):
            # Assign to buckets based on sampled frame count within phase range
            # Since sampling is random, distribute evenly across buckets
            bucket_idx = idx % len(buckets)
            bucket_indices[bucket_idx].append(idx)
        
        # Print statistics
        total = sum(len(indices) for indices in bucket_indices.values())
        print(f"  Total samples in phase: {total}")
        for bucket_idx, indices in bucket_indices.items():
            bucket_min, bucket_max = buckets[bucket_idx]
            print(f"    Bucket [{bucket_min:2d}-{bucket_max:2d}]: {len(indices):6d} samples")
        
        phase['bucket_indices'] = bucket_indices
        return bucket_indices
    
    def set_epoch(self, epoch: int):
        """Update current training epoch for curriculum progression."""
        self.epoch = epoch
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
    
    def update_dataset_for_phase(self):
        """Update the dataset's frame sampling parameters for the current phase."""
        if not hasattr(self.dataset, 'set_frame_range'):
            return  # Dataset doesn't support dynamic frame ranges
        
        phase = self._get_current_phase()
        # Update dataset to sample within this phase's frame range
        self.dataset.set_frame_range(phase['min_frames'], phase['max_frames'])
    
    def get_current_batch_size(self):
        """Get batch size for current curriculum phase."""
        phase = self._get_current_phase()
        return phase['batch_size']
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches for current curriculum phase."""
        phase = self._get_current_phase()
        bucket_indices = self._assign_to_buckets(phase)
        batch_size = self.get_current_batch_size()
        
        # Generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_batches = []
        
        # Create batches within each bucket
        for bucket_idx, indices in bucket_indices.items():
            if len(indices) == 0:
                continue
            
            # Distribute across GPUs
            # Each GPU gets a subset of the bucket
            indices_per_gpu = len(indices) // self.world_size
            start_idx = self.rank * indices_per_gpu
            end_idx = start_idx + indices_per_gpu if self.rank < self.world_size - 1 else len(indices)
            gpu_indices = indices[start_idx:end_idx]
            
            # Shuffle within bucket
            if self.shuffle:
                perm = torch.randperm(len(gpu_indices), generator=g).tolist()
                gpu_indices = [gpu_indices[i] for i in perm]
            
            # Create batches
            for i in range(0, len(gpu_indices), batch_size):
                batch = gpu_indices[i:i + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batches across buckets
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]
        
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        """Total number of batches in current phase."""
        phase = self._get_current_phase()
        bucket_indices = self._assign_to_buckets(phase)
        batch_size = self.get_current_batch_size()
        
        total_batches = 0
        for indices in bucket_indices.values():
            # Account for distributed training
            indices_per_gpu = len(indices) // self.world_size
            if self.drop_last:
                total_batches += indices_per_gpu // batch_size
            else:
                total_batches += (indices_per_gpu + batch_size - 1) // batch_size
        
        return total_batches


class TemporalClipSampler:
    """
    Sample variable-duration clips from videos to maintain fine temporal detail.
    
    Problem: Sampling 4 frames from 2s video → 0.5s between frames → loses detail
    Solution: Sample CLIPS of shorter duration, then uniformly sample frames within clip
    
    Examples:
        Video: 60 frames over 2.0s (30fps)
        
        Option 1 - Full video sampling (COARSE):
            Sample 4 frames → indices [0, 20, 40, 60] → 0.67s between frames
        
        Option 2 - Clip sampling (FINE):
            Sample 0.5s clip starting at t=0.8s → frames [24-39]
            Sample 4 frames from clip → indices [24, 29, 34, 39] → 0.17s between frames
            ✅ 4x finer temporal resolution!
    
    Args:
        clip_prob: Probability of sampling a clip vs full video
        min_fraction: Minimum fraction of video duration for clip (0.0-1.0)
        max_fraction: Maximum fraction of video duration for clip (0.0-1.0)
    """
    
    def __init__(
        self,
        clip_prob: float = 0.5,
        min_fraction: float = 0.2,
        max_fraction: float = 0.8,
    ):
        self.clip_prob = clip_prob
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
    
    def should_sample_clip(self) -> bool:
        """Decide whether to sample clip or full video."""
        return np.random.random() < self.clip_prob
    
    def sample_clip_params(self, video_duration: float, target_frames: int, fps: float, seed: Optional[int] = None):
        """
        Determine clip start/end for sampling.
        
        Args:
            video_duration: Duration of the video in seconds
            target_frames: Number of frames to sample
            fps: Frames per second of the video
            seed: Optional random seed for reproducible sampling
        
        Returns:
            (clip_start_time, clip_end_time, clip_duration)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random clip duration as fraction of video length
        fraction = np.random.uniform(self.min_fraction, self.max_fraction)
        clip_duration = fraction * video_duration
        
        # Random start time (ensure clip fits in video)
        max_start = video_duration - clip_duration
        clip_start = np.random.uniform(0, max_start)
        clip_end = clip_start + clip_duration
        
        return clip_start, clip_end, clip_duration
