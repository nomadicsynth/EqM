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
    
    Phase 1 (steps 0-10k):     1-4 frames,  batch_size=64/gpu  (256 total)
    Phase 2 (steps 10k-20k):   3-8 frames,  batch_size=32/gpu  (128 total)
    Phase 3 (steps 20k-40k):   5-16 frames, batch_size=16/gpu  (64 total)
    Phase 4 (steps 40k+):      1-16 frames, batch_size=16/gpu  (64 total) [full range]
    
    VRAM usage stays constant: batch_size × num_frames ≈ constant
    
    The curriculum_schedule specifies ABSOLUTE batch sizes per GPU, not multipliers.
    This makes it explicit and easy to understand from the CLI.
    
    Args:
        dataset: VideoDataset with variable-length support
        batch_size: Default/fallback batch size per GPU (typically from CLI --global-batch-size)
        max_frames: Maximum frames to sample
        curriculum_schedule: List of (step, min_frames, max_frames, batch_size_per_gpu)
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
        steps_per_phase: int = 10000,
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
        self.step = 0  # Global training step (updated externally)
        
        # Automatic curriculum: Start at num_frames, then double frames while halving batch
        if curriculum_schedule is None:
            curriculum_schedule = []
            current_step = 0
            current_max_frames = max_frames  # Start at the specified max_frames
            current_batch = batch_size
            
            while current_batch >= 1:
                curriculum_schedule.append((current_step, 1, current_max_frames, current_batch))
                
                # Stop if batch size would go below 1
                if current_batch == 1:
                    break
                    
                # Move to next phase: double frames, halve batch
                current_step += steps_per_phase
                current_max_frames *= 2  # Double the frame count
                current_batch = max(1, current_batch // 2)
            
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        
        # Create buckets for each phase
        self.phase_buckets = self._create_phase_buckets()
        
        # Do not print schedule here; caller (training script) will describe the final schedule
    
    def _create_phase_buckets(self):
        """Pre-create bucket structures for each curriculum phase."""
        phase_buckets = []
        
        for step_start, min_frames, max_frames, batch_size_per_gpu in self.curriculum_schedule:
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
                'step_start': step_start,
                'min_frames': min_frames,
                'max_frames': max_frames,
                'batch_size': batch_size_per_gpu,
                'buckets': buckets,
                'bucket_indices': None,  # Computed lazily
            })
        
        return phase_buckets

    def describe(self, world_size: int = 1, rank: int = 0):
        """Print a human-readable description of the curriculum schedule.

        Call this after the sampler is final (e.g. after steps_per_phase is computed).
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

        for step, min_f, max_f, bs in self.curriculum_schedule:
            global_batch = bs * world_size
            print(f"Step {step:6d}+: frames=[{min_f:2d}, {max_f:2d}], "
                  f"batch_size={bs}/gpu ({global_batch} global)")
        print("=" * 80)
    
    def _get_current_phase(self):
        """Get current curriculum phase based on training step."""
        current_phase = self.phase_buckets[0]
        for phase in self.phase_buckets:
            if self.step >= phase['step_start']:
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
        
        # Note: This assumes dataset can report frame counts without loading
        # You may need to adapt based on your dataset structure
        for idx in range(len(self.dataset)):
            # Get actual frame count for this video
            # This is a simplification - you may want to cache this
            video, _, _ = self.dataset[idx]
            num_frames = video.shape[1]  # (C, T, H, W)
            
            # Skip if outside phase range
            if num_frames < min_frames or num_frames > max_frames:
                continue
            
            # Find appropriate bucket
            for bucket_idx, (bucket_min, bucket_max) in enumerate(buckets):
                if bucket_min <= num_frames <= bucket_max:
                    bucket_indices[bucket_idx].append(idx)
                    break
        
        # Print statistics
        total = sum(len(indices) for indices in bucket_indices.values())
        print(f"  Total samples in phase: {total}")
        for bucket_idx, indices in bucket_indices.items():
            bucket_min, bucket_max = buckets[bucket_idx]
            print(f"    Bucket [{bucket_min:2d}-{bucket_max:2d}]: {len(indices):6d} samples")
        
        phase['bucket_indices'] = bucket_indices
        return bucket_indices
    
    def set_step(self, step: int):
        """Update current training step for curriculum progression."""
        self.step = step
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch
    
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
        min_clip_duration: Minimum clip duration (seconds)
        max_clip_duration: Maximum clip duration (seconds)
    """
    
    def __init__(
        self,
        clip_prob: float = 0.5,
        min_clip_duration: float = 0.5,
        max_clip_duration: float = 1.5,
    ):
        self.clip_prob = clip_prob
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
    
    def should_sample_clip(self) -> bool:
        """Decide whether to sample clip or full video."""
        return np.random.random() < self.clip_prob
    
    def sample_clip_params(self, video_duration: float, target_frames: int, fps: float):
        """
        Determine clip start/end for sampling.
        
        Returns:
            (clip_start_time, clip_end_time, clip_duration)
        """
        # Random clip duration
        clip_duration = np.random.uniform(self.min_clip_duration, self.max_clip_duration)
        clip_duration = min(clip_duration, video_duration)  # Can't exceed video length
        
        # Ensure clip can contain target_frames with reasonable spacing
        min_duration_for_frames = (target_frames - 1) / fps  # Minimum duration to fit frames
        clip_duration = max(clip_duration, min_duration_for_frames * 1.2)  # Add 20% buffer
        
        # Random start time (ensure clip fits in video)
        max_start = video_duration - clip_duration
        if max_start <= 0:
            # Clip must be full video
            return 0.0, video_duration, video_duration
        
        clip_start = np.random.uniform(0, max_start)
        clip_end = clip_start + clip_duration
        
        return clip_start, clip_end, clip_duration


# Usage example - integrate into VideoDataset.__getitem__:
"""
class VideoDataset(Dataset):
    def __init__(self, ..., clip_sampler: Optional[TemporalClipSampler] = None):
        self.clip_sampler = clip_sampler
    
    def __getitem__(self, idx):
        frames, fps = self._read_video_frames(path)
        n_frames = frames.shape[0]
        video_duration = n_frames / fps
        
        # Determine if we're sampling a clip or full video
        if self.clip_sampler and self.clip_sampler.should_sample_clip():
            # Sample a temporal clip
            clip_start, clip_end, clip_duration = self.clip_sampler.sample_clip_params(
                video_duration, target_frames, fps
            )
            
            # Convert to frame indices
            start_frame = int(clip_start * fps)
            end_frame = int(clip_end * fps)
            clip_frames = frames[start_frame:end_frame]
            
            # Now sample target_frames uniformly from this clip
            clip_n_frames = clip_frames.shape[0]
            if clip_n_frames >= target_frames:
                indices = np.linspace(0, clip_n_frames - 1, num=target_frames, dtype=int)
                sampled_frames = clip_frames[indices]
                time_span = clip_duration  # Actual time span of sampled frames
            else:
                # Clip too short, pad
                sampled_frames = clip_frames
                time_span = clip_duration
        else:
            # Sample from full video (existing logic)
            indices = np.linspace(0, n_frames - 1, num=target_frames, dtype=int)
            sampled_frames = frames[indices]
            time_span = video_duration
        
        # Continue with transforms...
"""
