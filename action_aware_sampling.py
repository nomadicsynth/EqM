"""
Intelligent clip sampling that preserves action semantics.

Problem: Random clips may miss the labeled action
Solution: Multi-strategy sampling with action-aware heuristics
"""
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class ClipStrategy(Enum):
    """Different strategies for temporal clip sampling."""
    FULL_VIDEO = "full_video"           # Use entire video
    CENTER_CLIP = "center_clip"         # Sample from center (action likely here)
    RANDOM_CLIP = "random_clip"         # Random temporal window
    MOTION_WEIGHTED = "motion_weighted" # Sample where there's motion (future work)


class ActionAwareClipSampler:
    """
    Samples clips while trying to preserve the labeled action.
    
    Key Insight: Most action videos have the key action in the middle portion.
    UCF-101 and similar datasets are pre-trimmed to focus on the action.
    
    Strategy Mix:
        - 30% full video (learns long-term context)
        - 40% center-biased clips (captures main action)
        - 30% random clips (learns diverse temporal features)
    
    This balances:
        ✅ Fine temporal detail (clip sampling)
        ✅ Action relevance (center bias)
        ✅ Temporal diversity (random sampling)
    
    Args:
        full_video_prob: Probability of using full video
        center_bias_prob: Probability of center-biased clip (given not full video)
        min_clip_duration: Minimum clip duration (seconds)
        max_clip_duration: Maximum clip duration (seconds)
        center_fraction: Fraction of video considered "center" (0.5 = middle 50%)
    """
    
    def __init__(
        self,
        full_video_prob: float = 0.3,
        center_bias_prob: float = 0.57,  # 0.7 * 0.57 ≈ 0.4 overall
        min_clip_duration: float = 0.4,
        max_clip_duration: float = 1.5,
        center_fraction: float = 0.6,
    ):
        self.full_video_prob = full_video_prob
        self.center_bias_prob = center_bias_prob
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.center_fraction = center_fraction
    
    def select_strategy(self) -> ClipStrategy:
        """Choose sampling strategy for this sample."""
        rand = np.random.random()
        
        if rand < self.full_video_prob:
            return ClipStrategy.FULL_VIDEO
        elif rand < self.full_video_prob + (1 - self.full_video_prob) * self.center_bias_prob:
            return ClipStrategy.CENTER_CLIP
        else:
            return ClipStrategy.RANDOM_CLIP
    
    def sample_clip_window(
        self,
        video_duration: float,
        target_frames: int,
        fps: float,
        strategy: Optional[ClipStrategy] = None
    ) -> Tuple[float, float, float, ClipStrategy]:
        """
        Sample temporal window based on strategy.
        
        Returns:
            (start_time, end_time, duration, strategy_used)
        """
        if strategy is None:
            strategy = self.select_strategy()
        
        if strategy == ClipStrategy.FULL_VIDEO:
            return 0.0, video_duration, video_duration, strategy
        
        # Determine clip duration
        clip_duration = np.random.uniform(self.min_clip_duration, self.max_clip_duration)
        
        # Ensure clip duration is reasonable for target frames
        min_duration_for_frames = (target_frames - 1) / fps
        clip_duration = max(clip_duration, min_duration_for_frames * 1.2)
        clip_duration = min(clip_duration, video_duration)  # Can't exceed video
        
        if clip_duration >= video_duration * 0.95:
            # Clip is essentially full video
            return 0.0, video_duration, video_duration, ClipStrategy.FULL_VIDEO
        
        # Determine start time based on strategy
        max_start = video_duration - clip_duration
        
        if strategy == ClipStrategy.CENTER_CLIP:
            # Sample from center portion of video
            video_center = video_duration / 2
            center_window_size = video_duration * self.center_fraction
            
            # Start of center window
            center_start = max(0, video_center - center_window_size / 2)
            center_end = min(video_duration - clip_duration, video_center + center_window_size / 2)
            
            if center_end <= center_start:
                # Center window too small, fall back to random
                clip_start = np.random.uniform(0, max_start)
            else:
                clip_start = np.random.uniform(center_start, center_end)
                # Ensure clip fits
                clip_start = min(clip_start, video_duration - clip_duration)
        
        else:  # RANDOM_CLIP
            clip_start = np.random.uniform(0, max_start)
        
        clip_end = clip_start + clip_duration
        
        return clip_start, clip_end, clip_duration, strategy
    
    def get_frame_indices(
        self,
        total_frames: int,
        target_frames: int,
        clip_start_time: float,
        clip_end_time: float,
        fps: float
    ) -> Tuple[np.ndarray, float]:
        """
        Get frame indices to sample within the clip window.
        
        Returns:
            (frame_indices, actual_time_span)
        """
        # Convert times to frame indices
        start_frame = int(clip_start_time * fps)
        end_frame = int(clip_end_time * fps)
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        clip_frames = end_frame - start_frame
        
        if clip_frames >= target_frames:
            # Sample uniformly within clip
            indices = np.linspace(start_frame, end_frame - 1, num=target_frames, dtype=int)
            actual_time_span = (indices[-1] - indices[0]) / fps
        else:
            # Clip too short, take all frames (will be padded later)
            indices = np.arange(start_frame, end_frame, dtype=int)
            actual_time_span = (end_frame - start_frame - 1) / fps if clip_frames > 1 else 0.0
        
        return indices, actual_time_span


# Curriculum-aware clip sampling
class CurriculumClipSampler(ActionAwareClipSampler):
    """
    Adapts clip sampling strategy based on training curriculum.
    
    Early training (short clips): Higher clip sampling, shorter clips
    Later training (long clips): More full video, longer clips
    
    This ensures model sees fine detail early, then learns to integrate over time.
    """
    
    def __init__(
        self,
        base_full_video_prob: float = 0.3,
        base_center_bias_prob: float = 0.57,
        base_min_clip_duration: float = 0.3,
        base_max_clip_duration: float = 1.5,
        center_fraction: float = 0.6,
    ):
        super().__init__(
            full_video_prob=base_full_video_prob,
            center_bias_prob=base_center_bias_prob,
            min_clip_duration=base_min_clip_duration,
            max_clip_duration=base_max_clip_duration,
            center_fraction=center_fraction,
        )
        self.base_full_video_prob = base_full_video_prob
        self.base_min_clip_duration = base_min_clip_duration
        self.base_max_clip_duration = base_max_clip_duration
    
    def update_for_curriculum_phase(
        self,
        phase_min_frames: int,
        phase_max_frames: int,
        max_frames: int = 16
    ):
        """
        Adjust sampling strategy based on curriculum phase.
        
        Early phases (few frames): Aggressive clip sampling, short clips
        Later phases (many frames): More full videos, longer clips
        """
        # Progress through curriculum (0.0 = start, 1.0 = end)
        progress = (phase_max_frames / max_frames)
        
        # Increase full video probability as we progress
        # Early: 30% full → Late: 60% full
        self.full_video_prob = self.base_full_video_prob + progress * 0.3
        
        # Increase clip duration range as we progress
        # Early: [0.3, 1.5]s → Late: [0.5, 2.5]s
        self.min_clip_duration = self.base_min_clip_duration + progress * 0.2
        self.max_clip_duration = self.base_max_clip_duration + progress * 1.0


# Integration into VideoDataset
"""
class VideoDataset(Dataset):
    def __init__(
        self,
        ...,
        clip_sampler: Optional[ActionAwareClipSampler] = None,
        use_clip_sampling: bool = True,
    ):
        self.clip_sampler = clip_sampler
        self.use_clip_sampling = use_clip_sampling
        
        if use_clip_sampling and clip_sampler is None:
            self.clip_sampler = ActionAwareClipSampler()
    
    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels_idx[idx]
        frames, fps = self._read_video_frames(path)
        n_frames = frames.shape[0]
        video_duration = (n_frames - 1) / fps if n_frames > 1 else 0.0
        
        # Determine target number of frames
        if self.single_frame_prob > 0 and random.random() < self.single_frame_prob:
            target_frames = 1
        elif self.random_frames and self.min_frames < self.num_frames:
            target_frames = random.randint(self.min_frames, self.num_frames)
        else:
            target_frames = self.num_frames
        
        # Sample clip window
        if self.use_clip_sampling and self.clip_sampler and target_frames > 1:
            clip_start, clip_end, clip_duration, strategy = self.clip_sampler.sample_clip_window(
                video_duration, target_frames, fps
            )
            
            # Get frame indices within clip
            indices, time_span = self.clip_sampler.get_frame_indices(
                n_frames, target_frames, clip_start, clip_end, fps
            )
            
            # Extract frames
            clip = frames[indices]
            
        else:
            # Original full-video sampling
            if target_frames == 1:
                idx_frame = random.randint(0, n_frames - 1)
                indices = np.array([idx_frame])
                time_span = 0.0
            elif n_frames >= target_frames:
                indices = np.linspace(0, n_frames - 1, num=target_frames, dtype=int)
                time_span = (indices[-1] - indices[0]) / fps
            else:
                # Pad short video
                indices = np.arange(n_frames)
                time_span = video_duration
            
            clip = frames[indices]
        
        # Handle padding if needed
        if clip.shape[0] < target_frames:
            last = clip[-1:]
            reps = target_frames - clip.shape[0]
            clip = np.concatenate([clip, np.repeat(last, reps, axis=0)], axis=0)
        
        # Apply transforms...
        pil_frames = [Image.fromarray(f) for f in clip]
        if self.transform is not None:
            proc = [self.transform(p) for p in pil_frames]
            proc = torch.stack(proc, dim=0).permute(1, 0, 2, 3)
        else:
            proc = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        
        return proc, label, time_span
"""
