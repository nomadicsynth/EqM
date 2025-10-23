#!/usr/bin/env python3
"""
Analyze clip sampling distribution from real video dataset.

This script samples 100,000 clip parameters from the actual video dataset
and creates visualizations of clip length and position distributions.
"""
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from curriculum_sampler import TemporalClipSampler
import torch
from torchvision.io import read_video
import warnings
from scipy import stats

# Silence the TorchCodec migration warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*TorchCodec.*")

def get_video_durations(data_path, split='train', max_videos=None, cache_file=None):
    """
    Get video durations from the dataset.

    Args:
        data_path: Path to dataset root
        split: Dataset split ('train', 'test', etc.)
        max_videos: Maximum number of videos to process (for speed)
        cache_file: Optional cache file to save/load durations

    Returns:
        List of (video_path, duration_seconds, fps) tuples
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached durations from {cache_file}")
        durations = []
        with open(cache_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                durations.append((row[0], float(row[1]), float(row[2])))
        return durations

    # Load video paths from CSV
    csv_path = os.path.join(data_path, f"{split}.csv")
    video_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_dict[row['video_name']] = row['tag']

    # Find video files
    base = os.path.join(data_path, split) if os.path.isdir(os.path.join(data_path, split)) else data_path
    video_paths = []
    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if fn.lower().endswith(('.avi', '.mp4', '.mov')) and fn in video_dict:
                path = os.path.join(dirpath, fn)
                video_paths.append(path)

    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"Processing {len(video_paths)} videos for duration calculation...")

    durations = []
    for path in tqdm(video_paths, desc="Reading video durations"):
        try:
            # Read just the metadata, not the full video
            _, _, info = read_video(path, pts_unit='sec', start_pts=0, end_pts=0.1)  # Read minimal frames
            fps = info.get('video_fps', 30.0)
            # For duration, we need to know total frames or use metadata
            # Since read_video with end_pts=0.1 gives minimal info, let's try to get duration differently

            # Alternative: use ffprobe or just assume based on filename patterns
            # For UCF-101, videos are typically 2-10 seconds
            # For now, let's read a few frames to estimate

            frames, _, info = read_video(path, pts_unit='sec')
            if len(frames) > 0:
                duration = len(frames) / fps
                durations.append((path, duration, fps))
            else:
                # Skip videos with no frames
                continue

        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

    if cache_file:
        print(f"Saving durations to cache {cache_file}")
        with open(cache_file, 'w') as f:
            f.write("video_path,duration,fps\n")
            for path, dur, fps in durations:
                f.write(f"{path},{dur},{fps}\n")

    return durations

def sample_clip_params_batch(clip_sampler, video_durations, num_samples=None,
                           confidence_level=0.95, margin_of_error=0.01, pilot_samples=1000):
    """
    Sample clip parameters from the video dataset.

    Args:
        clip_sampler: TemporalClipSampler instance
        video_durations: List of (path, duration, fps) tuples
        num_samples: Fixed number of samples (if provided, overrides statistical calculation)
        confidence_level: Desired confidence level for statistical sampling
        margin_of_error: Desired margin of error as fraction of mean
        pilot_samples: Number of pilot samples for estimating std

    Returns:
        Dictionary with sampled parameters
    """
    if not video_durations:
        raise ValueError("No video durations available")

    # If num_samples is specified, use it directly
    if num_samples is not None:
        actual_samples = num_samples
        print(f"Using fixed sample size: {num_samples}")
    else:
        # Do pilot sampling to estimate required sample size
        print(f"Doing pilot sampling with {pilot_samples} samples to estimate required size...")

        pilot_data = []
        for i in range(pilot_samples):
            video_idx = np.random.choice(len(video_durations))
            video_path, video_duration, fps = video_durations[video_idx]

            clip_start, clip_end, clip_duration = clip_sampler.sample_clip_params(
                video_duration, 16, fps, seed=i
            )
            pilot_data.append(clip_duration)

        pilot_data = np.array(pilot_data)

        # Compute required sample size
        actual_samples = compute_required_sample_size(
            confidence_level=confidence_level,
            margin_of_error=margin_of_error,
            pilot_data=pilot_data
        )

        print(f"Pilot sampling complete:")
        print(f"  Pilot mean: {np.mean(pilot_data):.3f}s")
        print(f"  Pilot std:  {np.std(pilot_data, ddof=1):.3f}s")
        print(f"  Required samples for {confidence_level*100:.0f}% confidence, {margin_of_error*100:.1f}% margin: {actual_samples}")

    print(f"Sampling {actual_samples} clip parameters from {len(video_durations)} videos...")

    # Pre-compute for efficiency
    video_indices = np.arange(len(video_durations))
    target_frames = 16  # Typical value

    samples = {
        'clip_start': [],
        'clip_end': [],
        'clip_duration': [],
        'video_duration': [],
        'relative_start': [],  # clip_start / video_duration
        'relative_duration': [],  # clip_duration / video_duration
        'video_idx': []
    }

    for i in tqdm(range(actual_samples), desc="Sampling clips"):
        # Pick random video
        video_idx = np.random.choice(video_indices)
        video_path, video_duration, fps = video_durations[video_idx]

        # Sample clip parameters
        clip_start, clip_end, clip_duration = clip_sampler.sample_clip_params(
            video_duration, target_frames, fps, seed=i + pilot_samples  # Offset seed to avoid overlap with pilot
        )

        # Store results
        samples['clip_start'].append(clip_start)
        samples['clip_end'].append(clip_end)
        samples['clip_duration'].append(clip_duration)
        samples['video_duration'].append(video_duration)
        samples['relative_start'].append(clip_start / video_duration if video_duration > 0 else 0)
        samples['relative_duration'].append(clip_duration / video_duration if video_duration > 0 else 0)
        samples['video_idx'].append(video_idx)

    # Convert to numpy arrays
    for key in samples:
        samples[key] = np.array(samples[key])

    return samples

def create_analysis_plots(samples, output_dir="clip_analysis_plots"):
    """
    Create various plots analyzing clip parameter distributions.

    Args:
        samples: Dictionary with sampled parameters
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Clip duration distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.hist(samples['clip_duration'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Clip Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Clip Duration Distribution')
    plt.grid(True, alpha=0.3)

    # 2. Relative clip duration distribution
    plt.subplot(2, 3, 2)
    plt.hist(samples['relative_duration'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Clip Duration')
    plt.ylabel('Frequency')
    plt.title('Relative Clip Duration Distribution')
    plt.grid(True, alpha=0.3)

    # 3. Clip start position distribution
    plt.subplot(2, 3, 3)
    plt.hist(samples['clip_start'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Clip Start Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Clip Start Position Distribution')
    plt.grid(True, alpha=0.3)

    # 4. Relative start position distribution
    plt.subplot(2, 3, 4)
    plt.hist(samples['relative_start'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Start Position')
    plt.ylabel('Frequency')
    plt.title('Relative Start Position Distribution')
    plt.grid(True, alpha=0.3)

    # 5. Video duration distribution
    plt.subplot(2, 3, 5)
    plt.hist(samples['video_duration'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Video Duration Distribution')
    plt.grid(True, alpha=0.3)

    # 6. Clip duration vs video duration scatter
    plt.subplot(2, 3, 6)
    plt.scatter(samples['video_duration'], samples['clip_duration'], alpha=0.1, s=1)
    plt.xlabel('Video Duration (seconds)')
    plt.ylabel('Clip Duration (seconds)')
    plt.title('Clip vs Video Duration')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/clip_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 2D histogram of relative start vs relative duration
    plt.figure(figsize=(10, 8))
    plt.hist2d(samples['relative_start'], samples['relative_duration'],
               bins=50, cmap='viridis', cmin=1)
    plt.colorbar(label='Frequency')
    plt.xlabel('Relative Start Position')
    plt.ylabel('Relative Clip Duration')
    plt.title('Clip Position vs Duration Heatmap')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/position_duration_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Clip coverage analysis - how much of video is covered
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    coverage = samples['clip_duration'] / samples['video_duration']
    plt.hist(coverage, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Fraction of Video Covered by Clip')
    plt.ylabel('Frequency')
    plt.title('Video Coverage Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(coverage), color='red', linestyle='--', label=f'Mean: {np.mean(coverage):.3f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Clip position preference (start of video vs end)
    start_positions = samples['relative_start']
    plt.hist(start_positions, bins=20, alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('Relative Start Position')
    plt.ylabel('Density')
    plt.title('Clip Start Position Preference')
    plt.grid(True, alpha=0.3)
    # Add uniform distribution line for comparison
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Uniform')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/coverage_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Statistics summary
    print("\n" + "="*60)
    print("CLIP SAMPLING ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples: {len(samples['clip_duration'])}")
    print(f"Unique videos sampled: {len(set(samples['video_idx']))}")
    print()

    print("Clip Duration Statistics:")
    print(f"  Mean: {np.mean(samples['clip_duration']):.3f}s")
    print(f"  Std:  {np.std(samples['clip_duration']):.3f}s")
    print(f"  Min:  {np.min(samples['clip_duration']):.3f}s")
    print(f"  Max:  {np.max(samples['clip_duration']):.3f}s")
    print()

    print("Relative Clip Duration Statistics:")
    print(f"  Mean: {np.mean(samples['relative_duration']):.3f}")
    print(f"  Std:  {np.std(samples['relative_duration']):.3f}")
    print()

    print("Video Duration Statistics:")
    print(f"  Mean: {np.mean(samples['video_duration']):.3f}s")
    print(f"  Std:  {np.std(samples['video_duration']):.3f}s")
    print(f"  Min:  {np.min(samples['video_duration']):.3f}s")
    print(f"  Max:  {np.max(samples['video_duration']):.3f}s")
    print()

    print("Coverage Statistics:")
    print(f"  Mean coverage: {np.mean(coverage):.3f}")
    print(f"  Min coverage:  {np.min(coverage):.3f}")
    print(f"  Max coverage:  {np.max(coverage):.3f}")
    print("="*60)

    # Save statistics to file
    with open(f"{output_dir}/statistics.txt", 'w') as f:
        f.write("Clip Sampling Analysis Statistics\n")
        f.write("="*40 + "\n")
        f.write(f"Total samples: {len(samples['clip_duration'])}\n")
        f.write(f"Unique videos: {len(set(samples['video_idx']))}\n\n")

        f.write("Clip Duration (seconds):\n")
        f.write(f"  Mean: {np.mean(samples['clip_duration']):.3f}\n")
        f.write(f"  Std:  {np.std(samples['clip_duration']):.3f}\n")
        f.write(f"  Min:  {np.min(samples['clip_duration']):.3f}\n")
        f.write(f"  Max:  {np.max(samples['clip_duration']):.3f}\n\n")

        f.write("Relative Clip Duration:\n")
        f.write(f"  Mean: {np.mean(samples['relative_duration']):.3f}\n")
        f.write(f"  Std:  {np.std(samples['relative_duration']):.3f}\n\n")

        f.write("Video Duration (seconds):\n")
        f.write(f"  Mean: {np.mean(samples['video_duration']):.3f}\n")
        f.write(f"  Std:  {np.std(samples['video_duration']):.3f}\n")
        f.write(f"  Min:  {np.min(samples['video_duration']):.3f}\n")
        f.write(f"  Max:  {np.max(samples['video_duration']):.3f}\n\n")

        f.write("Coverage (clip_duration / video_duration):\n")
        f.write(f"  Mean: {np.mean(coverage):.3f}\n")
        f.write(f"  Min:  {np.min(coverage):.3f}\n")
        f.write(f"  Max:  {np.max(coverage):.3f}\n")

def create_venn_diagram(samples, output_dir="clip_analysis_plots"):
    """
    Create a Venn diagram showing relationships between clip categories.

    Args:
        samples: Dictionary with sampled parameters
        output_dir: Directory to save the plot
    """
    # Define categories
    video_duration_median = np.median(samples['video_duration'])
    start_position_median = np.median(samples['relative_start'])

    # Create sets
    short_videos = samples['video_duration'] < video_duration_median
    long_videos = samples['video_duration'] >= video_duration_median
    early_starts = samples['relative_start'] < start_position_median
    late_starts = samples['relative_start'] >= start_position_median

    # Calculate overlaps
    short_early = np.sum(short_videos & early_starts)
    short_late = np.sum(short_videos & late_starts)
    long_early = np.sum(long_videos & early_starts)
    long_late = np.sum(long_videos & late_starts)

    # Only short videos
    only_short = np.sum(short_videos & ~(early_starts | late_starts))
    # Only long videos
    only_long = np.sum(long_videos & ~(early_starts | late_starts))
    # Only early starts
    only_early = np.sum(early_starts & ~(short_videos | long_videos))
    # Only late starts
    only_late = np.sum(late_starts & ~(short_videos | long_videos))

    # All three intersections
    short_early_long_late = 0  # These are mutually exclusive by construction

    # Create Venn diagram
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define circle positions and sizes
    centers = [(3, 3), (7, 3), (5, 5)]  # short videos, long videos, early starts
    radii = [2.5, 2.5, 2.5]

    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Short Videos\n(< {:.1f}s)'.format(video_duration_median),
              'Long Videos\n(≥ {:.1f}s)'.format(video_duration_median),
              'Early Starts\n(< {:.1f})'.format(start_position_median)]

    # Draw circles
    circles = []
    for center, radius, color, label in zip(centers, radii, colors, labels):
        circle = plt.Circle(center, radius, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        circles.append(circle)

        # Add label
        ax.text(center[0], center[1], label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Add intersection labels
    # Center intersection (all three - but this is empty)
    ax.text(5, 4, '0', ha='center', va='center', fontsize=12, fontweight='bold')

    # Two-circle intersections
    # Short ∩ Early
    ax.text(3.8, 4.2, f'{short_early}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Short ∩ Late
    ax.text(4.2, 2.8, f'{short_late}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Long ∩ Early
    ax.text(6.2, 4.2, f'{long_early}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Long ∩ Late
    ax.text(5.8, 2.8, f'{long_late}', ha='center', va='center', fontsize=12, fontweight='bold')

    # Single set regions
    # Only short
    ax.text(2.2, 3, f'{only_short}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Only long
    ax.text(7.8, 3, f'{only_long}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Only early
    ax.text(5, 6.2, f'{only_early}', ha='center', va='center', fontsize=12, fontweight='bold')
    # Only late (this would be outside all circles, but since late is the complement of early,
    # and we're showing all clips, this is actually 0)
    ax.text(5, 1.8, f'{only_late}', ha='center', va='center', fontsize=12, fontweight='bold')

    # Add title
    ax.set_title('Clip Sampling Venn Diagram\nRelationships Between Video Duration and Clip Start Position',
                fontsize=14, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
                      for color in colors]
    ax.legend(legend_elements, labels, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/venn_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print Venn diagram statistics
    print("\n" + "="*60)
    print("VENN DIAGRAM BREAKDOWN")
    print("="*60)
    print(f"Video duration median: {video_duration_median:.2f}s")
    print(f"Start position median: {start_position_median:.3f}")
    print()
    print("Set sizes:")
    print(f"  Short videos (< {video_duration_median:.2f}s): {np.sum(short_videos)}")
    print(f"  Long videos (≥ {video_duration_median:.2f}s): {np.sum(long_videos)}")
    print(f"  Early starts (< {start_position_median:.3f}): {np.sum(early_starts)}")
    print(f"  Late starts (≥ {start_position_median:.3f}): {np.sum(late_starts)}")
    print()
    print("Intersections:")
    print(f"  Short ∩ Early: {short_early}")
    print(f"  Short ∩ Late: {short_late}")
    print(f"  Long ∩ Early: {long_early}")
    print(f"  Long ∩ Late: {long_late}")
    print(f"  Only Short: {only_short}")
    print(f"  Only Long: {only_long}")
    print(f"  Only Early: {only_early}")
    print(f"  Only Late: {only_late}")
    print("="*60)

def compute_required_sample_size(confidence_level=0.95, margin_of_error=0.01, estimated_std=None, pilot_data=None):
    """
    Compute required sample size for estimating mean with desired confidence.

    Args:
        confidence_level: Desired confidence level (e.g., 0.95 for 95%)
        margin_of_error: Desired margin of error as fraction of mean (e.g., 0.01 for 1%)
        estimated_std: Estimated standard deviation (if known)
        pilot_data: Pilot sample data to estimate std from

    Returns:
        Required sample size
    """
    # Z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    if pilot_data is not None:
        # Use pilot data to estimate std
        estimated_std = np.std(pilot_data, ddof=1)
        estimated_mean = np.mean(pilot_data)
        # Margin of error as fraction of mean
        margin_of_error_abs = margin_of_error * estimated_mean
    elif estimated_std is None:
        # Conservative estimate - assume std is 0.5 and mean is 1.0
        estimated_std = 0.5
        margin_of_error_abs = margin_of_error * 1.0
    else:
        # Use provided std, assume mean is 1.0 for margin calculation
        margin_of_error_abs = margin_of_error * 1.0

    # Sample size formula: n = (Z * σ / E)^2
    n = (z_score * estimated_std / margin_of_error_abs) ** 2

    return int(np.ceil(n))

def main():
    parser = argparse.ArgumentParser(description="Analyze clip sampling distribution")
    parser.add_argument("--data-path", type=str, required=True, help="Path to video dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=None, help="Fixed number of clip samples (if not specified, uses statistical calculation)")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level for statistical sampling (e.g., 0.95 for 95%)")
    parser.add_argument("--margin-of-error", type=float, default=0.01, help="Margin of error as fraction of mean (e.g., 0.01 for 1%)")
    parser.add_argument("--pilot-samples", type=int, default=1000, help="Number of pilot samples for estimating required sample size")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum videos to process for duration calculation")
    parser.add_argument("--cache-file", type=str, default="video_durations_cache.csv", help="Cache file for video durations")
    parser.add_argument("--output-dir", type=str, default="clip_analysis_plots", help="Output directory for plots")
    parser.add_argument("--clip-prob", type=float, default=0.3, help="Clip sampling probability")
    parser.add_argument("--min-fraction", type=float, default=0.2, help="Minimum clip fraction")
    parser.add_argument("--max-fraction", type=float, default=0.8, help="Maximum clip fraction")

    args = parser.parse_args()

    # Create clip sampler with specified parameters
    clip_sampler = TemporalClipSampler(
        clip_prob=args.clip_prob,
        min_fraction=args.min_fraction,
        max_fraction=args.max_fraction
    )

    print("Clip Sampler Configuration:")
    print(f"  Clip probability: {args.clip_prob}")
    print(f"  Min fraction: {args.min_fraction}")
    print(f"  Max fraction: {args.max_fraction}")
    print()

    # Get video durations
    video_durations = get_video_durations(
        args.data_path,
        args.split,
        args.max_videos,
        args.cache_file
    )

    if not video_durations:
        print("No videos found or processed. Exiting.")
        return

    print(f"Found {len(video_durations)} videos")
    print(f"Duration range: {min(d[1] for d in video_durations):.2f}s - {max(d[1] for d in video_durations):.2f}s")
    print()

    # Sample clip parameters
    samples = sample_clip_params_batch(
        clip_sampler, 
        video_durations, 
        num_samples=args.num_samples,
        confidence_level=args.confidence_level,
        margin_of_error=args.margin_of_error,
        pilot_samples=args.pilot_samples
    )

    # Create analysis plots
    create_analysis_plots(samples, args.output_dir)
    
    # Create Venn diagram
    create_venn_diagram(samples, args.output_dir)

    print(f"\nAnalysis complete! Plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()