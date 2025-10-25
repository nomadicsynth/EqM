import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import argparse
import hashlib
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )

def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample)}, step=step)


def array2grid(x, nrow=None, labels=None, padding=None):
    """
    Convert a batch tensor to a single image grid (numpy uint8 HWC).

    Args:
        x: torch.Tensor of shape (N,3,H,W) with value range approximately [-1, 1].
        nrow: optional int - number of columns in the grid. If None, rounds(sqrt(N)).
        labels: optional list of str of length N to draw on each cell (top-left).
        padding: optional padding (int). If None and nrow is provided, padding defaults to 0
                 to make cell placement exact; otherwise torchvision default is used.

    Returns:
        numpy array H x W x 3 (uint8)
    """
    N = x.size(0)
    if nrow is None:
        nrow = round(math.sqrt(N)) if N > 0 else 1

    # If caller requested a specific nrow, avoid extra padding so cell positions are predictable
    if padding is None:
        pad = 0 if nrow is not None else 2
    else:
        pad = padding

    grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1), padding=pad)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Draw labels if requested
    if labels is not None and len(labels) == N:
        img = Image.fromarray(grid)
        draw = ImageDraw.Draw(img)
        # Try to use a reasonable default font; fall back to PIL default
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        grid_h, grid_w = grid.shape[:2]
        ncol = nrow
        nrow_calc = int(math.ceil(N / float(ncol)))

        # Compute cell size (integer division is fine; padding was set to zero when nrow specified)
        cell_w = grid_w // ncol if ncol > 0 else grid_w
        cell_h = grid_h // nrow_calc if nrow_calc > 0 else grid_h

        # Draw semi-opaque background rectangles behind labels for readability
        img_rgba = img.convert('RGBA')
        overlay = Image.new('RGBA', img_rgba.size, (255, 255, 255, 0))
        od = ImageDraw.Draw(overlay)

        for i, txt in enumerate(labels):
            row = i // ncol
            col = i % ncol
            x0 = col * cell_w + 4
            y0 = row * cell_h + 4

            # Measure text size (robust across PIL versions)
            if font is not None and hasattr(font, 'getsize'):
                try:
                    text_w, text_h = font.getsize(txt)
                except Exception:
                    text_w, text_h = (len(txt) * 6, 12)
            else:
                # Fallback approximate size
                text_w, text_h = (len(txt) * 6, 12)

            # Rectangle slightly larger than text
            rect = (x0 - 2, y0 - 2, x0 + text_w + 2, y0 + text_h + 2)
            # Semi-opaque black background (alpha ~ 160/255)
            od.rectangle(rect, fill=(0, 0, 0, 160))

            # Draw text with a small black offset for outline, then white text
            if font is not None:
                od.text((x0 + 1, y0 + 1), txt, font=font, fill=(0, 0, 0, 255))
                od.text((x0, y0), txt, font=font, fill=(255, 255, 255, 255))
            else:
                od.text((x0 + 1, y0 + 1), txt, fill=(0, 0, 0, 255))
                od.text((x0, y0), txt, fill=(255, 255, 255, 255))

        # Composite the overlay onto the base image and convert back to RGB numpy
        composite = Image.alpha_composite(img_rgba, overlay)
        grid = np.array(composite.convert('RGB'))

    return grid


def make_video_grid(frames, n_frames, duration=None, nrow=None):
    """
    Build a labeled grid for video frames where each video's frames are laid out horizontally
    (one row per video).

    Args:
        frames: torch.Tensor shape (N*T, 3, H, W) ordered as [vid0_f0, vid0_f1, ..., vid1_f0, ...]
        n_frames: int T, number of frames per video.
        duration: optional float total duration of each clip in seconds. If provided,
                  timestamps will be included in labels as seconds with 3 decimal places.
        nrow: optional override for number of columns; if None, will be set to n_frames.

    Returns:
        numpy uint8 H x W x 3 image (grid) where each row corresponds to a video and
        columns are frames in temporal order. Each cell is labeled "vid{idx} t=X.XXXs".
    """
    if not torch.is_tensor(frames):
        raise TypeError("frames must be a torch.Tensor")
    if nrow is None:
        nrow = n_frames

    total = frames.size(0)
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    if total % n_frames != 0:
        raise ValueError(f"Total frames ({total}) is not divisible by n_frames ({n_frames})")

    n_videos = total // n_frames
    # seconds per frame if duration provided
    if duration is not None:
        sec_per = float(duration) / float(n_frames)
    else:
        sec_per = None

    labels = []
    for vid in range(n_videos):
        for f in range(n_frames):
            if sec_per is not None:
                ts = f * sec_per
                labels.append(f"vid{vid} t={ts:.3f}s")
            else:
                labels.append(f"vid{vid} f{f}")

    return array2grid(frames, nrow=nrow, labels=labels)