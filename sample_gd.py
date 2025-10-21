# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal sampling script for EqM using PyTorch DDP.
"""
import math
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm
from models import EqM_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F
import imageio
from torch.amp import autocast


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def main(args):
    """
    Trains a new EqM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    n_gpus = torch.cuda.device_count()
    # disable flash for energy training
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if getattr(args, 'video', False):
        input_size = (args.clip_len, latent_size, latent_size)
    else:
        input_size = latent_size
    model = EqM_models[args.model](
        input_size=input_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
        else:
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)

        ema = ema.to(device)
        model = model.to(device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[device])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Determine the dtype for autocast during sampling
    if args.use_bf16:
        autocast_dtype = torch.bfloat16
        print("Using BF16 for inference")
    elif args.use_amp:
        autocast_dtype = torch.float16
        print("Using FP16 for inference")
    else:
        autocast_dtype = torch.float32
        print("Using FP32 for inference")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # if args.ebm == 'none':
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    if getattr(args, 'video', False):
        zs = torch.randn(n, 4, args.clip_len, latent_size, latent_size, device=device)
    else:
        zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward    

    if rank == 0:
        os.makedirs(args.folder, exist_ok=True)
        if getattr(args, 'video', False):
            print(f"Generating videos at {args.target_fps} FPS with {args.clip_len} frames")
    # Compute time_scale for video generation
    if getattr(args, 'video', False):
        # time_scale is seconds per frame
        time_scale = 1.0 / args.target_fps  # e.g., 30 fps -> 0.033 sec/frame
    else:
        time_scale = 1.0
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / args.global_batch_size) * args.global_batch_size)
    if rank == 0:
        sample_type = "videos" if getattr(args, 'video', False) else "images"
        print(f"Total number of {sample_type} that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(total_samples // args.global_batch_size)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    n = int(args.global_batch_size // dist.get_world_size())
    for i in pbar:
        with torch.no_grad():
            if getattr(args, 'video', False):
                z = torch.randn(n, 4, args.clip_len, latent_size, latent_size, device=device)
            else:
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)
            t = torch.ones((n,)).to(z).to(device)
            if use_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, time_scale=time_scale)
                t = torch.cat([t, t], 0)
            else:
                model_kwargs = dict(y=y, time_scale=time_scale)
            xt = z
            m = torch.zeros_like(xt).to(xt).to(device)
            
            # Use autocast for the sampling loop
            with autocast(device_type='cuda', dtype=autocast_dtype, enabled=args.use_amp or args.use_bf16):
                for i in range(args.num_sampling_steps-1):
                    if args.sampler == 'gd':
                        out = model_fn(xt, t, **model_kwargs)
                        if not torch.is_tensor(out):
                            out = out[0]
                    if args.sampler == 'ngd':
                        x_ = xt + args.stepsize*m*args.mu
                        out = model_fn(x_, t, **model_kwargs)
                        if not torch.is_tensor(out):
                            out = out[0]
                        m = out
                    
                    xt = xt + out*args.stepsize
                    t += args.stepsize
            
            if use_cfg:
                xt, _ = xt.chunk(2, dim=0)
            if getattr(args, 'video', False):
                # xt: (n, 4, T, H_lat, W_lat)
                N, C, T, H_lat, W_lat = xt.shape
                xt_frames = xt.permute(0, 2, 1, 3, 4).reshape(N * T, C, H_lat, W_lat)
                # Decode in batches to avoid OOM
                decoded_frames = []
                batch_size = args.decode_batch_size  # Adjust based on GPU memory
                for i in range(0, xt_frames.shape[0], batch_size):
                    batch = xt_frames[i:i+batch_size]
                    decoded = vae.decode(batch / 0.18215).sample
                    decoded_frames.append(decoded)
                samples = torch.cat(decoded_frames, dim=0)
                # samples: (N*T, 3, H, W)
                samples = samples.reshape(N, T, 3, args.image_size, args.image_size)
                # permute to (N, T, H, W, 3) for imageio
                samples = samples.permute(0, 1, 3, 4, 2)
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
                for i, video in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    imageio.mimsave(f"{args.folder}/{index:06d}.gif", video, fps=args.target_fps)
            else:
                samples = vae.decode(xt / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for i, sample in enumerate(samples):
                    index = i * dist.get_world_size() + rank + total
                    Image.fromarray(sample).save(f"{args.folder}/{index:06d}.png")
        total += args.global_batch_size
        dist.barrier()
    if rank == 0:
        if not getattr(args, 'video', False):
            print("Creating .npz file")
            create_npz_from_sample_folder(args.folder, 50000)
        print("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will sample EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom EqM checkpoint")
    parser.add_argument("--stepsize", type=float, default=0.0017,
                        help="step size eta")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--folder", type=str, default='samples')
    parser.add_argument("--sampler", type=str, default='gd', choices=['gd', 'ngd'])
    parser.add_argument("--mu", type=float, default=0.3,
                        help="NAG-GD hyperparameter mu")
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--uncond", type=bool, default=True,
                        help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="energy formulation")
    parser.add_argument("--video", action="store_true", help="Enable video sampling mode")
    parser.add_argument("--clip-len", type=int, default=16, help="Number of frames per video clip")
    parser.add_argument("--target-fps", type=int, default=30, help="Target frame rate for generated videos (affects temporal dynamics)")
    parser.add_argument("--decode-batch-size", type=int, default=64, help="Batch size for VAE decoding")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision inference (FP16)")
    parser.add_argument("--use-bf16", action="store_true", help="Enable bfloat16 mixed precision inference (BF16)")
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
