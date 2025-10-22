#!/usr/bin/env python3
"""
Validate FID implementation by computing FID for multiple checkpoints.
Expectation: Later checkpoints should have lower (better) FID scores.
"""

import sys
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from scipy import linalg
from pathlib import Path
import argparse
from tqdm import tqdm
from copy import deepcopy

from models import EqM_models
from download import find_model
from diffusers.models import AutoencoderKL
from torchvision import models, transforms
from video_dataset import VideoDataset
from torch.utils.data import DataLoader


def compute_fid_from_inception_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet Inception Distance between two Gaussian distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def get_inception_features(images, inception_model, batch_size=32, device='cuda'):
    """Extract InceptionV3 features from images."""
    inception_model.eval()
    features = []
    
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size].to(device)
        # Resize to 299x299 for InceptionV3
        if batch.shape[-1] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize from [-1, 1] to [0, 1] then to ImageNet stats
        batch = (batch + 1) / 2  # [-1, 1] -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        
        feat = inception_model(batch)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def compute_stats(features):
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def center_crop_arr(pil_image, image_size):
    """Center cropping from train.py"""
    from PIL import Image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def generate_samples_ngd(model, vae, num_samples, num_frames, latent_size, num_classes,
                         stepsize, mu, num_steps, time_scale, cfg_scale, device, use_bf16):
    """Generate samples using NGD sampling."""
    model.eval()
    all_frames = []
    
    batch_size = min(num_samples, 8)  # Process in small batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    for _ in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - len(all_frames))
        
        # Initialize noise
        zs = torch.randn(current_batch_size, 4, num_frames, latent_size, latent_size, device=device)
        ys = torch.randint(num_classes, size=(current_batch_size,), device=device)
        
        # Setup CFG
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([num_classes] * current_batch_size, device=device)
            ys = torch.cat([ys, y_null], 0)
            model_fn = model.forward_with_cfg
            model_kwargs = dict(y=ys, cfg_scale=cfg_scale, time_scale=time_scale)
        else:
            model_fn = model.forward
            model_kwargs = dict(y=ys, time_scale=time_scale)
        
        # NGD sampling
        xt = zs.clone()
        t = torch.ones((xt.shape[0],), device=device)
        m = torch.zeros_like(xt)
        
        with autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_bf16):
            for _ in range(num_steps - 1):
                x_ = xt + stepsize * m * mu
                out = model_fn(x_, t, **model_kwargs)
                if not torch.is_tensor(out):
                    out = out[0]
                m = out
                xt = xt + out * stepsize
                t += stepsize
        
        samples = xt
        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        
        # Decode frames
        N, C, T, H, W = samples.shape
        samples_frames = samples.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        decoded_frames = vae.decode(samples_frames / 0.18215).sample
        
        # Take middle frame
        mid_frame_idx = T // 2
        middle_frames = decoded_frames.reshape(N, T, 3, decoded_frames.shape[-2], decoded_frames.shape[-1])[:, mid_frame_idx]
        all_frames.append(middle_frames.cpu())
    
    return torch.cat(all_frames, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Validate FID implementation')
    parser.add_argument('--data-path', type=str, default='datasets/ucf-101',
                        help='Path to dataset')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='results/050-EqM-S-4-Linear-velocity-None/checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--model', type=str, default='EqM-S/4',
                        help='Model architecture')
    parser.add_argument('--num-frames', type=int, default=4,
                        help='Number of frames')
    parser.add_argument('--video-duration', type=float, default=2.0,
                        help='Video duration in seconds')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--stepsize', type=float, default=0.003,
                        help='NGD step size')
    parser.add_argument('--mu', type=float, default=0.35,
                        help='NGD momentum')
    parser.add_argument('--num-steps', type=int, default=250,
                        help='Number of sampling steps')
    parser.add_argument('--cfg-scale', type=float, default=4.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--num-real-samples', type=int, default=100,
                        help='Number of real samples for FID')
    parser.add_argument('--num-gen-samples', type=int, default=100,
                        help='Number of generated samples for FID')
    parser.add_argument('--use-bf16', action='store_true',
                        help='Use bfloat16 precision')
    parser.add_argument('--use-rope', action='store_true',
                        help='Use RoPE')
    parser.add_argument('--vae', type=str, default='ema',
                        help='VAE type')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    
    # Load InceptionV3
    print("Loading InceptionV3...")
    from torchvision.models import Inception_V3_Weights
    inception_model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    # Load real data and extract features
    print(f"Loading real data from {args.data_path}...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = VideoDataset(args.data_path, split='train', num_frames=args.num_frames, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Collecting {args.num_real_samples} real frames...")
    real_frames_list = []
    for x, y, time_spans in tqdm(loader, desc="Collecting real frames"):
        if len(real_frames_list) * 8 >= args.num_real_samples:
            break
        # x: (N, C, T, H, W)
        N, C, T, H, W = x.shape
        mid_frame_idx = T // 2
        frames = x[:, :, mid_frame_idx, :, :]  # (N, C, H, W)
        real_frames_list.append(frames)
    
    real_frames = torch.cat(real_frames_list, dim=0)[:args.num_real_samples]
    print(f"Extracting InceptionV3 features from real frames...")
    real_features = get_inception_features(real_frames, inception_model, device=device)
    real_mu, real_sigma = compute_stats(real_features)
    print(f"Real features: shape={real_features.shape}, mu={real_mu[:5]}, sigma_trace={np.trace(real_sigma):.2f}")
    
    # Find checkpoints to evaluate
    checkpoint_dir = Path(args.checkpoint_dir)
    all_ckpts = sorted(checkpoint_dir.glob("*.pt"))
    
    # Select checkpoints: early, middle, and late
    if len(all_ckpts) < 3:
        selected_ckpts = all_ckpts
    else:
        # Select 5 checkpoints evenly spaced
        indices = np.linspace(0, len(all_ckpts) - 1, min(5, len(all_ckpts)), dtype=int)
        selected_ckpts = [all_ckpts[i] for i in indices]
    
    print(f"\nFound {len(all_ckpts)} checkpoints, evaluating {len(selected_ckpts)}:")
    for ckpt in selected_ckpts:
        print(f"  - {ckpt.name}")
    
    # Evaluate each checkpoint
    results = []
    latent_size = args.image_size // 8
    
    for ckpt_path in selected_ckpts:
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint: {ckpt_path.name}")
        print(f"{'='*80}")
        
        # Load model
        print("Loading model...")
        model = EqM_models[args.model](
            input_size=(args.num_frames, latent_size, latent_size),
            num_classes=args.num_classes,
            uncond=True,
            ebm='none',
            use_rope=args.use_rope
        ).to(device)
        
        ema = deepcopy(model).to(device)
        
        # Load checkpoint
        state_dict = find_model(str(ckpt_path))
        if 'ema' in state_dict:
            ema_state = state_dict['ema']
        else:
            ema_state = state_dict
        
        # Handle pos_embed mismatch
        if 'pos_embed_default' in ema_state:
            current_shape = ema.state_dict()['pos_embed_default'].shape
            ckpt_shape = ema_state['pos_embed_default'].shape
            if current_shape != ckpt_shape:
                print(f"Removing pos_embed_default (shape mismatch: {ckpt_shape} vs {current_shape})")
                del ema_state['pos_embed_default']
        
        ema.load_state_dict(ema_state, strict=False)
        ema.eval()
        
        # Generate samples
        time_scale = args.video_duration / (args.num_frames - 1) if args.num_frames > 1 else 0.0
        print(f"Generating {args.num_gen_samples} samples (time_scale={time_scale:.4f})...")
        
        generated_frames = generate_samples_ngd(
            ema, vae, args.num_gen_samples, args.num_frames, latent_size,
            args.num_classes, args.stepsize, args.mu, args.num_steps,
            time_scale, args.cfg_scale, device, args.use_bf16
        )
        
        # Extract features
        print("Extracting InceptionV3 features from generated frames...")
        gen_features = get_inception_features(generated_frames, inception_model, device=device)
        gen_mu, gen_sigma = compute_stats(gen_features)
        
        # Compute FID
        fid_score = compute_fid_from_inception_stats(real_mu, real_sigma, gen_mu, gen_sigma)
        
        print(f"\n{'='*80}")
        print(f"Checkpoint: {ckpt_path.name}")
        print(f"FID Score: {fid_score:.2f}")
        print(f"{'='*80}\n")
        
        results.append({
            'checkpoint': ckpt_path.name,
            'step': int(ckpt_path.stem),
            'fid': fid_score
        })
        
        # Clean up
        del model, ema
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Checkpoint':<20} {'Step':<10} {'FID Score':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['checkpoint']:<20} {r['step']:<10} {r['fid']:<10.2f}")
    print("="*80)
    
    # Check if FID improves over training
    if len(results) >= 2:
        fid_values = [r['fid'] for r in results]
        if fid_values[-1] < fid_values[0]:
            print("✓ VALIDATION PASSED: FID improved from first to last checkpoint")
            print(f"  First: {fid_values[0]:.2f} → Last: {fid_values[-1]:.2f} (Δ = {fid_values[0] - fid_values[-1]:.2f})")
        else:
            print("✗ VALIDATION FAILED: FID did not improve")
            print(f"  First: {fid_values[0]:.2f} → Last: {fid_values[-1]:.2f}")


if __name__ == '__main__':
    main()
