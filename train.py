# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM using PyTorch DDP.
"""
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
from video_dataset import VideoDataset
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
from torch.amp import autocast, GradScaler
import torch.optim.lr_scheduler
from scipy import linalg

try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def compute_fid_from_inception_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Frechet Inception Distance (FID) between two Gaussian distributions.
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small value for numerical stability
    
    Returns:
        FID score (float)
    """
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
def get_inception_features(images, inception_model, batch_size=32):
    """
    Extract InceptionV3 features from images.
    
    Args:
        images: Tensor of images (N, C, H, W) in range [-1, 1]
        inception_model: InceptionV3 model
        batch_size: Batch size for processing
    
    Returns:
        features: (N, 2048) array of features
    """
    inception_model.eval()
    features = []
    
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size]
        # Resize to 299x299 for InceptionV3
        if batch.shape[-1] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize from [-1, 1] to [0, 1] then to ImageNet stats
        batch = (batch + 1) / 2  # [-1, 1] -> [0, 1]
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        
        with torch.no_grad():
            feat = inception_model(batch)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


@torch.no_grad()
def compute_stats(features):
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

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

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ.get("ENTITY")
        project = os.environ.get("PROJECT")
        if args.wandb and entity and project:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if getattr(args, 'video', False):
        input_size = (args.num_frames, latent_size, latent_size)
    else:
        input_size = latent_size
    model = EqM_models[args.model](
        input_size=input_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
        use_rope=args.use_rope
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Setup optimizer
    if args.use_muon:
        if not MUON_AVAILABLE:
            raise ImportError("Muon optimizer not available. Install it with: pip install muon")
        
        logger.info("Using Muon optimizer for hidden weights")
        
        # Categorize parameters for Muon
        # Hidden weights: 2D+ parameters in transformer blocks
        # Hidden gains/biases: 1D parameters in transformer blocks
        # Non-hidden: embeddings, timestep embedder, label embedder, patch embedder (optionally)
        
        hidden_weights = []
        hidden_gains_biases = []
        nonhidden_params = []
        
        # Transformer blocks contain the main hidden weights
        for block in model.blocks:
            for name, param in block.named_parameters():
                if param.ndim >= 2:
                    hidden_weights.append(param)
                else:
                    hidden_gains_biases.append(param)
        
        # Final layer
        for name, param in model.final_layer.named_parameters():
            if param.ndim >= 2:
                hidden_weights.append(param)
            else:
                hidden_gains_biases.append(param)
        
        # Embedders and patch projection
        nonhidden_params.extend(model.t_embedder.parameters())
        nonhidden_params.extend(model.y_embedder.parameters())
        
        # Patch embedder: controlled by --muon-patch-embed flag
        if args.muon_patch_embed:
            for param in model.x_embedder.parameters():
                if param.ndim >= 2:
                    hidden_weights.append(param)
                else:
                    nonhidden_params.append(param)
        else:
            nonhidden_params.extend(model.x_embedder.parameters())
        
        # Positional embeddings (if not using RoPE)
        if hasattr(model, 'pos_embed'):
            nonhidden_params.append(model.pos_embed)
        
        # RoPE parameters (if using RoPE)
        if args.use_rope:
            for block in model.blocks:
                if hasattr(block, 'rope'):
                    nonhidden_params.extend(block.rope.parameters())
        
        logger.info(f"Muon: {len(hidden_weights)} hidden weight tensors (2D+)")
        logger.info(f"AdamW: {len(hidden_gains_biases)} hidden gain/bias tensors (1D)")
        logger.info(f"AdamW: {len(nonhidden_params)} non-hidden parameter tensors")
        
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=args.muon_lr, weight_decay=args.weight_decay),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
        ]
        opt = MuonWithAuxAdam(param_groups)
    else:
        # Standard AdamW optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup mixed precision training
    scaler = GradScaler('cuda', enabled=args.use_amp)
    # Determine the dtype for autocast
    if args.use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16 if args.use_amp else torch.float32

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            model_state = state_dict["model"]
            ema_state = state_dict["ema"]
            
            # For 3D models, remove pos_embed_default if shapes don't match
            # This is safe because pos_embed is dynamically computed via get_pos_embed()
            if 'pos_embed_default' in model_state:
                current_shape = model.state_dict()['pos_embed_default'].shape
                ckpt_shape = model_state['pos_embed_default'].shape
                if current_shape != ckpt_shape:
                    if rank == 0:
                        logger.info(f"INFO: Removing pos_embed_default from checkpoint due to shape mismatch")
                        logger.info(f"      Checkpoint shape: {ckpt_shape}, Current model shape: {current_shape}")
                        logger.info(f"      This is expected when using different num_frames at training vs checkpoint")
                    del model_state['pos_embed_default']
                    del ema_state['pos_embed_default']
            
            model.load_state_dict(model_state, strict=False)
            ema.load_state_dict(ema_state, strict=False)
            opt.load_state_dict(state_dict["opt"])
            scheduler_state = state_dict.get("scheduler")
        else:
            model_state = state_dict
            ema_state = state_dict
            
            # Same for single state dict format
            if 'pos_embed_default' in model_state:
                current_shape = model.state_dict()['pos_embed_default'].shape
                ckpt_shape = model_state['pos_embed_default'].shape
                if current_shape != ckpt_shape:
                    if rank == 0:
                        logger.info(f"INFO: Removing pos_embed_default from checkpoint due to shape mismatch")
                        logger.info(f"      Checkpoint shape: {ckpt_shape}, Current model shape: {current_shape}")
                        logger.info(f"      This is expected when using different num_frames at training vs checkpoint")
                    del model_state['pos_embed_default']
                    
            model.load_state_dict(model_state, strict=False)
            ema.load_state_dict(ema_state, strict=False)
            scheduler_state = state_dict.get("scheduler")

        ema = ema.to(device)
        model = model.to(device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load InceptionV3 for FID computation if enabled
    inception_model = None
    if args.compute_fid and rank == 0:
        logger.info("Loading InceptionV3 for FID computation...")
        from torchvision.models import Inception_V3_Weights
        inception_model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception_model.fc = torch.nn.Identity()  # Remove final classification layer
        inception_model = inception_model.to(device)
        inception_model.eval()
        requires_grad(inception_model, False)

    # Log mixed precision configuration
    if args.use_bf16:
        logger.info("Mixed precision training enabled with BF16")
    elif args.use_amp:
        logger.info("Mixed precision training enabled with FP16")
    else:
        logger.info("Training in FP32 (no mixed precision)")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if getattr(args, 'video', False):
        dataset = VideoDataset(args.data_path, split='train', num_frames=args.num_frames, transform=transform)
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if isinstance(dataset, VideoDataset):
        logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")
    else:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Setup learning rate scheduler
    max_train_steps = args.max_steps if args.max_steps is not None else args.epochs * len(loader)
    if args.lr_schedule == 'cosine':
        import math
        def cosine_schedule(step):
            """Cosine schedule that decays to min_lr_factor of initial LR"""
            progress = step / max_train_steps
            min_factor = args.min_lr_factor
            return min_factor + (1 - min_factor) * (1 + math.cos(math.pi * progress)) / 2
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_schedule)
    elif args.lr_schedule == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=args.min_lr_factor, total_iters=max_train_steps)
    else:
        scheduler = None

    # Load scheduler state if resuming from checkpoint
    if args.ckpt is not None and scheduler is not None and 'scheduler_state' in locals() and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # For FID: accumulate real frame features
    real_features_accumulated = []
    real_features_collected = False  # Flag to collect real features only once

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
        # video latents will have shape (N, C, T, latent, latent)
        zs = torch.randn(n, 4, args.num_frames, latent_size, latent_size, device=device)
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

    logger.info(f"Training for {args.epochs} epochs...")
    max_train_steps = args.max_steps if args.max_steps is not None else args.epochs * len(loader)
    for epoch in tqdm(range(args.epochs), desc="Training", disable=rank != 0, unit="epoch", dynamic_ncols=True):
        if train_steps >= max_train_steps:
            break
        sampler.set_epoch(epoch)
        # logger.info(f"Beginning epoch {epoch}...")
        for batch in tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0, leave=False, unit="batch", dynamic_ncols=True):
            if train_steps >= max_train_steps:
                break
            if getattr(args, 'video', False):
                x, y, time_spans = batch
                # x: (N, C, T, H, W), time_spans: (N,) in seconds
                N, C, T, H, W = x.shape
                # encode frames by flattening N*T into batch for the VAE
                x_frames = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W).to(device)
                y = torch.as_tensor(y, device=device, dtype=torch.long)
                time_spans = torch.as_tensor(time_spans, device=device, dtype=torch.float32)
                # Compute time_scale: seconds per frame
                time_scale = (time_spans / (T - 1)).mean().item()  # Average across batch
                with torch.no_grad():
                    lat = vae.encode(x_frames).latent_dist.sample().mul_(0.18215)
                # reshape latents back to (N, C_latent, T, latent_H, latent_W)
                C_lat = lat.shape[1]
                lat = lat.reshape(N, T, C_lat, lat.shape[-2], lat.shape[-1]).permute(0, 2, 1, 3, 4)
                x = lat
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                time_scale = 1.0  # No time scaling for 2D images
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y, time_scale=time_scale, return_act=args.disp, train=True)

            # Use automatic mixed precision if enabled
            with autocast(device_type='cuda', dtype=autocast_dtype, enabled=args.use_amp or args.use_bf16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model.module)
            if scheduler:
                scheduler.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Generate samples:
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Generating samples at step {train_steps}...")
                    ema.eval()
                    
                    # Collect real features for FID (only once)
                    if args.compute_fid and not real_features_collected:
                        logger.info("Collecting real frame features for FID...")
                        real_frames_list = []
                        frame_count = 0
                        target_samples = min(args.fid_num_samples, len(dataset))
                        
                        # Collect frames from real data
                        for batch_data in loader:
                            if frame_count >= target_samples:
                                break
                            
                            if getattr(args, 'video', False):
                                x_real, _, _ = batch_data
                                # x_real: (N, C, T, H, W)
                                N, C, T, H, W = x_real.shape
                                # Take middle frame from each video
                                mid_frame_idx = T // 2
                                frames = x_real[:, :, mid_frame_idx, :, :]  # (N, C, H, W)
                            else:
                                frames, _ = batch_data
                            
                            frames = frames.to(device)
                            real_frames_list.append(frames)
                            frame_count += frames.shape[0]
                        
                        real_frames = torch.cat(real_frames_list, dim=0)[:target_samples]
                        real_features = get_inception_features(real_frames, inception_model, batch_size=32)
                        real_features_accumulated = real_features
                        real_features_collected = True
                        logger.info(f"Collected {real_features.shape[0]} real frame features")
                    
                    with torch.no_grad():
                        # Compute time_scale for video (temporal spacing between frames)
                        if getattr(args, 'video', False):
                            # Calculate time_scale from video duration
                            # e.g., 2.0s duration with 4 frames = 2.0/3 = 0.667 s/frame
                            time_scale_sample = args.sample_video_duration / (args.num_frames - 1) if args.num_frames > 1 else 0.0
                            logger.info(f"Sampling with time_scale={time_scale_sample:.4f} s/frame ({args.sample_video_duration}s over {args.num_frames} frames)")
                        else:
                            time_scale_sample = 1.0
                        
                        # Generate multiple batches for FID if needed
                        generated_frames_list = []
                        num_fid_batches = (args.fid_num_samples + local_batch_size - 1) // local_batch_size if args.compute_fid else 1
                        
                        for fid_batch_idx in range(num_fid_batches):
                            # Initialize for gradient descent sampling
                            if getattr(args, 'video', False):
                                zs_batch = torch.randn(local_batch_size, 4, args.num_frames, latent_size, latent_size, device=device)
                            else:
                                zs_batch = torch.randn(local_batch_size, 4, latent_size, latent_size, device=device)
                            
                            ys_batch = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
                            
                            if use_cfg:
                                zs_batch = torch.cat([zs_batch, zs_batch], 0)
                                y_null_batch = torch.tensor([args.num_classes] * local_batch_size, device=device)
                                ys_batch = torch.cat([ys_batch, y_null_batch], 0)
                            
                            xt = zs_batch.clone()
                            t = torch.ones((xt.shape[0],)).to(xt).to(device)
                            m = torch.zeros_like(xt).to(xt).to(device)
                            
                            if use_cfg:
                                sample_kwargs = dict(y=ys_batch, cfg_scale=args.cfg_scale, time_scale=time_scale_sample)
                            else:
                                sample_kwargs = dict(y=ys_batch, time_scale=time_scale_sample)
                            
                            # Use autocast for sampling
                            with autocast(device_type='cuda', dtype=autocast_dtype, enabled=args.use_amp or args.use_bf16):
                                # Gradient descent sampling loop
                                if args.sample_method == 'ngd':
                                    # Nesterov Accelerated Gradient Descent
                                    for _ in range(args.num_sample_steps - 1):
                                        x_ = xt + args.sample_stepsize * m * args.sample_mu
                                        out = model_fn(x_, t, **sample_kwargs)
                                        if not torch.is_tensor(out):
                                            out = out[0]
                                        m = out
                                        xt = xt + out * args.sample_stepsize
                                        t += args.sample_stepsize
                                elif args.sample_method == 'gd':
                                    # Standard Gradient Descent
                                    for _ in range(args.num_sample_steps - 1):
                                        out = model_fn(xt, t, **sample_kwargs)
                                        if not torch.is_tensor(out):
                                            out = out[0]
                                        xt = xt + out * args.sample_stepsize
                                        t += args.sample_stepsize
                                else:  # 'ode'
                                    # Use ODE sampler (original method)
                                    sample_fn = transport_sampler.sample_ode(
                                        sampling_method="dopri5",
                                        num_steps=args.num_sample_steps,
                                    )
                                    xt = sample_fn(zs_batch, model_fn, **sample_kwargs)[-1]
                            
                            samples = xt
                            if use_cfg:
                                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                            
                            # Decode samples from latent space
                            if getattr(args, 'video', False):
                                # For video: decode frames
                                N, C, T, H, W = samples.shape
                                samples_frames = samples.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
                                decoded_frames = vae.decode(samples_frames / 0.18215).sample
                                # Take middle frame from each video for visualization
                                mid_frame_idx = T // 2
                                samples_to_log = decoded_frames.reshape(N, T, 3, decoded_frames.shape[-2], decoded_frames.shape[-1])[:, mid_frame_idx]
                            else:
                                samples_to_log = vae.decode(samples / 0.18215).sample
                            
                            generated_frames_list.append(samples_to_log)
                            
                            # Log first batch to wandb for visualization
                            if fid_batch_idx == 0 and args.wandb:
                                wandb_utils.log_image(samples_to_log, step=train_steps)
                        
                        # Compute FID if enabled
                        if args.compute_fid and real_features_collected:
                            logger.info("Computing FID...")
                            generated_frames = torch.cat(generated_frames_list, dim=0)[:args.fid_num_samples]
                            generated_features = get_inception_features(generated_frames, inception_model, batch_size=32)
                            
                            # Compute statistics
                            real_mu, real_sigma = compute_stats(real_features_accumulated)
                            gen_mu, gen_sigma = compute_stats(generated_features)
                            
                            # Compute FID
                            fid_score = compute_fid_from_inception_stats(real_mu, real_sigma, gen_mu, gen_sigma)
                            logger.info(f"FID Score: {fid_score:.2f}")
                            
                            if args.wandb:
                                wandb_utils.log({"fid": fid_score}, step=train_steps)
                        
                        logger.info(f"Samples generated and logged to wandb")
                dist.barrier()

            # Save EqM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--sample-every", type=int, default=5000, help="Generate samples every N steps (0 to disable)")
    parser.add_argument("--num-sample-steps", type=int, default=250, help="Number of sampling steps for visualization")
    parser.add_argument("--sample-stepsize", type=float, default=0.003, help="Step size for gradient descent sampling")
    parser.add_argument("--sample-mu", type=float, default=0.35, help="Momentum parameter for NGD sampling")
    parser.add_argument("--sample-method", type=str, default="ngd", choices=["gd", "ngd", "ode"], help="Sampling method for visualization")
    parser.add_argument("--sample-video-duration", type=float, default=2.0, help="Video duration for sampling (used to compute time_scale)")
    parser.add_argument("--compute-fid", action="store_true", help="Compute frame-wise FID during sampling")
    parser.add_argument("--fid-num-samples", type=int, default=50, help="Number of videos/images to generate for FID computation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear", "cosine"], default="constant", help="Learning rate schedule")
    parser.add_argument("--min-lr-factor", type=float, default=0.1, help="Minimum learning rate as a factor of initial LR (for cosine/linear schedules)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to an EqM checkpoint to continue training from")
    parser.add_argument("--disp", action="store_true",
                        help="Toggle to enable Dispersive Loss")
    parser.add_argument("--uncond", type=bool, default=True,
                        help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="energy formulation")
    parser.add_argument("--video", action="store_true", help="Enable video training mode")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames per video clip")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training (FP16)")
    parser.add_argument("--use-bf16", action="store_true", help="Enable bfloat16 mixed precision training (BF16)")
    parser.add_argument("--use-rope", action="store_true", help="Use Rotary Position Embedding (RoPE) instead of fixed sinusoidal embeddings. Allows training with varying number of frames.")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer for hidden weights (2D+ params in transformer blocks)")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon optimizer")
    parser.add_argument("--muon-patch-embed", action="store_true", help="Apply Muon to patch embedding projection layer (experimental)")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
