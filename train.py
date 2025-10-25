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
import signal
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
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
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
    class TqdmLoggingHandler(logging.Handler):
        """Logging handler that uses tqdm.write to avoid breaking progress bars."""
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                # Use tqdm.write which is safe with tqdm progress bars
                from tqdm import tqdm
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    logger = logging.getLogger(__name__)
    # If rank 0, attach a tqdm-safe console handler and a file handler
    if dist.get_rank() == 0:
        logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicate logs when reusing logger
        logger.handlers = []

        console_handler = TqdmLoggingHandler()
        # Colored timestamp for console (blue) similar to previous behavior
        colored_fmt = '\033[34m%(asctime)s\033[0m %(message)s'
        console_formatter = logging.Formatter(f'[{colored_fmt}', datefmt='%Y-%m-%d %H:%M:%S')
        # The console formatter should wrap the timestamp in brackets like before
        # e.g., [[34m2025-10-23 12:34:56[0m] message
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (if a logging_dir was provided) - use plain formatting
        if logging_dir is not None:
            try:
                file_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                fh = logging.FileHandler(f"{logging_dir}/log.txt")
                fh.setFormatter(file_formatter)
                logger.addHandler(fh)
            except Exception:
                # If file handler cannot be created, log to console only
                logger.warning(f"Could not create log file at {logging_dir}/log.txt")
    else:
        # Non-zero ranks get a NullHandler to avoid noisy logging
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
    import warnings
    warnings.filterwarnings('ignore', message='.*barrier.*using the device under current context.*')
    dist.init_process_group("nccl")

    # Calculate effective batch size with gradient accumulation
    effective_batch_size = args.global_batch_size * args.gradient_accumulation_steps
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    if rank == 0:
        print(f"Global batch size: {args.global_batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}, Local batch size per GPU: {local_batch_size}")

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

    # If a checkpoint is provided, try to read its saved args to match num_classes/uncond/etc.
    if args.ckpt is not None:
        try:
            ck_meta = find_model(args.ckpt)
            if isinstance(ck_meta, dict):
                # If checkpoint saved args, use them to override CLI where appropriate
                if 'args' in ck_meta:
                    ck_args = ck_meta['args']
                    try:
                        if hasattr(ck_args, 'uncond'):
                            args.uncond = bool(getattr(ck_args, 'uncond'))
                            print(f"Overriding --uncond from checkpoint: {args.uncond}")
                        if hasattr(ck_args, 'num_classes'):
                            args.num_classes = int(getattr(ck_args, 'num_classes'))
                            print(f"Overriding --num-classes from checkpoint: {args.num_classes}")
                    except Exception:
                        pass
                else:
                    # Fallback: try to infer class count from saved embedding table
                    model_state = ck_meta.get('model', None)
                    if isinstance(model_state, dict):
                        for k, v in model_state.items():
                            if k.endswith('y_embedder.embedding_table.weight'):
                                M = v.shape[0]
                                # Determine whether checkpoint used an unconditional/CFG token
                                use_uncond = False
                                try:
                                    ck_args = ck_meta.get('args', None)
                                    if ck_args is not None and hasattr(ck_args, 'uncond'):
                                        use_uncond = bool(getattr(ck_args, 'uncond'))
                                except Exception:
                                    use_uncond = args.uncond
                                try:
                                    if use_uncond and M > 1:
                                        args.num_classes = int(M - 1)
                                    else:
                                        args.num_classes = int(M)
                                    print(f"Inferred --num-classes={args.num_classes} from checkpoint embedding size={M} (use_uncond={use_uncond})")
                                except Exception:
                                    pass
                                break
        except Exception as e:
            print(f"Warning: failed to read checkpoint metadata: {e}")

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

    # Apply torch.compile if requested (PyTorch 2.0+)
    if args.use_compile:
        # Skip compile - it's not working and i'll figure it out later
        # TODO: Fix torch.compile usage
        logger.warning("torch.compile() not working, skipping compilation")
        # if hasattr(torch, 'compile'):
        #     logger.info("Compiling model with torch.compile()...")
        #     model = torch.compile(model)
        # else:
        #     logger.warning("torch.compile() not available (requires PyTorch 2.0+), skipping compilation")

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

    # Custom collate function to handle variable-length video sequences
    # Determine the minimum temporal frames required by the model's patch embedding
    try:
        _x_embed = model.module.x_embedder if hasattr(model, 'module') else model.x_embedder
        # temporal patch size (pt): how many raw frames map to one temporal patch
        pt = int(_x_embed.patch_size[0])
        min_required_frames = pt
    except Exception:
        # Fallback: require at least 1 temporal frame / patch
        pt = 1
        min_required_frames = 1

    def video_collate_fn(batch):
        """Collate function for variable-length video sequences.

        Returns:
            videos_batch: (N, C, T_max, H, W) tensor where T_max is the
                maximum target-frames among samples in this batch.
            labels_batch, time_spans_batch, num_real_frames_batch: tensors
                preserving the true (non-padded) frame counts per sample.
        """
        videos, labels, time_spans = zip(*batch)

        # Track number of real (non-padded) frames for each video
        num_real_frames = [v.shape[1] for v in videos]  # videos are (C, T, H, W)

        # Use the maximum target-frames present in this batch (not args.num_frames).
        # Ensure it's at least the model's temporal patch size and pad to the
        # nearest multiple of the temporal patch size (pt) so Conv3d patching
        # produces an integer number of temporal patches.
        batch_max_frames = max(max(num_real_frames), min_required_frames)
        # Ceil to nearest multiple of pt
        if pt > 1 and (batch_max_frames % pt) != 0:
            batch_max_frames = ((batch_max_frames + pt - 1) // pt) * pt

        # Pad all videos to batch_max_frames by repeating the last frame only when needed
        padded_videos = []
        for video in videos:
            C, T, H, W = video.shape
            if T < batch_max_frames:
                # Repeat last frame to reach batch_max_frames
                last_frame = video[:, -1:, :, :]  # (C, 1, H, W)
                padding = last_frame.repeat(1, batch_max_frames - T, 1, 1)
                video = torch.cat([video, padding], dim=1)  # (C, batch_max_frames, H, W)
            padded_videos.append(video)

        # Stack into batch
        videos_batch = torch.stack(padded_videos, dim=0)  # (N, C, T, H, W)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        time_spans_batch = torch.tensor(time_spans, dtype=torch.float32)
        num_real_frames_batch = torch.tensor(num_real_frames, dtype=torch.long)

        return videos_batch, labels_batch, time_spans_batch, num_real_frames_batch

    if getattr(args, 'video', False):
        # Create clip sampler if enabled
        clip_sampler = None
        if getattr(args, 'use_clip_sampling', False):
            from curriculum_sampler import TemporalClipSampler
            clip_sampler = TemporalClipSampler(
                clip_prob=args.clip_full_prob,
                min_fraction=0.2,  # 20% of video duration
                max_fraction=0.8   # 80% of video duration
            )

        dataset = VideoDataset(
            args.data_path, 
            split='train', 
            num_frames=args.num_frames, 
            transform=transform,
            random_frames=args.random_frames,
            min_frames=args.min_frames,
            single_frame_prob=args.single_frame_prob,
            clip_sampler=clip_sampler,
            clips_per_video=args.clips_per_video
        )
    else:
        dataset = ImageFolder(args.data_path, transform=transform)

    # Compute per-epoch batches and max training steps BEFORE creating the sampler/loader
    # This lets us compute curriculum phase lengths deterministically and create the sampler once.
    dataset_size = len(dataset)
    # Global batches per epoch (using global batch size)
    import math
    global_batches_per_epoch = math.ceil(dataset_size / float(args.global_batch_size))
    # Batches per GPU/process per epoch
    per_gpu_batches_per_epoch = math.ceil(global_batches_per_epoch / float(dist.get_world_size()))
    # Compute total training steps we will run (per-GPU steps)
    max_train_steps = args.max_steps if args.max_steps is not None else int(args.epochs * per_gpu_batches_per_epoch)

    if isinstance(dataset, VideoDataset):
        logger.info(f"Dataset contains {len(dataset.video_paths):,} videos ({args.data_path})")
        if args.clips_per_video > 1:
            logger.info(f"Augmenting with {args.clips_per_video} clips per video: {dataset_size:,} total samples")
        if args.random_frames:
            logger.info(f"Random frame sampling enabled: {args.min_frames}-{args.num_frames} frames per clip")
        if args.single_frame_prob > 0:
            logger.info(f"Single-frame probability: {args.single_frame_prob:.2%} (image augmentation)")
    else:
        logger.info(f"Dataset contains {dataset_size:,} images ({args.data_path})")

    # Create sampler (curriculum if enabled, otherwise standard distributed)
    if args.use_curriculum and isinstance(dataset, VideoDataset):
        from curriculum_sampler import CurriculumTemporalSampler

        # Compute number of curriculum phases from local batch size by halving until 1
        num_phases = 0
        tmp_bs = local_batch_size
        while True:
            num_phases += 1
            if tmp_bs == 1:
                break
            tmp_bs = max(1, tmp_bs // 2)

        computed_epochs_per_phase = max(1, args.epochs // max(1, num_phases))

        sampler = CurriculumTemporalSampler(
            dataset,
            batch_size=local_batch_size,
            max_frames=args.num_frames,
            world_size=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.global_seed,
            epochs_per_phase=computed_epochs_per_phase,
            logger=logger,
        )
        logger.info(f"Curriculum learning: {len(sampler.curriculum_schedule)} phases, {computed_epochs_per_phase} epochs/phase, {args.epochs} total epochs")
        try:
            sampler.describe(world_size=dist.get_world_size())
        except Exception:
            logger.info("(Failed to print curriculum description)")
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )

    # Create DataLoader using the sampler we just created
    # Note: CurriculumTemporalSampler yields full batches of indices (i.e., acts as a batch_sampler).
    # When using it we must pass it as `batch_sampler` to DataLoader and not provide `batch_size` or `sampler`.
    # If the sampler implements curriculum behaviour it yields full batches of indices
    # (i.e., acts as a batch_sampler). Detect this by checking for the sampler API
    # method `get_current_batch_size` which CurriculumTemporalSampler provides.
    if getattr(args, 'use_curriculum', False) and hasattr(sampler, 'get_current_batch_size'):
        # IMPORTANT: Do not use persistent_workers when curriculum updates dataset
        # attributes at runtime (e.g., set_frame_range). Persistent workers keep
        # copies of the Dataset in worker processes, so updates made in the main
        # process won't be visible to them. Disable persistent_workers so workers
        # are recreated and pick up updated frame ranges when the sampler calls
        # update_dataset_for_phase().
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=video_collate_fn if getattr(args, 'video', False) else None,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=video_collate_fn if getattr(args, 'video', False) else None
        )

    # Setup learning rate scheduler
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

    # Setup signal handler for graceful shutdown on Ctrl+C
    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # For FID: accumulate real frame features
    real_features_accumulated = []
    real_features_collected = False  # Flag to collect real features only once

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    accumulation_counter = 0  # Track gradient accumulation steps

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
    # Determine max_train_steps. If max_steps provided, use it. Otherwise, compute a
    # deterministic total by summing the number of batches that the sampler will yield
    # for each epoch. This is important when using CurriculumTemporalSampler because
    # the per-epoch batch count can change by phase; summing len(sampler) across epochs
    # produces the true number of optimizer steps the run will execute.
    if args.max_steps is not None:
        max_train_steps = args.max_steps
    else:
        if args.use_curriculum and hasattr(sampler, '__len__'):
            # Save current epoch to restore later
            try:
                saved_epoch = sampler.epoch
            except Exception:
                saved_epoch = None

            total_steps = 0
            for e in range(args.epochs):
                sampler.set_epoch(e)
                # Update dataset/frame-range for this phase so sampler length is correct
                if hasattr(sampler, 'update_dataset_for_phase'):
                    sampler.update_dataset_for_phase()
                try:
                    total_steps += len(sampler)
                except Exception:
                    # Fallback: estimate using per_gpu_batches_per_epoch
                    total_steps += per_gpu_batches_per_epoch

            # Restore sampler epoch
            if saved_epoch is not None:
                sampler.set_epoch(saved_epoch)
                if hasattr(sampler, 'update_dataset_for_phase'):
                    sampler.update_dataset_for_phase()

            max_train_steps = int(total_steps)
        else:
            # Non-curriculum: use stable estimate based on dataset_size/global batch size
            max_train_steps = args.epochs * per_gpu_batches_per_epoch

    # Initialize optimizer gradients for gradient accumulation
    opt.zero_grad()

    # Persist previous phase key across epochs so we only log on the first epoch
    # (prev_phase_key is None -> first time) and when the phase actually changes.
    prev_phase_key = None
    phase_logged_batches = 0

    # Track the most recently computed gradient norm (for logging / wandb)
    last_grad_norm = None

    for epoch in tqdm(range(args.epochs), desc="Training", disable=rank != 0, unit="epoch", dynamic_ncols=True):
        if train_steps >= max_train_steps or interrupted:
            break
        sampler.set_epoch(epoch)
        # Update dataset frame sampling for current curriculum phase
        if args.use_curriculum and isinstance(dataset, VideoDataset):
            sampler.update_dataset_for_phase()

        # logger.info(f"Beginning epoch {epoch}...")
        # Use a stable per-epoch batch count for the progress bar when using curriculum.
        # Curriculum phases can change the sampler's batch_size (per-GPU), which changes
        # len(loader) for that epoch. To avoid the progress bar jumping (e.g., 110 -> 220),
        # use the precomputed `per_gpu_batches_per_epoch` as the total when curriculum is enabled.
        # After updating the sampler for the current phase, ask the sampler for its
        # current length. This is the authoritative number of batches that will be
        # yielded this epoch. Fall back to the precomputed per_gpu_batches_per_epoch
        # estimate only if the sampler doesn't implement __len__.
        try:
            epoch_total = len(sampler) if hasattr(sampler, '__len__') else int(per_gpu_batches_per_epoch)
        except Exception:
            # As a last resort, try len(loader) (works for non-batch_sampler case)
            try:
                epoch_total = len(loader)
            except Exception:
                epoch_total = int(per_gpu_batches_per_epoch)

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", disable=rank != 0, leave=False, unit="batch", dynamic_ncols=True, total=epoch_total)):
            if train_steps >= max_train_steps or interrupted:
                break
            if getattr(args, 'video', False):
                x, y, time_spans, num_real_frames = batch
                # x: (N, C, T, H, W), time_spans: (N,) in seconds, num_real_frames: (N,) frame counts
                N, C, T, H, W = x.shape
                # encode frames by flattening N*T into batch for the VAE
                x_frames = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W).to(device)
                y = torch.as_tensor(y, device=device, dtype=torch.long)
                time_spans = torch.as_tensor(time_spans, device=device, dtype=torch.float32)
                num_real_frames = torch.as_tensor(num_real_frames, device=device, dtype=torch.long)
                # Compute time_scale as seconds per frame (frame duration).
                # To scale RoPE to the actual clip length, use duration / num_frames
                # so that e.g. 4 frames over 2s => 0.5s per frame. For zero/negative
                # duration or zero frames, use 0.
                time_scale = torch.where(
                    (num_real_frames > 0) & (time_spans > 0),
                    time_spans / num_real_frames.float(),
                    torch.zeros_like(time_spans)
                )
                # Number of temporal patches for the model's masking logic
                num_real_patches = (num_real_frames + pt - 1) // pt

                # --- Telemetry: log the first few batches when a curriculum phase changes ---
                try:
                    # Determine current phase key from dataset and sampler (min_frames, num_frames, batch_size)
                    current_min = getattr(dataset, 'min_frames', None)
                    current_num = getattr(dataset, 'num_frames', None)
                    current_bs = sampler.get_current_batch_size() if hasattr(sampler, 'get_current_batch_size') else local_batch_size
                    current_phase_key = (int(current_min) if current_min is not None else None,
                                         int(current_num) if current_num is not None else None,
                                         int(current_bs))
                except Exception:
                    current_phase_key = None

                if rank == 0:
                    # If phase changed, reset counter and log phase header
                    if current_phase_key != prev_phase_key:
                        prev_phase_key = current_phase_key
                        phase_logged_batches = 0
                        logger.info(f"Curriculum phase changed: min_frames={current_phase_key[0]}, num_frames={current_phase_key[1]}, batch_size_per_gpu={current_phase_key[2]}")

                    # Log detailed batch telemetry for the first few batches of this phase
                    if phase_logged_batches < 5:
                        # Compute padded T for this batch and tokens
                        raw_frames = num_real_frames.cpu().numpy()
                        patch_counts = ((raw_frames + pt - 1) // pt)
                        # spatial grid from patch embed (t,h,w) -> h,w
                        try:
                            spatial = _x_embed.spatial_grid_size
                            h_grid, w_grid = int(spatial[0]), int(spatial[1])
                        except Exception:
                            h_grid, w_grid = 1, 1
                        tokens_per_sample = patch_counts * h_grid * w_grid
                        avg_tokens = float(tokens_per_sample.mean())
                        tokens_per_step = avg_tokens * float(current_bs)
                        max_raw = int(raw_frames.max())
                        padded_to = int(((max_raw + pt - 1) // pt) * pt)
                        logger.debug(f" batch={batch_idx}: raw_frames_min={int(raw_frames.min())}, raw_frames_max={max_raw}, padded_to={padded_to}, pt={pt}, patches_min={int(patch_counts.min())}, patches_max={int(patch_counts.max())}, avg_tokens/sample={avg_tokens:.1f}, tokens/step~{tokens_per_step:.1f}")
                        phase_logged_batches += 1
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
                # No time scaling for 2D images - create tensor of ones
                time_scale = torch.ones(x.shape[0], device=device, dtype=torch.float32)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Pass number of real RAW FRAMES to the model (used for masking). The
            # model's masking helper (`create_temporal_mask`) expects raw frame
            # counts and will internally convert to patch counts. Previously we
            # passed already-converted patch counts which caused only the first
            # temporal patch to be marked valid (effectively freezing later
            # frames). Use `num_real_frames` here to ensure correct masking.
            model_kwargs = dict(
                y=y,
                time_scale=time_scale,
                return_act=args.disp,
                train=True,
                num_real_frames=num_real_frames if getattr(args, 'video', False) else None
            )

            # Use automatic mixed precision if enabled
            with autocast(device_type='cuda', dtype=autocast_dtype, enabled=args.use_amp or args.use_bf16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / args.gradient_accumulation_steps

            # Accumulate gradients
            scaler.scale(loss).backward()
            accumulation_counter += 1

            # Only step optimizer after accumulating enough gradients
            if accumulation_counter >= args.gradient_accumulation_steps:
                # --- Gradient Clipping & Norm Logging ---
                # If AMP is enabled and grads are scaled, unscale before measuring/clipping.
                if scaler.is_enabled():
                    scaler.unscale_(opt)

                # Compute global gradient L2 norm (over all parameters)
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += float(param_norm.item()) ** 2
                last_grad_norm = float(total_norm_sq**0.5)

                if args.grad_clip > 0:
                    # Clip gradients (operates on unscaled grads if AMP)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                accumulation_counter = 0

                # Update EMA and scheduler only after actual optimizer step
                update_ema(ema, model.module)
                if scheduler:
                    scheduler.step()

                train_steps += 1

            # Log loss values (use unscaled loss for logging)
            running_loss += loss.item() * args.gradient_accumulation_steps
            log_steps += 1

            # Generate samples (check before logging to ensure correct step order):
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Generating samples at step {train_steps}...")
                    ema.eval()

                    # Collect real features for FID (only once)
                    if args.compute_fid and not real_features_collected:
                        logger.info("Collecting real frame features for FID...")
                        real_features_list = []
                        frame_count = 0
                        target_samples = min(args.fid_num_samples, len(dataset))

                        # Process frames in chunks to reduce peak VRAM usage
                        for batch_data in loader:
                            if frame_count >= target_samples:
                                break

                            # Batch returned by DataLoader may be a tuple/list with
                            # different lengths depending on dataset/sampler configuration.
                            # For video mode the collate function returns either
                            # (videos, labels, time_spans) or (videos, labels, time_spans, num_real_frames).
                            # For image mode it may be (images, labels) or include extra metadata.
                            # To be robust, always take the first element as the input frames/videos.
                            if isinstance(batch_data, (list, tuple)):
                                input0 = batch_data[0]
                            else:
                                input0 = batch_data

                            if getattr(args, 'video', False):
                                x_real = input0
                                # x_real: (N, C, T, H, W)
                                N, C, T, H, W = x_real.shape
                                # Take middle frame from each video
                                mid_frame_idx = T // 2
                                frames = x_real[:, :, mid_frame_idx, :, :]  # (N, C, H, W)
                            else:
                                frames = input0

                            # Process frames immediately and extract features
                            # This avoids accumulating frames in memory
                            frames = frames.to(device)
                            batch_features = get_inception_features(frames, inception_model, batch_size=32)
                            real_features_list.append(batch_features)
                            frame_count += frames.shape[0]

                            # Free memory immediately
                            del frames
                            torch.cuda.empty_cache()

                        # Concatenate all features (much smaller than raw frames)
                        real_features_accumulated = np.concatenate(real_features_list, axis=0)[:target_samples]
                        real_features_collected = True
                        logger.info(f"Collected {real_features_accumulated.shape[0]} real frame features")
                        del real_features_list
                        torch.cuda.empty_cache()

                    with torch.no_grad():
                        # Compute time_scale for video (temporal spacing between frames)
                        if getattr(args, 'video', False):
                            # Use seconds-per-frame (duration / num_frames) to match training's
                            # time_scale definition. Guard against zero/negative duration or
                            # zero frames.
                            if args.sample_video_duration > 0 and args.num_frames > 0:
                                time_scale_scalar = args.sample_video_duration / float(args.num_frames)
                            else:
                                time_scale_scalar = 0.0
                            logger.info(f"Sampling with time_scale={time_scale_scalar:.6f} s/frame ({args.num_frames} frames over {args.sample_video_duration}s, pt={pt})")
                        else:
                            time_scale_scalar = 1.0

                        # Generate multiple batches for FID if needed
                        generated_frames_list = []
                        num_fid_batches = (args.fid_num_samples + local_batch_size - 1) // local_batch_size if args.compute_fid else 1

                        from tqdm import tqdm as _tqdm  # local alias to avoid shadowing outer tqdm usage
                        # Collect the first video from the first few fid batches so we can
                        # assemble a 4x4 logged grid (4 videos x 4 frames). We gather the
                        # first video (index 0) from up to `target_videos` batches below.
                        collected_videos = []
                        target_videos = 4
                        target_frames = 4

                        for fid_batch_idx in _tqdm(range(num_fid_batches),
                                                   desc="FID batches" if rank == 0 else None,
                                                   disable=(rank != 0),
                                                   dynamic_ncols=True):
                            # Initialize for gradient descent sampling
                            if getattr(args, 'video', False):
                                zs_batch = torch.randn(local_batch_size, 4, args.num_frames, latent_size, latent_size, device=device) * 0.18215
                            else:
                                zs_batch = torch.randn(local_batch_size, 4, latent_size, latent_size, device=device) * 0.18215

                            ys_batch = torch.randint(args.num_classes, size=(local_batch_size,), device=device)

                            if use_cfg:
                                zs_batch = torch.cat([zs_batch, zs_batch], 0)
                                y_null_batch = torch.tensor([args.num_classes] * local_batch_size, device=device)
                                ys_batch = torch.cat([ys_batch, y_null_batch], 0)

                            xt = zs_batch.clone()
                            t = torch.ones((xt.shape[0],)).to(xt).to(device)
                            m = torch.zeros_like(xt).to(xt).to(device)

                            # Create time_scale tensor for this batch
                            batch_size_with_cfg = xt.shape[0]
                            time_scale_sample = torch.full((batch_size_with_cfg,), time_scale_scalar, device=device, dtype=torch.float32)

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
                                # For video: decode all frames and prepare flattened labeled images
                                N, C, T, H, W = samples.shape
                                samples_frames = samples.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
                                decoded_frames = vae.decode(samples_frames / 0.18215).sample

                                # decoded_frames: (N*T, 3, H, W)
                                # Reshape to (N, T, 3, H, W) so we can build labels per-frame
                                decoded_frames = decoded_frames.reshape(N, T, 3, decoded_frames.shape[-2], decoded_frames.shape[-1])

                                # Flatten into order: video0 frames (0..T-1), video1 frames, ...
                                samples_to_log = decoded_frames.reshape(N * T, 3, decoded_frames.shape[-2], decoded_frames.shape[-1])

                                # Record metadata for the first batch so we can build labels later
                                # We prefer to build labels in the helper (timestamps etc.)
                                first_batch_video_T = T
                                # Prefer explicit sample duration if available; fallback to None
                                try:
                                    first_batch_video_duration = args.sample_video_duration
                                except Exception:
                                    first_batch_video_duration = None
                            else:
                                samples_to_log = vae.decode(samples / 0.18215).sample

                            # Collect the first video from this fid batch (if video mode)
                            if getattr(args, 'video', False):
                                # decoded_frames: (N, T, 3, H, W)
                                try:
                                    # take as many videos from this batch as needed (in-order)
                                    need = target_videos - len(collected_videos)
                                    if need > 0:
                                        # clamp to available videos in this batch
                                        take = min(need, decoded_frames.shape[0])
                                        for vi in range(take):
                                            collected_videos.append(decoded_frames[vi].cpu().clone())
                                except Exception:
                                    # ignore collection failures
                                    pass
                            else:
                                # For images, keep the first fid batch for logging as before
                                if fid_batch_idx == 0:
                                    first_batch_samples = samples_to_log.clone()

                            # For FID: extract features immediately to save VRAM
                            if args.compute_fid:
                                batch_features = get_inception_features(samples_to_log, inception_model, batch_size=32)
                                generated_frames_list.append(batch_features)
                            else:
                                generated_frames_list.append(samples_to_log)

                            # Free memory
                            del samples, samples_to_log
                            if getattr(args, 'video', False):
                                del samples_frames, decoded_frames
                            torch.cuda.empty_cache()

                        # After collecting videos from fid batches, assemble the first_batch_samples
                        # as the first video from the first `target_videos` batches (one row per video,
                        # `target_frames` columns). If needed, pad temporally or repeat videos to reach
                        # the target shape.
                        if getattr(args, 'video', False) and len(collected_videos) > 0:
                            # Ensure we have at least target_videos by repeating collected videos
                            while len(collected_videos) < target_videos:
                                collected_videos.extend(collected_videos[: (target_videos - len(collected_videos))])

                            # Prepare selected videos: ensure each has exactly target_frames frames
                            sel_list = []
                            for vid in collected_videos[:target_videos]:
                                # vid shape: (T_vid, 3, H, W)
                                T_vid = vid.shape[0]
                                if T_vid < target_frames:
                                    last = vid[-1:].repeat(target_frames - T_vid, 1, 1, 1)
                                    vid_padded = torch.cat([vid, last], dim=0)
                                else:
                                    vid_padded = vid[:target_frames]
                                # now vid_padded: (target_frames, 3, H, W)
                                sel_list.append(vid_padded.unsqueeze(0))

                            sel = torch.cat(sel_list, dim=0)  # (target_videos, target_frames, 3, H, W)
                            # Flatten to (target_videos * target_frames, 3, H, W)
                            first_batch_samples = sel.reshape(target_videos * target_frames, sel.shape[2], sel.shape[3], sel.shape[4]).clone()
                            first_batch_video_T = target_frames
                            first_batch_video_duration = getattr(args, 'sample_video_duration', None)

                        # Compute FID if enabled
                        fid_score = None
                        if args.compute_fid and real_features_collected:
                            logger.info("Computing FID...")
                            # Features were already extracted in the loop above
                            generated_features = np.concatenate(generated_frames_list, axis=0)[:args.fid_num_samples]

                            # Compute statistics
                            real_mu, real_sigma = compute_stats(real_features_accumulated)
                            gen_mu, gen_sigma = compute_stats(generated_features)

                            # Compute FID
                            fid_score = compute_fid_from_inception_stats(real_mu, real_sigma, gen_mu, gen_sigma)
                            logger.info(f"FID Score: {fid_score:.2f}")

                            # Clean up
                            del generated_features
                            torch.cuda.empty_cache()

                        logger.info(f"Samples generated")
                dist.barrier()

            # Log training metrics (and samples if generated at this step):
            # When using curriculum the per-GPU batch size may change across phases.
            # To keep the number of logs per epoch roughly constant, scale the
            # logging frequency by the factor (base_batch / current_phase_batch).
            if getattr(args, 'use_curriculum', False) and hasattr(sampler, 'get_current_batch_size'):
                try:
                    current_bs = sampler.get_current_batch_size()
                    base_bs = local_batch_size
                    # factor >= 1 (if current_bs is smaller than base_bs)
                    factor = max(1, int(base_bs // max(1, current_bs)))
                except Exception:
                    factor = 1
                effective_log_every = max(1, args.log_every * factor)
            else:
                effective_log_every = args.log_every

            if train_steps % effective_log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                current_lrs = [group["lr"] for group in opt.param_groups]
                grad_norm_display = last_grad_norm if last_grad_norm is not None else float("nan")
                if args.use_muon and len(current_lrs) > 1:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR_Muon: {current_lrs[0]:.6e}, LR_AdamW: {current_lrs[1]:.6e}, Grad Norm: {grad_norm_display:.4f}")
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lrs[0]:.6e}, Grad Norm: {grad_norm_display:.4f}")

                # Collect all wandb metrics to log at once
                if args.wandb:
                    import wandb

                    log_dict = {
                        "train loss": avg_loss,
                        "train steps/sec": steps_per_sec,
                        "grad_norm": None if last_grad_norm is None else last_grad_norm,
                    }
                    # Add learning rates with descriptive labels
                    if args.use_muon and len(current_lrs) > 1:
                        log_dict["lr_muon"] = current_lrs[0]  # Hidden weights (2D+ params)
                        log_dict["lr_adamw"] = current_lrs[1]  # Gains/biases + non-hidden params
                    else:
                        log_dict["learning rate"] = current_lrs[0]
                    # Add samples and FID if they were generated at this step
                    if 'first_batch_samples' in locals():
                        # If video sampling produced per-frame labels/nrow, pass them through
                        if 'first_batch_video_T' in locals():
                            # Use helper to build grid and labels (one row per video, T columns)
                            duration = first_batch_video_duration if 'first_batch_video_duration' in locals() else None
                            sample_grid = wandb_utils.make_video_grid(first_batch_samples, n_frames=first_batch_video_T, duration=duration)
                            # clear helper metadata
                            del first_batch_video_T
                            if 'first_batch_video_duration' in locals():
                                del first_batch_video_duration
                        else:
                            sample_grid = wandb_utils.array2grid(first_batch_samples)

                        log_dict["samples"] = wandb.Image(sample_grid)
                        del first_batch_samples  # Clear for next time
                    if 'fid_score' in locals() and fid_score is not None:
                        log_dict["fid"] = fid_score
                        fid_score = None  # Clear for next time

                    wandb_utils.log(log_dict, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

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

    # Save final checkpoint
    if rank == 0:
        if interrupted:
            logger.info("Training interrupted. Saving checkpoint...")
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        if interrupted:
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        else:
            logger.info(f"Saved final checkpoint to {checkpoint_path}")
    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if interrupted:
        logger.info("Training interrupted by user.")
    else:
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
    parser.add_argument("--sample-video-duration", type=float, default=1.0, help="Video duration for sampling (slots semantics: time_scale = duration / num_frames)")
    parser.add_argument("--compute-fid", action="store_true", help="Compute frame-wise FID during sampling")
    parser.add_argument("--fid-num-samples", type=int, default=50, help="Number of videos/images to generate for FID computation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear", "cosine"], default="constant", help="Learning rate schedule")
    parser.add_argument("--min-lr-factor", type=float, default=0.1, help="Minimum learning rate as a factor of initial LR (for cosine/linear schedules)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to an EqM checkpoint to continue training from")
    parser.add_argument("--disp", action="store_true", help="Toggle to enable Dispersive Loss")
    parser.add_argument("--uncond", type=bool, default=True, help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none", help="energy formulation")
    parser.add_argument("--video", action="store_true", help="Enable video training mode")
    parser.add_argument("--num-frames", type=int, default=16, help="Maximum number of frames per video clip")
    parser.add_argument("--random-frames", action="store_true", help="Randomly sample number of frames between min-frames and num-frames for each video")
    parser.add_argument("--min-frames", type=int, default=1, help="Minimum number of frames when using --random-frames (default: 1)")
    parser.add_argument("--single-frame-prob", type=float, default=0.0, help="Probability (0.0-1.0) of extracting a single frame (image) from each video (default: 0.0)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training (FP16)")
    parser.add_argument("--use-bf16", action="store_true", help="Enable bfloat16 mixed precision training (BF16)")
    parser.add_argument("--use-rope", action="store_true", help="Use Rotary Position Embedding (RoPE) instead of fixed sinusoidal embeddings. Allows training with varying number of frames.")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer for hidden weights (2D+ params in transformer blocks)")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon optimizer")
    parser.add_argument("--muon-patch-embed", action="store_true", help="Apply Muon to patch embedding projection layer (experimental)")
    parser.add_argument("--use-compile", action="store_true", help="Use torch.compile() for faster training (requires PyTorch 2.0+)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps (allows larger effective batch size with less VRAM)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")

    # Curriculum learning arguments
    parser.add_argument("--use-curriculum", action="store_true", help="Enable curriculum learning: automatically double frame count and halve batch size each phase (steps per phase computed from total training length)")

    # Clip sampling arguments
    parser.add_argument("--use-clip-sampling", action="store_true", help="Sample temporal clips from videos for finer motion detail")
    parser.add_argument("--clip-full-prob", type=float, default=0.3, help="Probability of using full video (default: 0.3)")
    parser.add_argument("--clip-center-prob", type=float, default=0.4, help="Probability of using center-biased clip (default: 0.4)")
    parser.add_argument("--center-bias-fraction", type=float, default=0.6, help="Fraction of video center to bias towards (default: 0.6)")
    parser.add_argument("--clips-per-video", type=int, default=1, help="Number of clips to sample per video for data augmentation (default: 1)")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
