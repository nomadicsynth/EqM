# Mixed Precision Training Guide

This guide explains how to use the mixed precision training features in EqM.

## Overview

Mixed precision training uses lower precision (FP16 or BF16) arithmetic to speed up training and reduce memory usage while maintaining model accuracy. Two modes are now supported:

1. **FP16 (Automatic Mixed Precision)**: Uses 16-bit floating point with gradient scaling
2. **BF16 (Brain Float 16)**: Uses bfloat16 format (better numerical stability, no gradient scaling needed)

## Training with Mixed Precision

### Using BF16 (Recommended for modern GPUs)

BF16 is recommended for NVIDIA Ampere GPUs (A100, RTX 3090, etc.) and newer:

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --use-bf16 \
    --video \
    --clip-len 4
```

### Using FP16 (For older GPUs)

FP16 works on older GPUs (V100, etc.):

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --use-amp \
    --video \
    --clip-len 4
```

### Full Precision (FP32)

Default behavior without any flags:

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --video \
    --clip-len 4
```

## Sampling/Inference with Mixed Precision

You can also use mixed precision during sampling for faster inference:

### BF16 Sampling

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 4 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples \
    --use-bf16 \
    --num-fid-samples 1
```

### FP16 Sampling

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 4 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples \
    --use-amp \
    --num-fid-samples 1
```

## Benefits

- **Speed**: 1.5-3x faster training depending on model and GPU
- **Memory**: ~40-50% less GPU memory usage
- **Throughput**: Larger batch sizes possible with same GPU
- **Quality**: Minimal to no impact on final model quality

## Hardware Requirements

- **BF16**: Requires NVIDIA Ampere (A100, RTX 30 series) or newer
- **FP16**: Works on NVIDIA Volta (V100) and newer
- **FP32**: Works on all GPUs

## Notes

- BF16 has better numerical stability than FP16 (wider dynamic range)
- FP16 uses gradient scaling to prevent underflow
- Both options are safe to use and automatically fall back to FP32 if needed
- You can train with one precision and sample with another
