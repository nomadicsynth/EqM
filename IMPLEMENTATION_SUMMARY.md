# Mixed Precision Training - Implementation Summary

## Changes Made

### 1. Modified `EqM/train.py`

**Added imports:**

- `from torch.cuda.amp import autocast, GradScaler`

**New features:**

- Added GradScaler for automatic loss scaling (FP16)
- Added autocast context manager for mixed precision operations
- Dynamic dtype selection based on flags (BF16, FP16, or FP32)

**New command-line arguments:**

- `--use-amp`: Enable FP16 automatic mixed precision training
- `--use-bf16`: Enable BF16 mixed precision training

**Code modifications:**

- Wrapped model forward pass in autocast context
- Applied gradient scaling with scaler.scale(), scaler.step(), scaler.update()
- Added logging to show which precision mode is active

### 2. Modified `EqM/sample_gd.py`

**Added imports:**

- `from torch.cuda.amp import autocast`

**New features:**

- Added autocast context manager for inference
- Dynamic dtype selection for sampling

**New command-line arguments:**

- `--use-amp`: Enable FP16 automatic mixed precision inference
- `--use-bf16`: Enable BF16 mixed precision inference

**Code modifications:**

- Wrapped sampling loop in autocast context
- Added print statements showing which precision is being used

### 3. Created Documentation

**Files created:**

- `MIXED_PRECISION.md`: Comprehensive guide on using mixed precision

## Usage Examples

### Training with BF16 (Recommended)

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --use-bf16 \
    --video \
    --clip-len 4 \
    --global-batch-size 256
```

### Training with FP16

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --use-amp \
    --video \
    --clip-len 4 \
    --global-batch-size 256
```

### Sampling with BF16

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 4 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples \
    --use-bf16 \
    --num-fid-samples 1000
```

## Technical Details

### BF16 vs FP16

**BF16 (Brain Float 16):**

- 8-bit exponent (same as FP32) → better numerical range
- 7-bit mantissa → slightly lower precision
- No gradient scaling needed
- Better numerical stability
- Requires Ampere GPUs or newer

**FP16 (Half Precision):**

- 5-bit exponent → smaller numerical range
- 10-bit mantissa → better precision
- Requires gradient scaling to prevent underflow
- Works on Volta GPUs and newer

### Gradient Scaling (FP16 only)

The GradScaler automatically:

1. Scales up loss to prevent gradient underflow
2. Checks for inf/NaN gradients
3. Unscales gradients before optimizer step
4. Updates scale factor dynamically

This is handled automatically when `--use-amp` is enabled.

### Autocast Context

The autocast context manager automatically:

- Converts operations to lower precision where safe
- Keeps certain operations in FP32 for numerical stability
- Works transparently with existing code

## Performance Benefits

Expected speedups (depends on GPU and model size):

- **Training**: 1.5-3x faster
- **Memory**: 40-50% reduction
- **Batch size**: Can increase by ~2x

## Backward Compatibility

- Without flags: Runs in FP32 (no change from original)
- Old checkpoints: Fully compatible
- Mixed usage: Can train in one precision, sample in another
