# RoPE (Rotary Position Embedding) for Progressive Video Training

## Overview

RoPE enables training with **variable number of frames**, allowing you to progressively increase the temporal context as training progresses. This is particularly useful for video generation where you want to:

1. Start training with lower number of frames (less memory, faster iterations)
2. Gradually increase number of frames as the model stabilizes
3. Sample any number of frames during inference (even different from training)

## Why RoPE?

**Traditional Positional Embeddings (default):**

- Fixed sinusoidal embeddings computed at model initialization
- Tied to specific `num_frames` used during training
- Require checkpoint surgery (removing `pos_embed_default`) to use different number of frames at inference
- Cannot change number of frames during training without issues

**RoPE (Rotary Position Embedding):**

- ✅ Applied dynamically based on actual input sequence length
- ✅ Naturally generalizes to any number of frames (≥ patch_size)
- ✅ No fixed buffers - position encoded in attention mechanism
- ✅ Enables progressive training with increasing number of frames
- ✅ Same checkpoint works for any number of frames during inference

## Usage

### Training with RoPE

Add `--use-rope` flag to your training command:

```bash
torchrun --nproc_per_node=8 EqM/train.py \
  --video \
  --num-frames 4 \
  --model "EqM-S/4" \
  --data-path datasets/ucf-101 \
  --use-rope \
  --other-args...
```

### Progressive Training Strategy

Start with small number of frames and increase gradually:

```bash
# Phase 1: Small num_frames (0-50k steps)
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 4 --use-rope \
  --max-steps 50000 \
  --other-args...

# Phase 2: Medium num_framess (50k-100k steps)  
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 8 --use-rope \
  --ckpt results/xxx/checkpoints/0050000.pt \
  --max-steps 100000 \
  --other-args...

# Phase 3: Large num_framess (100k+ steps)
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 16 --use-rope \
  --ckpt results/xxx/checkpoints/0100000.pt \
  --other-args...
```

### Inference with RoPE

Sample **any** number of frames (doesn't need to match training):

```bash
# Trained with num_frames=4, sample with num_frames=16
torchrun --nproc_per_node=1 EqM/sample_gd.py \
  --video \
  --num-frames 16 \
  --model "EqM-S/4" \
  --ckpt results/xxx/checkpoints/best.pt \
  --use-rope \
  --other-args...
```

**Important:** You MUST use `--use-rope` if the model was trained with RoPE!

## Constraints

1. **Temporal dimension constraint:** `num_frames` must be ≥ `patch_size`
   - For `EqM-S/4`: `num_frames` ≥ 4
   - For `EqM-S/2`: `num_frames` ≥ 2

2. **Spatial dimensions:** Fixed by model architecture (256x256 for standard configs)

3. **Memory scaling:** Longer clips use more memory:
   - `num_frames=4`: baseline memory
   - `num_frames=8`: ~2x memory
   - `num_frames=16`: ~4x memory

   Adjust `--global-batch-size` accordingly!

## Converting Existing Checkpoints

**Cannot convert:** Models trained without RoPE cannot be converted to use RoPE (different architecture).

**Must retrain:** Start fresh training with `--use-rope` flag.

## Performance Considerations

1. **RoPE adds minimal overhead:** ~1-2% slower than fixed embeddings
2. **Better generalization:** Can adapt to any sequence length
3. **Training flexibility:** Curriculum learning with increasing number of frames
4. **Inference flexibility:** One checkpoint for any number of frames

## Example: Full Progressive Training

```bash
# Start with small num_frames, increase every 50k steps
EXP_NAME="progressive_rope_training"
MODEL="EqM-S/4"
DATA="datasets/ucf-101"

# Phase 1: 4 frames (0-50k)
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 4 --use-rope \
  --model $MODEL --data-path $DATA \
  --results-dir results/$EXP_NAME \
  --max-steps 50000 \
  --global-batch-size 256

# Phase 2: 8 frames (50k-100k)
CKPT=$(ls results/$EXP_NAME/*/checkpoints/0050000.pt)
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 8 --use-rope \
  --model $MODEL --data-path $DATA \
  --results-dir results/$EXP_NAME \
  --ckpt $CKPT \
  --max-steps 100000 \
  --global-batch-size 128  # Reduce for memory

# Phase 3: 16 frames (100k+)
CKPT=$(ls results/$EXP_NAME/*/checkpoints/0100000.pt)
torchrun --nproc_per_node=8 EqM/train.py \
  --video --num-frames 16 --use-rope \
  --model $MODEL --data-path $DATA \
  --results-dir results/$EXP_NAME \
  --ckpt $CKPT \
  --epochs 100 \
  --global-batch-size 64  # Further reduce for memory
```

## Technical Details

RoPE works by:

1. Splitting the feature dimension into temporal, height, and width components (1/3 each dimension gets equal capacity)
   - This is a simple, fixed strategy that works across any number of frames
   - Temporal gets 1/3 even when grid is small (e.g., 1×8×8) because it encodes time_scale
   - As num_frames increases, temporal patches increase while spatial stays fixed
2. Applying rotary embeddings separately to each dimension
3. Encoding position information through rotation in feature space
4. Computing positions dynamically based on actual input grid size

This allows the same learned attention patterns to work across different sequence lengths!

**Note on terminology:** `--num-frames` refers to the **number of frames**, not temporal duration. The actual temporal duration is `num_frames × time_scale`, where `time_scale` is seconds per frame.

## References

- Original RoPE paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Adapted for 3D video patches with separate temporal/spatial encoding
