# Temporal Encoding for Video Generation

## Problem Statement

The original video generation model had **no concept of frame rate or absolute time**. It treated the temporal dimension exactly like spatial dimensions:

- Training sampled N frames uniformly from videos, but didn't record how much real time those frames represented
- The model learned "what comes after what" but not "how fast things should move"
- Generated videos had to be manually retimed to "look right"
- No way to control or specify the speed of generated motion

This meant the model learned an **average motion speed** from training data, but had no temporal grounding.

## Solution: Time-Scaled Positional Embeddings

We modified the model to use **real time** (seconds) instead of frame indices in the temporal positional encoding. Now the model learns consistent temporal frequencies regardless of frame rate.

### Key Concept

Instead of encoding temporal positions as `[0, 1, 2, 3, ...]` (frame indices), we encode them as `[0.0, 0.033, 0.067, 0.100, ...]` (actual seconds at 30 FPS) or `[0.0, 0.042, 0.083, 0.125, ...]` (actual seconds at 24 FPS).

This makes the sinusoidal positional encoding represent **real motion speeds** rather than arbitrary frame counts.

## Changes Made

### 1. `video_dataset.py` - Capture Temporal Metadata

**Modified `_read_video_frames()`:**

```python
def _read_video_frames(self, path):
    frames, _, info = read_video(path, pts_unit='sec')
    fps = info.get('video_fps', 30.0)  # Extract source FPS
    return frames, fps
```

**Modified `__getitem__()`:**

```python
def __getitem__(self, idx):
    frames, fps = self._read_video_frames(path)
    # ... sample frames ...
    # Calculate actual time span of sampled frames
    time_span = (indices[-1] - indices[0]) / fps  # in seconds
    return proc, label, time_span  # Now returns 3 values
```

### 2. `models.py` - Time-Scaled Positional Encoding

**Modified `get_3d_sincos_pos_embed()`:**

```python
def get_3d_sincos_pos_embed(embed_dim, grid_size, time_scale=1.0, ...):
    gt, gh, gw = grid_size
    grid_t = np.arange(gt, dtype=np.float32) * time_scale  # Scale by real time
    # ... rest of function ...
```

**Modified `EqM.__init__()`:**

- For 3D (video) models, store grid size and use a buffer for default positional encoding
- For 2D (image) models, keep the fixed positional embedding parameter

**Added `get_pos_embed()` method:**

```python
def get_pos_embed(self, time_scale=1.0):
    """Compute positional embedding with temporal scaling."""
    if self.is3d:
        pos_embed = get_3d_sincos_pos_embed(
            self.hidden_size, self.grid_size, time_scale=time_scale
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0).to(device)
    else:
        return self.pos_embed  # Fixed for 2D
```

**Modified `forward()` and `forward_with_cfg()`:**

- Added `time_scale` parameter (default 1.0 for backward compatibility)
- Use dynamic positional embedding: `pos_embed = self.get_pos_embed(time_scale)`

### 3. `train.py` - Training with Temporal Information

**Modified training loop:**

```python
if getattr(args, 'video', False):
    x, y, time_spans = batch  # Unpack time_spans
    time_spans = torch.as_tensor(time_spans, device=device, dtype=torch.float32)
    # Compute time_scale: seconds per frame (averaged across batch)
    time_scale = (time_spans / (T - 1)).mean().item()
else:
    time_scale = 1.0  # No time scaling for images

model_kwargs = dict(y=y, time_scale=time_scale, return_act=args.disp, train=True)
```

### 4. `sample_gd.py` - Generation with Target FPS

**Added `--target-fps` argument:**

```python
parser.add_argument("--target-fps", type=int, default=30, 
                    help="Target frame rate for generated videos (affects temporal dynamics)")
```

**Compute time_scale for generation:**

```python
if getattr(args, 'video', False):
    time_scale = 1.0 / args.target_fps  # e.g., 30 fps -> 0.033 sec/frame
else:
    time_scale = 1.0
```

**Pass time_scale to model:**

```python
model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, time_scale=time_scale)
out = model_fn(xt, t, **model_kwargs)
```

**Save videos at target FPS:**

```python
imageio.mimsave(f"{args.folder}/{index:06d}.gif", video, fps=args.target_fps)
```

## How It Works

### Training

1. **Video is loaded** with metadata (FPS)
2. **Frames are sampled** uniformly (e.g., frames 0, 30, 60, 90 from a 30fps video)
3. **Time span calculated**: `(90 - 0) / 30fps = 3.0 seconds`
4. **Time scale computed**: `3.0 / (4 - 1) = 1.0 sec/frame` (for 4-frame clip)
5. **Positional encoding scaled**: Temporal grid becomes `[0, 1.0, 2.0, 3.0]` seconds
6. **Model learns**: "This much motion happens per second"

### Generation

1. **User specifies target FPS** (e.g., `--target-fps 30`)
2. **Time scale computed**: `1.0 / 30 = 0.033 sec/frame`
3. **Positional encoding scaled**: Temporal grid becomes `[0, 0.033, 0.067, 0.100, ...]` seconds
4. **Model generates**: Motion appropriate for 30 FPS playback
5. **Video saved at 30 FPS**: Matches the temporal dynamics the model generated

## Benefits

### 1. **Temporal Consistency**

- Model learns consistent motion speeds across videos with different source FPS
- "1 second of motion" has the same meaning regardless of frame count

### 2. **Controllable Generation**

- Want slow-motion? Generate at 60 FPS: `--target-fps 60`
- Want fast action? Generate at 15 FPS: `--target-fps 15`
- Model adjusts temporal dynamics accordingly

### 3. **Better Generalization**

- Training data can mix videos at different frame rates
- Model learns temporal patterns in real time, not arbitrary frame indices

### 4. **No Manual Retiming**

- Generated videos play at the correct speed by design
- No need to experiment with different playback rates

## Usage Examples

### Training with Temporal Encoding

```bash
torchrun --nproc_per_node=1 EqM/train.py \
    --data-path datasets/ucf-101/train \
    --model "EqM-S/4" \
    --video \
    --clip-len 16 \
    --global-batch-size 256
```

The model automatically uses temporal information from the dataset.

### Generating at Different Frame Rates

**Standard 30 FPS:**

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 16 \
    --target-fps 30 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples_30fps
```

**Slow motion at 60 FPS:**

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 16 \
    --target-fps 60 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples_60fps
```

**Fast action at 15 FPS:**

```bash
torchrun --nproc_per_node=1 EqM/sample_gd.py \
    --video \
    --clip-len 16 \
    --target-fps 15 \
    --model "EqM-S/4" \
    --ckpt results/027-EqM-S-4-Linear-velocity-None/checkpoints/0000100.pt \
    --folder video_samples_15fps
```

## Technical Details

### Time Scale Calculation

**During Training:**

```python
# For a clip with time_span = 0.5 seconds and 16 frames:
time_scale = 0.5 / (16 - 1) = 0.033 sec/frame
```

**During Generation:**

```python
# For target_fps = 30:
time_scale = 1.0 / 30 = 0.033 sec/frame
```

### Positional Encoding Math

The sinusoidal encoding with time scaling:

```python
# Without time scaling (old):
temporal_positions = [0, 1, 2, 3, 4, ...]  # frame indices

# With time scaling (new):
temporal_positions = [0, 0.033, 0.067, 0.100, 0.133, ...]  # seconds

# Applied to sinusoidal embedding:
omega = 1.0 / (10000 ** (2 * i / embed_dim))
embedding = sin(temporal_position * omega)
```

The key insight: by using seconds instead of indices, the frequencies in the embedding correspond to **real temporal patterns** (Hz) rather than arbitrary frame counts.

### Backward Compatibility

- **Default `time_scale=1.0`**: Without video mode, behaves exactly like the original model
- **Image models**: Completely unchanged, use fixed 2D positional embeddings
- **Old checkpoints**: Fully compatible (though may not have learned optimal temporal scaling)

## Limitations & Future Work

### Current Limitations

1. **Averaging time_scale across batch**: Currently uses mean time_scale per batch. Could support per-sample scaling for more flexibility.

2. **Fixed during generation**: Time scale is set once per batch. Could allow dynamic adjustment during sampling.

3. **Training data requirements**: Works best when training data has consistent, known frame rates. Mixed or unknown FPS data will average together.

### Potential Improvements

1. **Adaptive time scaling**: Learn to predict optimal time_scale from video content
2. **Multi-scale temporal encoding**: Encode both fine and coarse temporal patterns
3. **Explicit FPS conditioning**: Add FPS as an additional input embedding (like class labels)
4. **Temporal interpolation**: Generate intermediate frames for frame rate upscaling

## Comparison to Other Approaches

### vs. No Temporal Encoding (Original)

- ✅ Better temporal consistency
- ✅ Controllable motion speed
- ✅ No manual retiming needed
- ⚠️ Slightly more complex

### vs. FPS Conditioning

- ✅ More fundamental (affects positional encoding)
- ✅ No additional parameters to learn
- ⚠️ Less flexible than learned conditioning

### vs. Learned Temporal Embeddings

- ✅ Explicit, interpretable control
- ✅ No additional training needed
- ⚠️ Less expressive than fully learned embeddings

## Conclusion

By scaling temporal positional encodings with real time information, the model gains explicit temporal grounding. It now understands "seconds" rather than just "frame indices", making generated videos play at the correct speed and allowing controllable motion dynamics through the `--target-fps` parameter.

This is a simple but powerful change that addresses a fundamental limitation in video generation models: the lack of temporal calibration.
