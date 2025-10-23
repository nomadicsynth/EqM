import torch as th

class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


def mean_flat_masked(x, mask=None):
    """
    Take the mean over all non-batch dimensions, optionally with masking.
    
    Args:
        x: (B, ...) tensor
        mask: (B, num_patches) boolean tensor where True = valid, False = padding
              If None, behaves like mean_flat
    
    Returns:
        (B,) tensor of masked means
    """
    if mask is None:
        return mean_flat(x)
    
    # x is typically (B, C, T, H, W) for videos
    # mask is (B, num_patches) where num_patches = (T//pt) * (H//ph) * (W//pw)
    # We need to reshape/expand mask to match x's spatial structure
    
    B = x.shape[0]
    
    # For now, assume x is video latents: (B, C, T, H, W)
    if x.ndim == 5:
        C, T, H, W = x.shape[1:]
        # Mask covers temporal-spatial patches, expand to full resolution
        # This is an approximation - we mask entire frames if their patches are masked
        # Reshape mask to (B, T//pt, H//ph, W//pw) then expand
        num_patches = mask.shape[1]
        # Assume patches are ordered as (t, h, w)
        # Calculate patches per frame
        patches_per_frame = num_patches // T  # This assumes T matches actual temporal dim
        
        # Create frame-level mask: if any patch in a frame is masked, mask the whole frame
        frame_mask = mask.view(B, T, -1).any(dim=2)  # (B, T)
        
        # Expand to (B, C, T, H, W)
        frame_mask = frame_mask.view(B, 1, T, 1, 1).expand_as(x)
        
        # Apply mask and compute mean only over valid positions
        x_masked = x * frame_mask.float()
        valid_count = frame_mask.float().sum(dim=list(range(1, x.ndim)))  # (B,)
        valid_count = th.clamp(valid_count, min=1.0)  # Avoid division by zero
        
        return x_masked.sum(dim=list(range(1, x.ndim))) / valid_count
    else:
        # Fallback to regular mean_flat if not a video tensor
        return mean_flat(x)

def log_state(state):
    result = []
    
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    
    return '\n'.join(result)