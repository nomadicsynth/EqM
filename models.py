# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class PatchEmbed3D(nn.Module):
    """3D patch embedding (for videos).

    Splits (N, C, T, H, W) into patches of size (pt, ph, pw) and projects to embed_dim.
    Returns (N, num_patches, embed_dim) to match the 2D PatchEmbed API.
    """
    def __init__(self, input_size, patch_size, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        # input_size: tuple (T, H, W) or int (assumed H=W and T given elsewhere)
        if isinstance(input_size, int):
            raise ValueError("PatchEmbed3D requires input_size as (T,H,W) tuple")
        T, H, W = input_size
        if isinstance(patch_size, int):
            pt = ph = pw = patch_size
        else:
            pt, ph, pw = patch_size
        self.input_size = (T, H, W)
        self.patch_size = (pt, ph, pw)
        self.grid_size = (T // pt, H // ph, W // pw)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # Store spatial grid size separately for dynamic temporal inference
        self.spatial_grid_size = (H // ph, W // pw)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(pt, ph, pw), stride=(pt, ph, pw), bias=bias)

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)  # (N, D, T//pt, H//ph, W//pw)
        N, D, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, D)
        return x


class FinalLayer3D(nn.Module):
    """Final linear layer for 3D patches.
    Projects hidden vectors to patch volume values: pt*ph*pw*out_channels per token.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        if isinstance(patch_size, int):
            pt = ph = pw = patch_size
        else:
            pt, ph, pw = patch_size
        self.pt, self.ph, self.pw = pt, ph, pw
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, pt * ph * pw * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def create_temporal_mask(num_real_frames, grid_size, device):
    """
    Create attention mask for padded video sequences.
    
    Args:
        num_real_frames: (B,) tensor of real frame counts per video
        grid_size: (t, h, w) tuple of patch grid dimensions
        device: torch device
    
    Returns:
        mask: (B, num_patches) boolean tensor where True = valid, False = padding
    """
    if num_real_frames is None:
        return None
    
    t, h, w = grid_size
    B = num_real_frames.shape[0]
    num_patches = t * h * w
    
    # Create mask: True for valid positions, False for padding
    mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)
    
    for b in range(B):
        real_t = num_real_frames[b].item()
        # All spatial patches are valid, but only first real_t temporal patches
        # Patches are ordered as: (t=0, h, w), (t=1, h, w), ..., (t=T-1, h, w)
        valid_patches = real_t * h * w
        mask[b, :valid_patches] = True
    
    return mask


#################################################################################
#                          Rotary Position Embedding (RoPE)                     #
#################################################################################

class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for video patches.
    Applies separate rotary embeddings for temporal, height, and width dimensions.
    This naturally generalizes to any sequence length without retraining.
    """
    def __init__(self, dim, max_freq=10000, time_scale=1.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        # TODO: Remove unused time_scale
        self.time_scale = time_scale

    def forward(self, x, grid_size, time_scale, patch_temporal=1):
        """
        Apply 3D RoPE to input tensor.
        
        Args:
            x: (batch, num_patches, dim) tensor
            grid_size: (t, h, w) tuple indicating patch grid dimensions
            time_scale: (batch,) tensor of temporal scaling factor (seconds per frame)
        
        Returns:
            x with rotary position embeddings applied
        """
        batch_size, num_patches, dim = x.shape
        t, h, w = grid_size
        
        # Split dimensions for temporal, height, width
        # Using 1/3 of dim for each spatial dimension
        dim_t = dim // 3
        dim_h = dim // 3
        dim_w = dim - dim_t - dim_h
        
        # Split x into batches and process each with its own time_scale
        x_t_list, x_h_list, x_w_list = [], [], []
        
        for b in range(batch_size):
            # Generate temporal positions as patch-center times in seconds.
            # For token index i (0..t-1), the patch covers frames [i*pt .. i*pt+pt-1].
            # The center frame index is i*pt + (pt-1)/2, so convert to seconds by
            # multiplying by time_scale (seconds per frame).
            pt = patch_temporal if isinstance(patch_temporal, int) else int(patch_temporal)
            # Patch-end anchoring: token represents the last frame in the patch
            pos_t = (torch.arange(t, device=x.device, dtype=torch.float32) * pt + (pt - 1)) * time_scale[b]
            pos_h = torch.arange(h, device=x.device, dtype=torch.float32)
            pos_w = torch.arange(w, device=x.device, dtype=torch.float32)
            
            # Create meshgrid and flatten to match patch ordering
            grid_t, grid_h, grid_w = torch.meshgrid(pos_t, pos_h, pos_w, indexing='ij')
            positions = torch.stack([
                grid_t.flatten(),
                grid_h.flatten(), 
                grid_w.flatten()
            ], dim=1)  # (num_patches, 3)
            
            # Apply rotary embedding to each dimension for this batch element
            x_b = x[b:b+1]  # (1, num_patches, dim)
            x_t_b, x_h_b, x_w_b = x_b.split([dim_t, dim_h, dim_w], dim=-1)
            
            x_t_b = self.apply_rotary_emb_1d(x_t_b, positions[:, 0], dim_t)
            x_h_b = self.apply_rotary_emb_1d(x_h_b, positions[:, 1], dim_h)
            x_w_b = self.apply_rotary_emb_1d(x_w_b, positions[:, 2], dim_w)
            
            x_t_list.append(x_t_b)
            x_h_list.append(x_h_b)
            x_w_list.append(x_w_b)
        
        # Concatenate all batch elements
        x_t = torch.cat(x_t_list, dim=0)
        x_h = torch.cat(x_h_list, dim=0)
        x_w = torch.cat(x_w_list, dim=0)
        
        return torch.cat([x_t, x_h, x_w], dim=-1)
    
    def apply_rotary_emb_1d(self, x, positions, dim):
        """
        Apply 1D rotary embedding to a subset of features.
        
        Args:
            x: (batch, num_patches, dim) tensor
            positions: (num_patches,) position indices
            dim: feature dimension
        """
        # Compute frequencies
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(self.max_freq) * torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim
        )
        
        # positions: (num_patches,), freqs: (half_dim,)
        # Create (num_patches, half_dim) matrix of position * frequency
        pos_freqs = positions[:, None] * freqs[None, :]  # (num_patches, half_dim)
        
        # Compute sin and cos
        sin = torch.sin(pos_freqs)  # (num_patches, half_dim)
        cos = torch.cos(pos_freqs)  # (num_patches, half_dim)
        
        # Split x into pairs for rotation
        # x: (batch, num_patches, dim) -> (batch, num_patches, half_dim, 2)
        x = x.view(x.shape[0], x.shape[1], half_dim, 2)
        
        # Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x_rotated = torch.stack([
            x[..., 0] * cos - x[..., 1] * sin,
            x[..., 0] * sin + x[..., 1] * cos
        ], dim=-1)
        
        # Reshape back
        return x_rotated.view(x.shape[0], x.shape[1], dim)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# For later potential experimentation
class VideoTemporalEmbedder(nn.Module):
    """
    Embeds video temporal parameters (duration, fps, num_frames) into vector representations.
    Uses continuous embeddings similar to timestep embeddings.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # Three separate MLPs for duration, fps, and frame count
        self.duration_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.fps_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frames_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Combine all three embeddings
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def continuous_embedding(values, dim, max_period=10000):
        """
        Create sinusoidal embeddings for continuous values.
        Similar to timestep embedding but for any continuous scalar.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=values.device)
        args = values[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, duration, fps, num_frames):
        """
        Args:
            duration: (N,) tensor of video durations in seconds
            fps: (N,) tensor of frame rates
            num_frames: (N,) tensor of frame counts
        """
        dur_freq = self.continuous_embedding(duration, self.frequency_embedding_size)
        fps_freq = self.continuous_embedding(fps, self.frequency_embedding_size)
        frames_freq = self.continuous_embedding(num_frames, self.frequency_embedding_size)
        
        dur_emb = self.duration_mlp(dur_freq)
        fps_emb = self.fps_mlp(fps_freq)
        frames_emb = self.frames_mlp(frames_freq)
        
        # Concatenate and combine
        combined = torch.cat([dur_emb, fps_emb, frames_emb], dim=1)
        return self.combine(combined)


#################################################################################
#                                 Core EqM Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE3D(hidden_size)

    def forward(self, x, c, grid_size=None, time_scale=None, attn_mask=None, patch_size=1):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Apply RoPE before attention if enabled
        x_normed = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_rope and grid_size is not None:
            # Determine temporal patch size (pt) from patch_size argument
            if isinstance(patch_size, int):
                pt = patch_size
            else:
                # patch_size may be tuple (pt, ph, pw)
                try:
                    pt = patch_size[0]
                except Exception:
                    pt = 1
            x_normed = self.rope(x_normed, grid_size, time_scale, patch_temporal=pt)
        
        # Apply attention with optional masking
        if attn_mask is not None:
            # timm's Attention doesn't support attn_mask directly, so we'll use a workaround
            # Store original attention and apply mask via hook or manual implementation
            # For now, use manual scaled dot-product attention
            attn_output = self._masked_attention(x_normed, attn_mask)
        else:
            attn_output = self.attn(x_normed)
        
        x = x + gate_msa.unsqueeze(1) * attn_output
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def _masked_attention(self, x, attn_mask):
        """
        Manual attention computation with masking support.
        attn_mask: (B, N) boolean mask where True = valid position, False = padding
        """
        B, N, C = x.shape
        # Use the existing attention's qkv projection
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.attn.scale  # (B, num_heads, N, N)
        
        # Apply mask: set padding positions to -inf so they get 0 attention after softmax
        if attn_mask is not None:
            # attn_mask is (B, N), expand to (B, 1, 1, N) for broadcasting
            mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EqM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        uncond=True,
        ebm='none',
        use_rope=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        # support 3D patching: if input_size is tuple (T,H,W) or patch_size is tuple length 3
        self.is3d = (isinstance(input_size, tuple) and len(input_size) == 3) or (not isinstance(patch_size, int) and len(patch_size) == 3)
        self.input_size = input_size
        self.num_heads = num_heads
        self.use_rope = use_rope

        if self.is3d:
            # input_size expected as (T,H,W)
            self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size, bias=True)
        else:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use RoPE or fixed sin-cos embedding
        if self.is3d:
            # Store grid size for dynamic computation
            if isinstance(patch_size, int):
                pt = ph = pw = patch_size
            else:
                pt, ph, pw = patch_size
            T, H, W = input_size
            self.grid_size = (T // pt, H // ph, W // pw)
            self.hidden_size = hidden_size
            if not use_rope:
                # Initialize with default time_scale=1.0 (frame indices)
                pos_embed_default = get_3d_sincos_pos_embed(hidden_size, self.grid_size, time_scale=1.0, patch_temporal=pt)
                self.register_buffer('pos_embed_default', torch.from_numpy(pos_embed_default).float().unsqueeze(0))
        else:
            if not use_rope:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_rope=use_rope) for _ in range(depth)
        ])
        if self.is3d:
            self.final_layer = FinalLayer3D(hidden_size, patch_size, self.out_channels)
        else:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.uncond = uncond
        self.ebm = ebm

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if getattr(self, 'is3d', False):
            # For 3D, pos_embed_default is already initialized in __init__
            # No need to initialize again here
            pass
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed conv weights (2D or 3D):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """Reconstruct images or videos from patch tokens.

        For 2D: x: (N, num_patches, p*p*C) -> (N, C, H, W)
        For 3D: x: (N, num_patches, pt*ph*pw*C) -> (N, C, T, H, W)
        """
        c = self.out_channels
        # get patch sizes from embedder
        if getattr(self, 'is3d', False):
            pt, ph, pw = self.x_embedder.patch_size
            # Dynamically compute grid size from number of patches
            num_patches = x.shape[1]
            if self.use_rope:
                # For RoPE, we need to infer grid size from the number of patches
                # Spatial grid size is fixed, temporal can vary
                gs_h, gs_w = self.x_embedder.spatial_grid_size
                gs_t = num_patches // (gs_h * gs_w)
                assert gs_t * gs_h * gs_w == num_patches, f"Cannot infer grid size: {num_patches} patches doesn't match grid {gs_t}x{gs_h}x{gs_w}"
            else:
                gs_t, gs_h, gs_w = self.x_embedder.grid_size
            
            N = x.shape[0]
            x = x.reshape(N, gs_t, gs_h, gs_w, pt, ph, pw, c)
            # reorder to (N, C, T, H, W)
            x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            imgs = x.reshape(N, c, gs_t * pt, gs_h * ph, gs_w * pw)
            return imgs
        else:
            p = self.x_embedder.patch_size[0]
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
            return imgs

    def get_pos_embed(self, time_scale=1.0, grid_size=None):
        """
        Compute positional embedding based on time_scale.
        For 3D models, this scales the temporal dimension.
        For 2D models, returns the fixed pos_embed.
        
        Args:
            time_scale: Temporal scaling factor (seconds per frame)
            grid_size: Optional (t, h, w) tuple. If None, uses self.grid_size from training.
        """
        if self.is3d:
            # Use provided grid_size if available (for variable-length inference)
            # Otherwise fall back to training grid_size
            if grid_size is None:
                grid_size = self.grid_size
            # Compute dynamic positional embedding with time scaling using patch-center times
            if isinstance(self.patch_size, int):
                pt = self.patch_size
            else:
                pt = self.patch_size[0]
            pos_embed = get_3d_sincos_pos_embed(self.hidden_size, grid_size, time_scale=time_scale, patch_temporal=pt)
            return torch.from_numpy(pos_embed).float().unsqueeze(0).to(self.pos_embed_default.device)
        else:
            return self.pos_embed

    def forward(self, x0, t, y, time_scale, return_act=False, get_energy=False, train=False, num_real_frames=None):
        """
        Forward pass of EqM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        time_scale: (N,) tensor of temporal scaling (seconds per frame for videos)
        num_real_frames: (N,) tensor of real (non-padded) frame counts for masking
        """
        x0.requires_grad_(True)
        if self.uncond: # removes noise/time conditioning by setting to 0
            t = torch.zeros_like(t)
        act = []
        
        # Embed patches
        x = self.x_embedder(x0)  # (N, num_patches, D)
        
        # Compute actual grid size from input for dynamic positional embeddings
        if self.is3d:
            N, C, T, H, W = x0.shape
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                pt, ph, pw = self.patch_size
            actual_grid_size = (T // pt, H // ph, W // pw)
        else:
            actual_grid_size = None
        
        # Add positional embedding if not using RoPE
        if not self.use_rope:
            pos_embed = self.get_pos_embed(time_scale, grid_size=actual_grid_size)
            x = x + pos_embed
        
        # Get conditioning
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        
        # Get grid size for RoPE and masking
        if self.use_rope and self.is3d:
            # Dynamically compute grid size from input
            N, C, T, H, W = x0.shape
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                pt, ph, pw = self.patch_size
            grid_size = (T // pt, H // ph, W // pw)
        elif self.use_rope:
            # 2D case
            grid_size = (int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        else:
            grid_size = None
        
        # Create attention mask for variable-length sequences
        if self.is3d and grid_size is not None:
            N, C, T, H, W = x0.shape
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                pt, ph, pw = self.patch_size
            mask_grid_size = (T // pt, H // ph, W // pw)
            attn_mask = create_temporal_mask(num_real_frames, mask_grid_size, x0.device)
        else:
            attn_mask = None
        
        # Transformer blocks
        for block in self.blocks:
            if self.use_rope:
                x = block(x, c, grid_size=grid_size, time_scale=time_scale, attn_mask=attn_mask)
            else:
                x = block(x, c, attn_mask=attn_mask)
            act.append(x)
            
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        # explicit energy
        E = 0
        if self.ebm in ('l2', 'dot', 'mean'):
            # sum over all non-batch dims
            sum_dims = tuple(range(1, x.dim()))
        if self.ebm == 'l2':
            E = -torch.sum(x**2, dim=sum_dims) / 2
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()], [x0], create_graph=train)[0]
        if self.ebm == 'dot':
            E = torch.sum(x * x0, dim=sum_dims)
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()], [x0], create_graph=train)[0]
        if self.ebm == 'mean':
            E = torch.sum(x * x0, dim=sum_dims)
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()], [x0], create_graph=train)[0]
        if get_energy:
            return x, -E
        if return_act: 
            return x, act
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, time_scale, return_act=False, get_energy=False, train=False, num_real_frames=None):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        time_scale: (N,) tensor of temporal scaling (seconds per frame for videos)
        num_real_frames: (N,) tensor of real (non-padded) frame counts for masking
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, time_scale=time_scale, return_act=return_act, get_energy=get_energy, train=train, num_real_frames=num_real_frames)
        if get_energy:
            x, E = model_out
            model_out=x
        if return_act:
            act = model_out[1]
            model_out = model_out[0]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1), act
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, time_scale=1.0, cls_token=False, extra_tokens=0, patch_temporal=1):
    """
    grid_size: tuple (T, H, W)
    time_scale: scaling factor for temporal dimension (in seconds per frame)
    returns: (T*H*W, embed_dim)
    """
    gt, gh, gw = grid_size
    pt = patch_temporal
    # Compute patch-end times: for token index i, end frame index = i*pt + (pt-1)
    # Treat time_scale as seconds per frame and convert to seconds per token end.
    grid_t = (np.arange(gt, dtype=np.float32) * pt + (pt - 1)) * time_scale
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_t, indexing='xy')  # w, h, t ordering
    grid = np.stack(grid, axis=0)  # (3, W, H, T)
    # reshape to (3, 1, T, H, W) to reuse 1d helpers
    grid = grid.reshape([3, -1])
    # We'll compute separate embeddings for t,h,w and concat
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3 * 1, grid[2])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3 * 1, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim - emb_t.shape[1] - emb_h.shape[1], grid[0])
    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)
    if cls_token and extra_tokens > 0:
        emb = np.concatenate([np.zeros([extra_tokens, embed_dim]), emb], axis=0)
    return emb


#################################################################################
#                                   EqM Configs                                  #
#################################################################################

def EqM_XL_2(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def EqM_XL_4(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def EqM_XL_8(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def EqM_L_2(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def EqM_L_4(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def EqM_L_8(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def EqM_B_2(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def EqM_B_4(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def EqM_B_8(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def EqM_S_2(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def EqM_S_4(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def EqM_S_8(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


EqM_models = {
    'EqM-XL/2': EqM_XL_2,  'EqM-XL/4': EqM_XL_4,  'EqM-XL/8': EqM_XL_8,
    'EqM-L/2':  EqM_L_2,   'EqM-L/4':  EqM_L_4,   'EqM-L/8':  EqM_L_8,
    'EqM-B/2':  EqM_B_2,   'EqM-B/4':  EqM_B_4,   'EqM-B/8':  EqM_B_8,
    'EqM-S/2':  EqM_S_2,   'EqM-S/4':  EqM_S_4,   'EqM-S/8':  EqM_S_8,
}
