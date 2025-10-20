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


#################################################################################
#                                 Core EqM Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
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

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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
        ebm='none'
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

        if self.is3d:
            # input_size expected as (T,H,W)
            self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size, bias=True)
        else:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding (2D or 3D):
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
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
            # build 3D pos embed using input_size and grid
            T, H, W = self.input_size
            if isinstance(self.patch_size, int):
                pt = ph = pw = self.patch_size
            else:
                pt, ph, pw = self.patch_size
            gs_t, gs_h, gs_w = T // pt, H // ph, W // pw
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], (gs_t, gs_h, gs_w))
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

    def forward(self, x0, t, y, return_act=False, get_energy=False, train=False):
        """
        Forward pass of EqM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x0.requires_grad_(True)
        if self.uncond: # removes noise/time conditioning by setting to 0
            t = torch.zeros_like(t)
        act = []
        x = self.x_embedder(x0) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
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

    def forward_with_cfg(self, x, t, y, cfg_scale, return_act=False, get_energy=False, train=False):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, return_act=return_act, get_energy=get_energy, train=train)
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


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple (T, H, W)
    returns: (T*H*W, embed_dim)
    """
    gt, gh, gw = grid_size
    grid_t = np.arange(gt, dtype=np.float32)
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
