import os, sys, json, imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Misc utils
# -----------------------------------------------------------------------------
def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10.0 * torch.log10(x)

def to8b(x):
    # x: numpy array in [0,1]
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# -----------------------------------------------------------------------------
# Positional encoding
# -----------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(self, input_dims, max_freq_log2, num_freqs, include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, num_freqs)
        self.register_buffer('freq_bands', freq_bands)  # [L]

        self.out_dim = input_dims * (1 + 2 * num_freqs) if include_input else input_dims * (2 * num_freqs)

    def forward(self, x):
        # x: [..., input_dims]
        out = []
        if self.include_input:
            out.append(x)
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


def get_embedder(multires, i_embed=0):
    if i_embed < 0:
        return nn.Identity(), 3
    embedder = Embedder(input_dims=3,
                        max_freq_log2=multires-1,
                        num_freqs=multires,
                        include_input=True,
                        log_sampling=True)
    return embedder, embedder.out_dim


# -----------------------------------------------------------------------------
# NeRF MLP
# -----------------------------------------------------------------------------
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=0,
                 output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # point-branch
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(1, D)]
        )

        if use_viewdirs:
            # output alpha from point-branch
            self.alpha_linear = nn.Linear(W, 1)
            # bottleneck
            self.feature_linear = nn.Linear(W, W)
            # view-dependent branch
            self.view_linears = nn.ModuleList([nn.Linear(W + input_ch_views, W//2)])
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # x: [N_rays*N_samples, input_ch + input_ch_views]
        input_pts, input_views = x[..., :self.input_ch], x[..., self.input_ch:]
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)                            # [*,1]
            features = self.feature_linear(h)                       # [*,W]
            h = torch.cat([features, input_views], -1)              # [*, W+input_ch_views]
            for l in self.view_linears:
                h = F.relu(l(h))
            rgb = self.rgb_linear(h)                                # [*,3]
            outputs = torch.cat([rgb, alpha], -1)                   # [*,4]
        else:
            outputs = self.output_linear(h)                         # [*,output_ch]

        return outputs


# -----------------------------------------------------------------------------
# Ray helpers
# -----------------------------------------------------------------------------
def get_rays(H, W, focal, c2w):
    """
    Get ray origins and directions for all pixels in an (H,W) image.
    c2w: [3,4] camera-to-world
    """
    i = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    j = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    dirs = torch.stack([(i - 0.5*W)/focal,
                        -(j - 0.5*H)/focal,
                        -torch.ones_like(i)], -1)  # [H,W,3]
    # Rotate ray directions from camera to world
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)   # [H,W,3]
    # All rays have same origin
    rays_o = c2w[:3,3].view(1,1,3).expand_as(rays_d)           # [H,W,3]
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - 0.5*W)/focal,
                     -(j - 0.5*H)/focal,
                     -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[...,None,:] * c2w[:3,:3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3,3], rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t.unsqueeze(-1) * rays_d

    # Project to NDC
    o0 = -1.0/(W/(2.0*focal)) * rays_o[...,0]/rays_o[...,2]
    o1 = -1.0/(H/(2.0*focal)) * rays_o[...,1]/rays_o[...,2]
    o2 = 1.0 + 2.0*near / rays_o[...,2]
    d0 = -1.0/(W/(2.0*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1.0/(H/(2.0*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2.0*near / rays_o[...,2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


# -----------------------------------------------------------------------------
# Hierarchical sampling
# -----------------------------------------------------------------------------
def sample_pdf(bins, weights, N_samples, det=False):
    """
    bins: [..., N+1]
    weights: [..., N]
    """
    # Add small eps and normalize
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)       # [..., N]
    cdf = torch.cumsum(pdf, dim=-1)                                # [..., N]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)     # [..., N+1]

    # Pick uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)                  # [..., N_samples]
    below = torch.clamp(inds-1, 0, cdf.shape[-1]-1)
    above = torch.clamp(inds,   0, cdf.shape[-1]-1)
    cdf_b = torch.gather(cdf, -1, below)
    cdf_a = torch.gather(cdf, -1, above)
    bins_b = torch.gather(bins, -1, below)
    bins_a = torch.gather(bins, -1, above)

    denom = cdf_a - cdf_b
    denom[denom<1e-5] = 1.0
    t = (u - cdf_b) / denom
    samples = bins_b + t * (bins_a - bins_b)
    return samples
