import torch

def get_nuclear_norm_regularizer(params):

    reg_loss = 0.
    n_layers = (len(params) - 1) // 2

    for name, param in params.items():
        if not param.ndim == 3: continue
        if not name.startswith('layers.'): continue
        layer_idx = int(name.split('.')[1])
        if 0 < layer_idx <= n_layers - 1:
            norm = torch.norm(param, p="nuc", dim=[1, 2])
            reg_loss = reg_loss + norm

    return reg_loss