import torch
import torch.nn as nn
import numpy as np


class GaborLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, use_activation=True, is_first=False, omega_0=30., scale_0=30.):
        super().__init__()

        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.is_first = is_first
        self.use_activation = use_activation

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linscale = nn.Linear(in_features, out_features, bias=bias)

    @torch.no_grad()
    def reset_parameters(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features,
                                         1 / self.in_features)
        else:
            self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                         np.sqrt(6 / self.in_features) / self.omega_0)

    

    def forward(self, input):

        output = self.linear(input)
        scale = self.linscale(input)
        if self.use_activation:
            # output = torch.sin(self.omega_0 * output)
            omega = self.omega_0 * output
            scale = scale* self.scale_0
        output = torch.cos(omega)*torch.exp(-(scale**2))
        return output

        # return output

class SIREN(nn.Module):
    def __init__(self,
                 in_features, hidden_features, out_features,
                 n_hidden_layers, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.,
                 first_scale_0=30, hidden_scale_0=30.,
                 pos_encode=False):
        super().__init__()

        self.pos_encode = pos_encode

        self.net = []
        self.net.append(GaborLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, scale_0=first_scale_0))

        for i in range(n_hidden_layers):
            self.net.append(GaborLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, scale_0=hidden_scale_0))

        self.net.append(GaborLayer(hidden_features, out_features, is_first=False, use_activation=(not outermost_linear),omega_0=hidden_omega_0, scale_0=hidden_scale_0))

        self.net = nn.Sequential(*self.net)


    def forward(self, coords):

        if self.pos_encode:
            coords = self.positional_encoding(coords)

        output = self.net(coords)

        return output
