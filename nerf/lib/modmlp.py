import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.util import get_subdict, get_coordinate_grid


def get_init_magnitude(in_features, init_mode, nonlinearity):

    if in_features == 0: return 0.

    if init_mode == "uniform":
        init_magnitude = np.sqrt(6. / in_features)
    elif init_mode == "normal":
        gain = torch.nn.init.calculate_gain(nonlinearity) if nonlinearity != "sin" else 1.
        init_magnitude = gain * np.sqrt(1. / in_features)
    elif init_mode == "first":
        init_magnitude = 1. / in_features
    else:
        raise NotImplementedError

    return init_magnitude


class ModLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, nonlinearity="sin",
                 mod_type="lora_vanilla", omega_0=1., init_mode="uniform"):
        super().__init__()

        self.omega_0 = omega_0
        self.nonlinearity = nonlinearity
        self.mod_type = mod_type
        self.init_mode = init_mode

        self.in_features = in_features
        self.out_features = out_features

        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.lora_alpha = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()


    @torch.no_grad()
    def reset_parameters(self):

        init_magnitude = get_init_magnitude(self.in_features, self.init_mode, self.nonlinearity)
        if not self.init_mode == "first": init_magnitude = init_magnitude / self.omega_0

        if self.init_mode == "uniform" or self.init_mode == "first":
            self.weight.uniform_(-init_magnitude, init_magnitude)
        elif self.init_mode == "normal":
            self.weight.normal_(std=init_magnitude)
        else:
            raise NotImplementedError

        if self.bias is not None:
            self.bias.zero_()
            # self.bias.uniform_(-init_magnitude, init_magnitude)

        self.lora_alpha.fill_(-3. / self.omega_0)


    @torch.no_grad()
    def apply_weight_norm(self, mod_params=None):

        init_magnitude = get_init_magnitude(self.in_features, self.init_mode, self.nonlinearity)
        if not self.init_mode == "first": init_magnitude = init_magnitude / self.omega_0

        if self.init_mode == "uniform" or self.init_mode == "first":

            pass

        elif self.init_mode == "normal":

            self.weight.div_(self.weight.std())
            self.weight.mul_(init_magnitude)

            if mod_params is not None:
                if "lora_w" in mod_params and len(mod_params["lora_w"]) > 0:
                    stddevs = mod_params["lora_w"].std(dim=[1, 2], keepdim=True)
                    mod_params["lora_w"].div_(stddevs)
                    mod_params["lora_w"].mul_(init_magnitude)

        else:
            raise NotImplementedError


    def apply_nonlinearity(self, x):

        if self.nonlinearity == "leaky_relu":
            output = F.leaky_relu(x)
        elif self.nonlinearity == "sin":
            output = torch.sin(self.omega_0 * x)
        elif self.nonlinearity == "selu":
            output = F.selu(x)
        else:
            output = x

        return output


    def get_mod_parameters(self, lora_rank, n, with_bias=False, device="cpu"):

        mod_params = {}

        if lora_rank == 0:

            mod_params["lora_w"] = torch.Tensor(0).to(device).requires_grad_(False)

        elif lora_rank >= min(self.in_features, self.out_features):

            lora_w = torch.Tensor(n, self.out_features, self.in_features)

            with torch.no_grad():
                lora_w_init = self.weight.unsqueeze(0).expand_as(lora_w)
                lora_w.copy_(lora_w_init)

            mod_params["lora_w"] = lora_w.to(device).requires_grad_(True)

        elif lora_rank >= min(self.in_features, self.out_features) and self.mod_type == "from_scratch":

            lora_w = torch.Tensor(n, self.out_features, self.in_features)
            init_magnitude = get_init_magnitude(self.in_features, self.init_mode, self.nonlinearity)

            if not self.init_mode == "first": init_magnitude = init_magnitude / self.omega_0

            if self.init_mode == "uniform" or self.init_mode == "first":
                init_magnitude = init_magnitude
                lora_w.uniform_(-init_magnitude, init_magnitude)
            elif self.init_mode == "normal":
                lora_w.normal_(std=init_magnitude)
            else:
                raise NotImplementedError

            mod_params["lora_w"] = lora_w.to(device).requires_grad_(True)

        elif self.mod_type == "lora_pcfa" or self.mod_type == "lora_pc":

            with torch.no_grad():

                U, s, VH = torch.linalg.svd(self.weight, full_matrices=False)

                if self.mod_type == "lora_pcfa":
                    lora_a_init = VH[:lora_rank, :]
                    lora_b_init = U[:, :lora_rank] @ torch.diag(s[:lora_rank])
                else:
                    S_principal_sqrt = torch.diag(torch.sqrt(s[:lora_rank]))
                    lora_a_init = S_principal_sqrt @ VH[:lora_rank, :]
                    lora_b_init = U[:, :lora_rank] @ S_principal_sqrt

                w_residual = U[:, lora_rank:] @ torch.diag(s[lora_rank:]) @ VH[lora_rank:, :]

                lora_a = torch.Tensor(1 if self.mod_type=="lora_pcfa" else n, lora_rank, self.in_features)
                lora_b = torch.Tensor(n, self.out_features, lora_rank)

                lora_a.copy_(lora_a_init.unsqueeze(0).expand_as(lora_a))
                lora_b.copy_(lora_b_init.unsqueeze(0).expand_as(lora_b))

            self.register_buffer("w_residual", w_residual.to(device).requires_grad_(False))
            mod_params = {"lora_b": lora_b.to(device).requires_grad_(True)}

            if self.mod_type == "lora_pcfa":
                self.register_buffer("lora_a", lora_a.to(device).requires_grad_(False))
            else:
                mod_params.update({"lora_a": lora_a.to(device).requires_grad_(True)})

        elif self.mod_type == "lora_vanilla":

            lora_a = torch.Tensor(n, lora_rank, self.in_features)
            lora_b = torch.Tensor(n, self.out_features, lora_rank)

            a_init_magnitude = get_init_magnitude(self.in_features, self.init_mode, "linear")
            b_init_magnitude = get_init_magnitude(lora_rank, self.init_mode, self.nonlinearity)

            lora_gain = 0.75 if self.nonlinearity == "sin" else 1.
            init_magnitude = np.sqrt(lora_gain * a_init_magnitude * b_init_magnitude)
            if not self.init_mode == "first": init_magnitude = init_magnitude / self.omega_0

            a_init_magnitude = init_magnitude
            b_init_magnitude = init_magnitude

            if self.init_mode == "uniform" or self.init_mode == "first":
                lora_a.uniform_(-a_init_magnitude, a_init_magnitude)
                lora_b.uniform_(-b_init_magnitude, b_init_magnitude)
            elif self.init_mode == "normal":
                lora_a.normal_(std=a_init_magnitude)
                lora_b.normal_(std=b_init_magnitude)
            else:
                raise NotImplementedError

            mod_params = { "lora_a": lora_a.to(device).requires_grad_(True),
                           "lora_b": lora_b.to(device).requires_grad_(True)}

        else:
            raise NotImplementedError

        if self.bias is not None and with_bias:
            bias = torch.Tensor(n, self.out_features).zero_()
            mod_params.update({"bias": bias.to(device).requires_grad_(True)})

        return mod_params

    def forward(self, input, mod_params=None, verbose=False):

        if mod_params is None or len(mod_params) == 0:

            output = torch.matmul(self.weight, input)

        elif "lora_w" in mod_params:

            if len(mod_params["lora_w"]) == 0:
                output = torch.matmul(self.weight, input)
            else:
                output = torch.matmul(mod_params["lora_w"], input)

        elif self.mod_type == "lora_pcfa" or self.mod_type == "lora_pc":

            output = torch.matmul(self.w_residual, input)

            lora_b = mod_params["lora_b"]

            if self.mod_type == "lora_pcfa":
                lora_a = self.lora_a.repeat(lora_b.size(0), 1, 1)
            else:
                lora_a = mod_params["lora_a"]

            lora_rank = lora_b.shape[-1]

            lora_w = torch.matmul(lora_b, lora_a)
            lora_output = torch.matmul(lora_w, input)

            if verbose:
                print(f"lora_rank={lora_rank}, base_output_std={output.std().item()}, lora_output_std={lora_output.std().item()}")

            output = output + lora_output

        else:

            output = torch.matmul(self.weight, input)

            lora_a = mod_params["lora_a"]
            lora_b = mod_params["lora_b"]
            # lora_alpha = self.lora_alpha
            lora_alpha = torch.sigmoid(self.omega_0 * self.lora_alpha)
            lora_rank = lora_b.shape[-1]

            lora_w = self.omega_0 * torch.matmul(lora_b, lora_a)
            lora_output = torch.matmul(lora_w, input)

            if verbose:
                print(f"lora_rank={lora_rank}, base_output_std={output.std().item()}, lora_output_std={lora_output.std().item()}")

            base_alpha = 0. if lora_alpha == 1. else torch.sqrt(1 - lora_alpha)
            output = torch.sqrt(lora_alpha) * lora_output + base_alpha * output

        if verbose:
            print(f"input_std={input.std().item()}, matmul_std={output.std().item()}")

        if self.bias is not None: output = output + self.bias[None, ..., None]

        if mod_params is not None and "bias" in mod_params:
            bias = mod_params["bias"]
            output = output + bias[..., None]
        elif self.bias is not None:
            output = output + self.bias[None, ..., None]

        output = self.apply_nonlinearity(output)

        return output


class ModMLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, n_hidden_layers,
                 inr_type="siren", mod_type="lora_vanilla", pos_encode=None, output_gain=1., use_skip=True):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.pos_encode = pos_encode
        self.inr_type = inr_type
        self.output_gain = output_gain

        self.layers = []

        self.to_output_indices = [n_hidden_layers]
        if use_skip:
            self.to_output_indices = list(range(1, n_hidden_layers, 2)) + self.to_output_indices

        if pos_encode is not None:
            self.layers.append(pos_encode)
        else:
            self.layers.append( ModLinear(
                in_features, hidden_features,
                init_mode="first", nonlinearity="sin", mod_type=mod_type, omega_0=30.))

        for i in range(n_hidden_layers):
            if inr_type == "siren":
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=10. ** 0.5))
            elif inr_type == "vanilla":
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="normal", nonlinearity="leaky_relu", mod_type=mod_type, omega_0=1.))
            else:
                raise NotImplementedError

        self.layers = nn.ModuleList(self.layers)

        self.to_output_layer = ModLinear(
            len(self.to_output_indices) * hidden_features, out_features,
            bias=False, init_mode="uniform", nonlinearity="linear", mod_type=mod_type, omega_0=1.)


    @torch.no_grad()
    def apply_weight_norm(self, mod_params=None):

        for i, layer in enumerate(self.layers):

            if i == 0 or i == len(self.layers) - 1:
                continue

            layer.apply_weight_norm(get_subdict(mod_params, f"layers.{i}."))


    def forward(self, input, mod_params=None, verbose=False):

        instance_shape = input.shape[2:]
        input = input.reshape(input.shape[0], input.shape[1], -1)

        x = self.pos_encode(input) if self.pos_encode else input

        outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
            if i in self.to_output_indices: outputs.append(x)

        x = torch.cat(outputs, dim=1)
        output = self.to_output_layer(x, mod_params=get_subdict(mod_params, f"to_output_layer."), verbose=verbose)
        output = self.output_gain * output.reshape(output.shape[0], output.shape[1], *instance_shape)

        return output


    def get_mod_parameters(self, lora_rank, n, with_bias=False, device="cpu"):

        # if isinstance(lora_rank, list):
        #     assert len(lora_rank) == len(self.layers)

        mod_params = {}

        for i, layer in enumerate(self.layers):
            layer_rank = lora_rank[i] if isinstance(lora_rank, list) else lora_rank
            layer_mod_params = layer.get_mod_parameters(layer_rank, n, with_bias=with_bias, device=device)
            mod_params.update({f"layers.{i}.{k}":v for k, v in layer_mod_params.items()})

        layer_mod_params = self.to_output_layer.get_mod_parameters(self.out_features, n, with_bias, device=device)
        mod_params.update({f"to_output_layer.{k}": v for k, v in layer_mod_params.items()})

        return mod_params


if __name__ == "__main__":

    model = ModMLP(2, 3, 32, n_hidden_layers=2)
    mod_params = model.get_mod_parameters(lora_rank=4, n=3)
    with torch.no_grad():
        query_coordinates = get_coordinate_grid(3, (64, 64)).detach().requires_grad_(False)
        output = model(query_coordinates, mod_params)
        print(output.shape, output.mean())