import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.util import get_subdict, get_coordinate_grid
import math


def get_weight_magnitude(in_features, init_mode, nonlinearity):

    if in_features == 0: return 0.

    if init_mode == "uniform":
        init_magnitude = np.sqrt(6. / in_features)
    elif init_mode == "normal":
        gain = torch.nn.init.calculate_gain(nonlinearity) if nonlinearity != "sin" and "Gabor" else 1.
        init_magnitude = gain * np.sqrt(1. / in_features)
    elif init_mode == "first":
        init_magnitude = 4.
    else:
        raise NotImplementedError
    # if nonlinearity == "Gabor" and init_mode != "first":
    #     init_magnitude = torch.tensor(init_magnitude, dtype=torch.cfloat)
        # init_complex = torch.complex(init_magnitude, torch.zeros_like(init_magnitude))
    return init_magnitude


class ModLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, nonlinearity="sin",
                 mod_type="lora_vanilla", omega_0=1., scale_0=1., init_mode="uniform"):
        super().__init__()

        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.nonlinearity = nonlinearity
        self.mod_type = mod_type
        self.init_mode = init_mode

        self.in_features = in_features
        self.out_features = out_features

        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.bias1 = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.lora_alpha = nn.Parameter(torch.Tensor(1))
     
        self.weight_magnitude = get_weight_magnitude(self.in_features, self.init_mode, self.nonlinearity)
        self.weight1_magnitude = get_weight_magnitude(self.in_features, self.init_mode, self.nonlinearity)

        self.reset_parameters()
        if nonlinearity == "Gabor":
            self.double_control= False
        else:
            self.double_control = False



    @torch.no_grad()
    def reset_parameters(self):

        if self.init_mode == "first":
            init_magnitude = np.sqrt(9. / self.in_features)
            self.weight.uniform_(-init_magnitude, init_magnitude)
        elif self.init_mode == "uniform":
            self.weight.uniform_(-1, 1)
        elif self.init_mode == "normal":
            self.weight.normal_(std=1.)
        else:
            raise NotImplementedError

        if self.bias is not None:
            self.bias.zero_()

        self.lora_alpha.fill_(-3. / self.omega_0)


    @torch.no_grad()
    def apply_weight_norm(self, mod_params=None):

        if self.init_mode == "uniform" or self.init_mode == "first":

            pass

        elif self.init_mode == "normal":

            out_features, in_features = self.weight.shape
            fro_norm = np.sqrt(in_features)

            demod = torch.rsqrt(self.weight.pow(2).sum(dim=1, keepdims=True) + 1e-8)
            self.weight.mul_(demod * fro_norm)
            # if self.double_control:
            #     demod = torch.rsqrt(self.weight1.pow(2).sum(dim=1, keepdims=True) + 1e-8)
            #     self.weight1.mul_(demod1 * fro_norm)
            if mod_params is not None:
                if "lora_w" in mod_params and len(mod_params["lora_w"]) > 0:
                    demod = torch.rsqrt(mod_params["lora_w"].pow(2).sum(dim=2, keepdims=True) + 1e-8)
                    mod_params["lora_w"].mul_(demod * fro_norm)
                    # if self.double_control and "lora_w1" in mod_params and len(mod_params["lora_w1"]) > 0:
                    #     demod1 = torch.rsqrt(mod_params["lora_w1"].pow(2).sum(dim=2, keepdims=True) + 1e-8)
                    #     mod_params["lora_w1"].mul_(demod1 * fro_norm)

        else:
            raise NotImplementedError


    def apply_nonlinearity(self, x, x1=None):

        if self.nonlinearity == "leaky_relu":
            output = F.leaky_relu(x)
        elif self.nonlinearity == "relu":
            output = F.relu(x)
        elif self.nonlinearity == "sin":
            output = torch.sin(self.omega_0 * x)
        elif self.nonlinearity == "selu":
            output = F.selu(x)
        elif self.nonlinearity == "Gabor":
            omega = self.omega_0 * x
            # print(x1)
            scale = x * self.scale_0
            output = torch.exp(1j*omega - scale.abs().square())
        else:
            output = x

        return output


    def get_mod_parameters(self, lora_rank, n, with_bias, device="cpu"):

        mod_params = {}


        if self.bias is not None and with_bias:
            bias = torch.Tensor(n, self.out_features).zero_()
            mod_params.update({"bias": bias.to(device).requires_grad_(True)})

        return mod_params

    def forward(self, input, mod_params=None, verbose=False):

        if mod_params is None or len(mod_params) == 0:
            if torch.is_complex(input):
                midv = torch.tensor(self.weight_magnitude * self.weight, dtype=torch.cfloat)
                output = torch.matmul(midv, input)
            else:
                output = torch.matmul(self.weight_magnitude * self.weight, input)
            # output1 = torch.matmul(self.weight1_magnitude * self.weight1, input)


        if verbose:
            print(f"input_std={input.std().item()}, matmul_std={output.std().item()}")

        if mod_params is not None and "bias" in mod_params:
            bias = mod_params["bias"]
            output = output + bias[..., None]
            if self.double_control:
                output1 = output1 + bias1[..., None]
        elif self.bias is not None:
            output = output + self.bias[None, ..., None]
            if self.double_control:
                output1 = output1 + self.bias1[None, ..., None]

        
        if self.double_control:
            output = self.apply_nonlinearity(output, output1)
            # print(1)
        else:
            output = self.apply_nonlinearity(output)

        return output


class ChebychevInput(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, poly_degree: int = 256):
        """
        Learnable Activation Block with Chebyshev polynomials.
        :param input_dim: Dimension of input features
        :param output_dim: Dimension of output features
        :param poly_degree: Degree of Chebyshev polynomial expansion
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_degree = poly_degree

        # Learnable Chebyshev coefficients
        self.coefficients = nn.Parameter(torch.randn(output_dim, input_dim, poly_degree + 1))
        fan_in = (input_dim * (poly_degree + 1))
        self.weight_magnitude = get_weight_magnitude(fan_in, "uniform", "linear")

        nn.init.uniform_(self.coefficients, -1., 1.)

        self.register_buffer("arange", torch.arange(0, poly_degree + 1, 1))

    def chebyshev_polynomials(self, x):
        """Computes Chebyshev polynomials up to a given degree."""
        batch_size, features, n_samples = x.shape
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.poly_degree + 1)  # Shape: (B, input_dim, n_samples, poly_degree)

        x = x.acos()
        x *= self.arange
        x = x.cos()

        return x

    def forward(self, x, mod_params=None, verbose=False):
        """Forward pass of the Learnable Activation Block."""
        # x = torch.tanh(x)  # Normalize input to [-1, 1] for Chebyshev polynomials

        # Compute Chebyshev polynomial basis
        T = self.chebyshev_polynomials(x)  # Shape: (B, input_dim, n_samples, poly_degree)

        # Apply learnable coefficients
        y = torch.einsum('bisp, oip -> bos', T, self.weight_magnitude * self.coefficients)  # Element-wise multiplication and sum

        return y


class FFM(nn.Module):    
    def __init__(self, in_features, out_features, coordinate_scales=[1.0, 1.0]):
        super().__init__()

        self.num_freq = out_features // 2
        self.out_features = out_features
        self.coordinate_scales = nn.Parameter(torch.tensor(coordinate_scales).unsqueeze(dim=0))
        self.coordinate_scales.requires_grad = False
        self.linear = nn.Linear(in_features, self.num_freq, bias=False)
        self.init_weights()
        self.linear.weight.requires_grad = False
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.normal_(std=1, mean=0)

    def forward(self, input, mod_params=None, verbose=False):
        print("input", input.shape)
        print("coordinate_scales", self.coordinate_scales.shape)
        return torch.cat((np.sqrt(2)*torch.sin(self.linear(self.coordinate_scales*input)), 
                          np.sqrt(2)*torch.cos(self.linear(self.coordinate_scales*input))), dim=-1)




class PE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, L: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Leng = L // (2 * input_dim)
        self.freq_bands = torch.arange(self.Leng).float() * math.pi

    def forward(self, x, mod_params=None, verbose=False):

        B, D, N = x.shape
        # print( x.shape)
        x = x.unsqueeze(-2)                      
        freqs = self.freq_bands.to(x.device).view(1, 1, self.Leng, 1) 
        x_proj = x * freqs                       

        sin = torch.sin(x_proj)                     
        cos = torch.cos(x_proj)                  

        pe = torch.cat([sin, cos], dim=2)          
        pe = pe.view(B, D * 2 * self.Leng, N)     
        # print(pe.shape)   
        return pe


class FullPE(nn.Module):
    def __init__(self, input_dim: int, L: int = 9):

        super().__init__()
        self.input_dim = input_dim
        self.max_freq = 2 ** L  
        self.L = L
        self.freq_bands = torch.arange(0, self.max_freq + 1).float() * math.pi  # [0, 1, ..., 2^L] * π
        self.linear = nn.Linear(input_dim, self.max_freq + 1, bias=False)

    def forward(self, x):
     
        B, D, N = x.shape  
        x = x.unsqueeze(-2)  
        freqs = self.freq_bands.to(x.device).view(1, 1, -1, 1)  
        x_proj = x * freqs  
        cosp = torch.cos(x_proj)  
        pe = cosp.view(B, D * (self.max_freq + 1), N) 
        pe =pe.unquzzeze(-1)
        pe = pe.permute(0, 2, 1, 3)
        pe = pe.expand(-1, -1, -1, 3)
        return pe


class ModMLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, n_hidden_layers,
                 inr_type="siren", mod_type="lora_vanilla", first_layer_mode="fourier_features", output_gain=1., skip_mode="cat"):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.first_layer_mode = first_layer_mode
        self.inr_type = inr_type
        self.output_gain = output_gain
        self.n_hidden_layers = n_hidden_layers

        self.to_output_indices = [n_hidden_layers]
        self.skip_mode = skip_mode
        if skip_mode in ["mul", "cat"]:
            self.to_output_indices = list(range(1, n_hidden_layers, 2)) + self.to_output_indices

        self.layers = []
        if first_layer_mode == "positional_encoding":
            raise NotImplementedError
        elif first_layer_mode == "fourier_features":
            self.layers.append( ModLinear(
                in_features, hidden_features,
                init_mode="first", nonlinearity="sin", mod_type=mod_type, omega_0=30.,scale_0=40.))
        elif first_layer_mode == "chebychev_polynomials":
            self.layers.append(ChebychevInput(in_features, hidden_features))
        elif first_layer_mode == "ffm":
            self.layers.append(FFM(in_features, out_features))
        elif first_layer_mode == "PE":
            self.layers.append(PE(in_features, hidden_features))


        if inr_type == "rinr":
            for i in range(2):
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,bias=False,
                    init_mode="normal", nonlinearity="relu", mod_type=mod_type, omega_0=1.))
            self.layers.append( ModLinear(
                hidden_features, hidden_features,bias=False,
                init_mode="normal", nonlinearity="linear", mod_type=mod_type, omega_0=1.))
        for i in range(n_hidden_layers):
            if inr_type == "wire":
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="uniform", nonlinearity="Gabor", mod_type=mod_type, omega_0=30., scale_0=40.))
            elif inr_type == "vanilla":
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="normal", nonlinearity="leaky_relu", mod_type=mod_type, omega_0=1.))
            elif inr_type == "rinr":
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="normal", nonlinearity="relu", mod_type=mod_type))
            elif inr_type == "siren":
                # if i==1:
                #     self.layers.append(nn.LayerNorm(hidden_features))
                self.layers.append( ModLinear(
                    hidden_features, hidden_features,
                    init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=30.))
            else:
                raise NotImplementedError
            
        # self.layers.append(conv1d(in_channels=1,out_channles= 2,kernal_szie=3, stride=2, padding=0))
        self.layers = nn.ModuleList(self.layers)

        # self.to_output_layer = ModLinear(
        #      len(self.to_output_indices) * hidden_features, out_features,
        #      bias=False, init_mode="uniform", nonlinearity="linear", mod_type=mod_type, omega_0=1.)

        last_layer_input_size = hidden_features * len(self.to_output_indices) if self.skip_mode == "cat" else hidden_features
        self.to_output_layer = ModLinear(
            last_layer_input_size, out_features,
            bias=False, init_mode="uniform", nonlinearity="linear", mod_type=mod_type, omega_0=1.)


    @torch.no_grad()
    def apply_weight_norm(self, mod_params=None):

        for i, layer in enumerate(self.layers):

            if i == 0 or i == len(self.layers) - 1:
                continue

            layer.apply_weight_norm(get_subdict(mod_params, f"layers.{i}."))


    def forward(self, input, mod_params=None, intermediate_outputs=False, verbose=False):

        instance_shape = input.shape[2:]
        input = input.reshape(input.shape[0], input.shape[1], -1)

        outputs = []
        activation = []
        x = input
        j=len(self.layers)
        for i, layer in enumerate(self.layers):
            # if self.convc==True and i==7:
            #     x = x.unsqueeze(1)
            #     x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
            #     x = x.permute(0, 2, 1, 3)
            #     x 
            if isinstance(layer, nn.LayerNorm):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
                # x = layer(x)
            else:
                x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
            if intermediate_outputs and i!=j:
                y = x
                activation.append(y.reshape(x.shape[0], x.shape[1], *instance_shape))
            if self.inr_type == "rinr" and i==1:
                mid = x
            elif self.inr_type == "rinr" and i==4:
                x=x*mid


            if i in self.to_output_indices: outputs.append(x)


        if self.skip_mode == "mul":
            x = torch.stack(outputs, dim=1).prod(dim=1)
        elif self.skip_mode == "cat":
            x = torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

        output = self.to_output_layer(x, mod_params=get_subdict(mod_params, f"to_output_layer."), verbose=verbose)
        output = self.output_gain * output.reshape(output.shape[0], output.shape[1], *instance_shape)
        # if self.inr_type == "rinr":
        #     output = output.real
        if not intermediate_outputs:
            return output.real
        else:
            return output.real, activation


    # def get_mod_parameters(self, lora_rank, n, with_bias, device="cpu"):

    #     # if isinstance(lora_rank, list):
    #     #     assert len(lora_rank) == len(self.layers)

    #     mod_params = {}

    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, ChebychevInput): continue
    #         # layer_rank = lora_rank[i] if isinstance(lora_rank, list) else lora_rank
    #         layer_mod_params = layer.get_mod_parameters(layer_rank, n, with_bias, device=device)
    #         mod_params.update({f"layers.{i}.{k}":v for k, v in layer_mod_params.items()})

    #     layer_mod_params = self.to_output_layer.get_mod_parameters(self.out_features, n, with_bias, device=device)
    #     mod_params.update({f"to_output_layer.{k}":v for k, v in layer_mod_params.items()})

    #     return mod_params


if __name__ == "__main__":

    model = ModMLP(2, 3, 32, n_hidden_layers=2)
    mod_params = model.get_mod_parameters(lora_rank=4, n=3)
    with torch.no_grad():
        query_coordinates = get_coordinate_grid(3, (64, 64)).detach().requires_grad_(False)
        output = model(query_coordinates, mod_params)
        print(output.shape, output.mean())