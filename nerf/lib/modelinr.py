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

    return init_magnitude




class ModLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, nonlinearity="sin",
                 mod_type="lora_vanilla", omega_0=1., scale_0=1., init_mode="uniform",ca=False):
        super().__init__()

        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.nonlinearity = nonlinearity
        self.mod_type = mod_type
        self.init_mode = init_mode
        self.device = 'cuda'
        self.in_features = in_features
        self.out_features = out_features

        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)).to(self.device) if bias else None
        self.a = nn.Parameter(torch.tensor(1.3))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(1.3))
        # self.d = nn.Parameter(torch.tensor(1.0))
     
        self.weight_magnitude = get_weight_magnitude(self.in_features, self.init_mode, self.nonlinearity)

        self.reset_parameters()
        self.ca = ca
        if self.ca:
            self.calayer=(ChannelAttention(self.out_features, reduction=16))  

        


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

        # self.lora_alpha.fill_(-3. / self.omega_0)



    def apply_nonlinearity(self, x, x1=None):

        if self.nonlinearity == "leaky_relu":
            output = F.leaky_relu(x)
        elif self.nonlinearity == "sigmoid":
            output = torch.sigmoid(x)
        elif self.nonlinearity == "relu":
            output = F.relu(x)
        elif self.nonlinearity == "mid":
            output = self.a*torch.sin(self.omega_0 * x)+ self.c*torch.sin(120* x)
        elif self.nonlinearity == "sin":
            output =torch.sin(self.omega_0 * x)
        elif self.nonlinearity == "selu":
            output = F.selu(x)
        elif self.nonlinearity == "low":
            output =torch.sin(5 * x)
        elif self.nonlinearity == "high":
            output = torch.sin(40 * x)
        elif self.nonlinearity == "Gabor":
            omega = self.omega_0 * x
            # print(x1)
            scale = x * self.scale_0
            output = torch.exp(1j*omega - scale.abs().square())
        else:
            output = x

        return output


    def forward(self, input, mod_params=None, verbose=False):

        if mod_params is None or len(mod_params) == 0:
            if torch.is_complex(input):
                midv = torch.tensor(self.weight_magnitude * self.weight, dtype=torch.cfloat)
                output = torch.matmul(midv, input)
            else:
                input.to(self.weight.device)
                output = torch.matmul(self.weight_magnitude * self.weight, input)
            # output1 = torch.matmul(self.weight1_magnitude * self.weight1, input)


        if verbose:
            print(f"input_std={input.std().item()}, matmul_std={output.std().item()}")

        if mod_params is not None and "bias" in mod_params:
            bias = mod_params["bias"]
            output = output + bias[..., None]

        elif self.bias is not None:
            output = output + self.bias[None, ..., None]


        
        if self.ca:
            output = self.calayer(output)
        output = self.apply_nonlinearity(output)

        return output

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, nonlinearity="sin", init_mode="normal", omega_0=1., scale_0=1.):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  

        self.fc1 = ModLinear(channels, channels // reduction,
                             nonlinearity="relu",
                             init_mode=init_mode,
                             omega_0=omega_0,
                             scale_0=scale_0)

        self.fc2 = ModLinear(channels // reduction, channels,
                             nonlinearity="sigmoid",  
                             init_mode=init_mode,
                             omega_0=omega_0,
                             scale_0=scale_0)

    def forward(self, x, mod_params=None, verbose=False):
        b, c, n = x.size()
        y = self.avg_pool(x).view(b, c).T 

        y = self.fc1(y)  
        y = self.fc2(y)  

        y = y.T.view(b, c, 1)  

        return x * y






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


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=32):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, C, H] → [B, C, 1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h = x.size()
        y = self.pool(x).view(b, c)   
        y = self.fc(y).view(b, c, 1)          
        return x * y 


class ModMLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, n_hidden_layers,
                 inr_type="siren", mod_type="lora_vanilla", first_layer_mode="fourier_features", output_gain=1., skip_mode="cat",mask=True):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.first_layer_mode = first_layer_mode
        self.inr_type = inr_type
        self.output_gain = output_gain
        self.n_hidden_layers = n_hidden_layers
        self.maskcontrol = mask
        self.to_output_indices = [n_hidden_layers]
        self.skip_mode = skip_mode
        if skip_mode in ["mul", "cat"]:
            self.to_output_indices = list(range(1, n_hidden_layers, 2)) + self.to_output_indices

        self.layers = []
        self.masks= []
        self.mask1 = []
        self.mask2 = []
        self.mask3 = []
        self.device = 'cuda'
        self.masknum = 2
        if first_layer_mode == "positional_encoding":
            raise NotImplementedError
        elif first_layer_mode == "fourier_features":
            self.layers.append( ModLinear(
                in_features, hidden_features,
                init_mode="first", nonlinearity="mid", mod_type=mod_type, omega_0=8.,scale_0=40.))
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
                # if i==0:
                #    self.layers.append(nn.LayerNorm(hidden_features))
                if i==0 :
                    self.layers.append( ModLinear(
                        hidden_features, hidden_features,
                        init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1., ca=False))
                else:
                    self.layers.append( ModLinear(
                        hidden_features, hidden_features,
                        init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
            else:
                raise NotImplementedError
        # self.layers.append(ChannelAttention(hidden_features, reduction=16))  
        self.layers = nn.ModuleList(self.layers)


        last_layer_input_size = hidden_features * len(self.to_output_indices) if self.skip_mode == "cat" else hidden_features
        self.to_output_layer = ModLinear(
            last_layer_input_size, out_features,
            bias=False, init_mode="uniform", nonlinearity="linear", mod_type=mod_type, omega_0=1.)
  
        if mask:
            self.masks.append( ModLinear(
                    hidden_features, hidden_features,bias=False,
                    init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
            for p in range(self.masknum-1):
                self.masks.append( ModLinear(
                    hidden_features, hidden_features,bias=True,
                    init_mode="uniform", nonlinearity="relu", mod_type=mod_type, omega_0=1.))
            self.masks = nn.ModuleList(self.masks)
        # self.mask1.append( ModLinear(
        #             hidden_features, hidden_features,bias=False,
        #             init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
        # for p in range(self.masknum-1):
        #     self.mask1.append( ModLinear(
        #             hidden_features, hidden_features,bias=True,
        #             init_mode="uniform", nonlinearity="relu", mod_type=mod_type, omega_0=1.))
        # self.mask1 = nn.ModuleList(self.mask1)
        # self.mask2.append( ModLinear(
        #                 hidden_features, hidden_features,bias=False,
        #                 init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
        # self.mask2 = nn.ModuleList(self.mask2)
        # self.mask3.append( ModLinear(
        #             hidden_features, hidden_features,bias=False,
        #             init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
        # self.mask3 = nn.ModuleList(self.mask3)
        # self.norm = SEBlock1D(256)
        # self.norm = nn.LayerNorm(hidden_features)

        # if mask:
        #     self.mask1.append( ModLinear(
        #             hidden_features, hidden_features,bias=False,
        #             init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))

        #     self.mask2.append( ModLinear(
        #             hidden_features, hidden_features,bias=True,
        #             init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
        #     self.mask1 = nn.ModuleList(self.mask1)


    def forward(self, input, mod_params=None, intermediate_outputs=False, verbose=False):

        instance_shape = input.shape[2:]
        input = input.reshape(input.shape[0], input.shape[1], -1)

        outputs = []
        activation = []
        x = input
        j=len(self.layers)
        # print(j)
        # exit()
        k=0
        for i, layer in enumerate(self.layers):

            # if isinstance(layer, nn.LayerNorm):
            #     x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            # else:
            # print(x.shape)
            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
            # if i==2:
            #     x1 = x
            #     x = x * x
                

            # if i==3:
            #     x = x * x1
            # print(x.shape)
            # exit()
            if self.maskcontrol and i==0:
                x1 = x
                # x3 = x
                # x4 = x
                # x5 = x
                for k, layerm in enumerate(self.masks):
                    # print(f"Mask layer {k} is on device: {x.device}")
                    
                    x1 = layerm(x1, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                    # x3 = self.mask1[k](x3, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                    if k==0:
                        x2 = x1
                        # x4 = x3
                        # x1 = x.transpose(1, 2)
                        # x1 = self.norm(x1)
                        # x1 = x1.transpose(1, 2)
                # x3 = self.mask1[0](x3, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                # x4 = self.mask2[0](x4, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                # x5 = self.mask3[0](x5, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)

            if i==2:
                x = x * x
                # x = self.norm(x)
                # x = x.transpose(1, 2)
                # x = self.norm(x)
                # x = x.transpose(1, 2)
                # print(x.shape)
                
            if self.maskcontrol and i==0:
                x = (x +x2)
                # x = self.norm(x)
            if self.maskcontrol and i==1:
                x = x*x1
               



  
            # if self.maskcontrol and k<self.masknum and i!=0:
            #     x1 =x
            #     masklayer = self.masks[k]
            #     x = masklayer(x, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
            #     x = x * x1
            if intermediate_outputs and i<=j-1:
                y = x
                activation.append(y.reshape(x.shape[0], x.shape[1], *instance_shape))
            # if self.inr_type == "rinr" and i==1:
            #     mid = x
            # elif self.inr_type == "rinr" and i==4:
            #     x=x*mid


            if i in self.to_output_indices: outputs.append(x)


        if self.skip_mode == "mul":
            x = torch.stack(outputs, dim=1).prod(dim=1)
        elif self.skip_mode == "cat":
            x = torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

        output = self.to_output_layer(x, mod_params=get_subdict(mod_params, f"to_output_layer."), verbose=verbose)
        output = self.output_gain * output.reshape(output.shape[0], output.shape[1], *instance_shape)

        if not intermediate_outputs:
            return output.real
        else:
            return output.real, activation




class ModMLP1(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, n_hidden_layers,
                 inr_type="siren", mod_type="lora_vanilla", first_layer_mode="fourier_features", output_gain=1., skip_mode="cat",mask=True):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.first_layer_mode = first_layer_mode
        self.inr_type = inr_type
        self.output_gain = output_gain
        self.n_hidden_layers = n_hidden_layers
        self.maskcontrol = mask
        self.to_output_indices = [n_hidden_layers]
        self.skip_mode = skip_mode
        if skip_mode in ["mul", "cat"]:
            self.to_output_indices = list(range(1, n_hidden_layers, 2)) + self.to_output_indices

        self.layers = []
        self.masks= []
        self.mask1 = []
        # self.mask2 = []
        self.device = 'cuda'
        self.masknum = 2
        if first_layer_mode == "positional_encoding":
            raise NotImplementedError
        elif first_layer_mode == "fourier_features":
            self.layers.append( ModLinear(
                in_features, hidden_features,
                init_mode="first", nonlinearity="mid", mod_type=mod_type, omega_0=8.,scale_0=40.))
        elif first_layer_mode == "chebychev_polynomials":
            self.layers.append(ChebychevInput(in_features, hidden_features))
        elif first_layer_mode == "ffm":
            self.layers.append(FFM(in_features, out_features))
        elif first_layer_mode == "PE":
            self.layers.append(PE(in_features, hidden_features))


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
                if i==0 :
                    self.layers.append( ModLinear(
                        hidden_features, hidden_features,
                        init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1., ca=False))
                else:
                    self.layers.append( ModLinear(
                        hidden_features, hidden_features,
                        init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
            else:
                raise NotImplementedError
        self.layers = nn.ModuleList(self.layers)


        last_layer_input_size = hidden_features * len(self.to_output_indices) if self.skip_mode == "cat" else hidden_features
        self.to_output_layer = ModLinear(
            last_layer_input_size, out_features,
            bias=False, init_mode="uniform", nonlinearity="linear", mod_type=mod_type, omega_0=1.)
  
        # if mask:
        #     self.masks.append( ModLinear(
        #         in_features, hidden_features,
        #         init_mode="first", nonlinearity="low", mod_type=mod_type, omega_0=8.,scale_0=40.))
        #     # self.masks.append( ModLinear(
        #     #         hidden_features, hidden_features,bias=True,
        #     #         init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
        #     for p in range(self.masknum):
        #         self.masks.append( ModLinear(
        #             hidden_features, hidden_features,bias=True,
        #             init_mode="uniform", nonlinearity="relu", mod_type=mod_type, omega_0=1.))
        #     self.masks = nn.ModuleList(self.masks)
        if mask:
            self.masks.append( ModLinear(
                 hidden_features, hidden_features,
                 init_mode="uniform", nonlinearity="sin", mod_type=mod_type, omega_0=1.))
            self.masks = nn.ModuleList(self.masks)
            self.mask1.append( ModLinear(
                 hidden_features, hidden_features,
                 init_mode="uniform", nonlinearity="mid", mod_type=mod_type, omega_0=1.))
            self.mask1 = nn.ModuleList(self.mask1)



    def forward(self, input, mod_params=None, intermediate_outputs=False, verbose=False):

        instance_shape = input.shape[2:]
        input = input.reshape(input.shape[0], input.shape[1], -1)

        outputs = []
        activation = []
        x = input
        j=len(self.layers)

        k=0
        for i, layer in enumerate(self.layers):


            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
            if i==0:
                x1= x
                x1 = self.masks[0](x1, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                x = x*x1
            if i==1:
                x1= x
                x1 = self.mask1[0](x1, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                x = x+x1
            # if self.maskcontrol and i==1:
            #     x = x * x1
            # if self.maskcontrol and i==0:
            #     x = x*x2

            # if i==2:
            #     x=x*x 

            if intermediate_outputs and i==1:
                y = x
                activation.append(y.reshape(x.shape[0], x.shape[1], *instance_shape))


            if i in self.to_output_indices: outputs.append(x)


        if self.skip_mode == "mul":
            x = torch.stack(outputs, dim=1).prod(dim=1)
        elif self.skip_mode == "cat":
            x = torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError

        output = self.to_output_layer(x, mod_params=get_subdict(mod_params, f"to_output_layer."), verbose=verbose)
        output = self.output_gain * output.reshape(output.shape[0], output.shape[1], *instance_shape)

        if not intermediate_outputs:
            return output.real
        else:
            return output.real, activation