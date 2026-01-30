import sys
import os
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

        self.log_file = os.path.join(os.getcwd(), "param_log.txt")


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
            output = self.a*torch.sin(self.omega_0 * x)+ self.c*torch.sin(120* x)+self.b*torch.sin(40 * x)
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
        if not torch.is_grad_enabled() and self.nonlinearity == "mid":  
            print(f"a={self.a.item():.4f}, b={self.b.item():.4f}, c={self.c.item():.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"a={self.a.item():.4f}, b={self.b.item():.4f}, c={self.c.item():.4f}\n")
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


class PE1(nn.Module):
    def __init__(self, input_dim: int, multires: int = 12, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.multires = multires
        self.include_input = include_input
        # 2^0, 2^1, ..., 2^(N-1)
        self.register_buffer(
            "freq_bands",
            (2.0 ** torch.arange(multires)).float() * math.pi
        )
        base = 2 * input_dim * multires
        self.out_dim = base + (input_dim if include_input else 0)

    def forward(self, x):  # x: (B, D, N)
        B, D, N = x.shape
        x_exp = x.unsqueeze(-2)                                  # (B, D, 1, N)
        freqs = self.freq_bands.view(1, 1, self.multires, 1).to(x.device)
        x_proj = x_exp * freqs                                   # (B, D, Nfreq, N)
        sin, cos = torch.sin(x_proj), torch.cos(x_proj)
        pe = torch.cat([sin, cos], dim=2).view(B, D * 2 * self.multires, N)
        if self.include_input:
            pe = torch.cat([x, pe], dim=1)
        return pe


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


            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)

            if self.maskcontrol and i==0:
                x1 = x

                for k, layerm in enumerate(self.masks):
                    # print(f"Mask layer {k} is on device: {x.device}")
                    
                    x1 = layerm(x1, mod_params=get_subdict(mod_params, f"layers.{k}."), verbose=verbose)
                    if k==0:
                        x2 = x1
  

            if self.maskcontrol and i==2:
                x = x * x

                
            if self.maskcontrol and i==0:
                x = (x +x2)
            if self.maskcontrol and i==1:
                x = x*x1
               



  

            if intermediate_outputs and i<=j-1:
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
        return output.real
        # if not intermediate_outputs:
        #     return output.real
        # else:
        #     return output.real, activation


"""
For  RINR
"""

class ModMLP1(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, n_hidden_layers,
                 inr_type="rinr", mod_type="lora_vanilla", first_layer_mode="PE", output_gain=1., skip_mode="cat",mask=True):
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
        self.masknum = 3
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
                self.pe = PE1(input_dim=in_features, multires=10, include_input=True)
                pe_dim = self.pe.out_dim
                self.layers.append( ModLinear(
                    pe_dim, hidden_features,
                    init_mode="uniform", nonlinearity="relu", mod_type=mod_type, omega_0=1.))


        for i in range(3):
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
  

        if mask:
            for i in range(3):
                self.masks.append( ModLinear(
                    pe_dim, pe_dim,bias = False,
                    init_mode="uniform", nonlinearity="relu", mod_type=mod_type, omega_0=1.))
            self.masks = nn.ModuleList(self.masks)




    def forward(self, input, mod_params=None, intermediate_outputs=False, verbose=False):

        instance_shape = input.shape[2:]
        input = input.reshape(input.shape[0], input.shape[1], -1)

        outputs = []
        activation = []
        x = input
        j=len(self.layers)
        x = self.pe(x) 
        x1 = x
        k=0
        for i, layer in enumerate(self.masks):
            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
        x = x*x1
        for i, layer in enumerate(self.layers):


            x = layer(x, mod_params=get_subdict(mod_params, f"layers.{i}."), verbose=verbose)
 
 


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




class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class INRG(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        instance_shape = coords.shape[2:]
        coords = coords.reshape(coords.shape[0], coords.shape[1], -1)
        # print(coords.shape)
        coords = coords.permute(0, 2, 1)  # Change to (B, N, D) format
        output = self.net(coords)
        output = output.permute(0, 2, 1)
        # print(output.shape)
        output =  output.reshape(output.shape[0], output.shape[1], *instance_shape)
        if self.wavelet == 'gabor':
            return output.real
         
        return output


class GaussLayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with Gaussian non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.scale = scale
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return torch.exp(-(self.scale*self.linear(input))**2)
    

class INRE(nn.Module):
    def __init__(self, in_features,
                 hidden_features, hidden_layers, 
                 out_features,outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=30.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        self.complex = False
        self.nonlin = GaussLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
                        
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        instance_shape = coords.shape[2:]
        coords = coords.reshape(coords.shape[0], coords.shape[1], -1)
        # print(coords.shape)
        coords = coords.permute(0, 2, 1) 
        output = self.net(coords)
        output = output.permute(0, 2, 1)
        # print(output.shape)
        output =  output.reshape(output.shape[0], output.shape[1], *instance_shape)         
        return output


class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class INRS(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        instance_shape = coords.shape[2:]
        coords = coords.reshape(coords.shape[0], coords.shape[1], -1)
        # print(coords.shape)
        coords = coords.permute(0, 2, 1)  
        output = self.net(coords)
        output = output.permute(0, 2, 1)
        # print(output.shape)
        output =  output.reshape(output.shape[0], output.shape[1], *instance_shape)      
        return output


"""
If the SMN in Class ModLinear does not work, try this SMN.
Plese check they are the same before use.
"""
class INRN(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=8.0, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer
        self.firlin = Oslayer 
        self.net = []
        self.mask=[]
        self.net.append(self.firlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))
        # print(in_features)
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.ModuleList(self.net)
        self.mask.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        semask = nn.Linear(hidden_features,
                                     hidden_features,
                                     dtype=dtype)
            
        with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
        self.mask.append(semask)
        self.mask.append(nn.ReLU())
        self.mask = nn.ModuleList(self.mask)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        instance_shape = coords.shape[2:]
        coords = coords.reshape(coords.shape[0], coords.shape[1], -1)
        # print(coords.shape)
        coords = coords.permute(0, 2, 1)
        x = coords
        for i, layer in enumerate(self.net):
            # print(coords.shape)
            x = layer(x)
            if i==0:
                x1 = x
                for k, layerm in enumerate(self.mask):
                    x1 = layerm(x1)
                    if i==0:
                        x2 = x1
            if i==0:
                x = (x +x2)
            if i==1:
                x = x*x1
            if i==2:
                x=x*x
        output = x    

        output = output.permute(0, 2, 1)
        # print(output.shape)
        output =  output.reshape(output.shape[0], output.shape[1], *instance_shape)      
        return output


class Oslayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=8.0, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.a = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(1.))
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        x = self.linear(input)
        return self.a*torch.sin(self.omega_0 * x)+self.b*torch.sin(40*x)+self.c*torch.sin(120*x)