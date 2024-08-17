import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
import torch.distributions.normal as normal

class Dir_Encoding(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, in_dim, out_dim, scale=25):
        super().__init__()

        self._B = nn.Parameter(torch.randn(
            (in_dim, out_dim)) * scale)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        x = (self.sigmoid(x) * 2 - 1) * torch.pi
        return torch.sin(x)
    


def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    
    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    # Sparse grid encoding
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency", 
                "n_frequencies": n_frequencies
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Identity encoding
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim

      
class Pos_Encoding(nn.Module):
    def __init__(self, cfg, bound, use_pe=False):
        super().__init__()
        self.use_pe = use_pe
        if self.use_pe:
        # Coordinate encoding
            self.pe_fn, self.pe_dim = get_encoder(cfg['pos']['method'], 
                                                n_bins=cfg['pos']['n_bins'])
        else:
            self.pe_fn, self.pe_dim = None, 0

        # Sparse parametric grid encoding
        dim_max = (bound[:,1] - bound[:,0]).max()
        self.resolution = int(dim_max / cfg['grid']['voxel_size'])
        self.grid_fn, self.grid_dim = get_encoder(cfg['grid']['method'], 
                                                  log2_hashmap_size=cfg['grid']['hash_size'], 
                                                  desired_resolution=self.resolution)
        print('Grid size:', self.grid_dim)
    
    def forward(self, pts):
        if self.use_pe:
            pe = self.pe_fn(pts)
        else:
            pe = None

        grid = self.grid_fn(pts)
        return pe, grid
    
    
class SDF(nn.Module):
    def __init__(self, pts_dim, hidden_dim=32, feature_dim=32):
        super().__init__()
        # in_dim = pts_dim + feature_dim
        in_dim = feature_dim
        self.decoder = tcnn.Network(n_input_dims=in_dim,
                                    n_output_dims=1,
                                    network_config={
                                        "otype": "CutlassMLP", # use CutlassMLP if not support FullyFusedMLP
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": hidden_dim,
                                        "n_hidden_layers": 1})
        
    def forward(self, x, f):
        # if not x == None:
        #     f = torch.cat((x, f), -1)

        return self.decoder(f)

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class MLP(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, hidden_dim=32):
        super(MLP, self).__init__()
        # self.hidden_layer = nn.Linear(n_input_dims, hidden_dim+1)
        # self.output_layer = nn.Linear(hidden_dim, n_output_dims)
        self.mlp = nn.Linear(n_input_dims, n_output_dims)

    def forward(self, x):
        x = self.mlp(x)
        return x

class SimpleSDF(nn.Module):
    def __init__(self, cfg, bounding_box, in_dim=3, hidden_dim=32):
        super().__init__()
        self.use_oneblob = cfg['use_oneblob']
        self.use_color = cfg['use_color']
        self.use_rot_scale = cfg['use_rot_scale']
        self.voxel_size = cfg['grid']['voxel_size']

        self.bounding_box = bounding_box
        dim_max = max(bounding_box[:, 1] - bounding_box[:, 0]).cpu().numpy()
        self.sigmoid_alpha = 5 /dim_max
        self.resolution = int(dim_max / self.voxel_size)

        self.grid_fn, self.grid_dim = get_encoder(cfg['grid']['method'], 
                                                  log2_hashmap_size=cfg['grid']['hash_size'], 
                                                  desired_resolution=self.resolution)
        n_input_dims = self.grid_dim
        if self.use_oneblob:
            self.pe_fn, self.pe_dim = get_encoder(cfg['pos']['method'], 
                                                n_bins=cfg['pos']['n_bins'])
            n_input_dims += self.pe_dim
        # self.decoder = MLP(n_input_dims, hidden_dim)
        self.sdf_decoder = MLP(n_input_dims, hidden_dim+1)
        self.opacity_decoder = MLP(hidden_dim, 1)
        # self.sdf2opacity = LaplaceDensity(**cfg['density']).cuda()
        if self.use_rot_scale:
            self.scale_decoder = MLP(hidden_dim, 3)
            self.rot_decoder = MLP(hidden_dim, 4)
        if self.use_color:
            self.color_decoder = MLP(hidden_dim+3, 3)


    def normalization(self, x):
        return 1 / (1 + torch.exp(-self.sigmoid_alpha * x))
        
    def forward(self, x, dir=None, batch=50000, return_opacity=False, return_rot_scale=False, return_color=False):
        num = x.shape[0]
        opacity = []
        sdf = []
        scale = []
        rot = []
        color = []
        for i in range(num // batch + 1):
            start = i * batch
            end = min((i + 1) * batch, num)
            _x = x[start:end]

            if end - start > 0:
                _x = _x.reshape(-1, 3)
                _p = self.normalization(_x)
                _grid = self.grid_fn(_p)
                if self.use_oneblob:
                    _pe = self.pe_fn(_p)
                    ft = self.sdf_decoder(torch.cat((_pe, _grid), dim=-1))
                else:
                    ft = self.sdf_decoder(_grid)
                # _sdf = self.sdf_decoder(ft)
                _sdf = ft[:, :1]
                sdf.append(_sdf)

                if return_opacity:
                    _opa = self.opacity_decoder(ft[:, 1:])
                    # _opa = self.sdf2opacity(_sdf)
                    opacity.append(_opa)

                if return_rot_scale:
                    _scale = self.scale_decoder(ft[:, 1:])
                    _rot = self.rot_decoder(ft[:, 1:])

                    scale.append(_scale)
                    rot.append(_rot)

                if return_color:
                    _d = dir[start:end]
                    _c = self.color_decoder(torch.cat((ft[:, 1:], _d), dim=-1))
                    color.append(_c)

        out = {}
        sdf = torch.cat(sdf, dim=0).cuda().float()
        out['sdf'] = sdf
        if return_opacity:
            opacity = torch.sigmoid(torch.cat(opacity, dim=0).cuda().float())
            out['opacity'] = opacity
        if return_rot_scale:
            scale = torch.cat(scale, dim=0).cuda().float()
            scale = torch.sigmoid(scale) * self.voxel_size * 10
            rot = torch.cat(rot, dim=0).cuda().float()
            rot = torch.nn.functional.normalize(rot)
            out['scale'] = scale
            out['rot'] = rot
        if return_color:
            color = torch.sigmoid(torch.cat(color, dim=0).cuda().float())
            out['color'] = color
        else:
            out['color'] = None

        return out


class SH(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()

        self.decoder = tcnn.Network(n_input_dims=in_dim,
                                          n_output_dims=out_dim,
                                          network_config={
                                            "otype": "CutlassMLP", # use CutlassMLP if not support FullyFusedMLP
                                            "activation": "ReLU",
                                            "output_activation": "None",
                                            "n_neurons": hidden_dim,
                                            "n_hidden_layers": 1})       
        
    def forward(self, x, d, f):
        if not x == None:
            f = torch.cat((x, f), -1)
        else:
            f = torch.cat((d, f), -1)
        out = self.decoder(f)
        square_sum = torch.sum(out**2, dim=-1)
        out = out / torch.sqrt(square_sum)[:, None]
        return out


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()
    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = self.get_alpha()
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
        # return (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
    
    def get_alpha(self):
        # alpha = torch.sigmoid(self.alpha)
        alpha = self.alpha
        return alpha
    

class BellDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        return beta*torch.exp(-beta * sdf) / (1 + torch.exp(-beta * sdf)) ** 2

    def get_beta(self):
        beta = self.beta + self.beta_min
        return beta


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)

class ScaleNetwork(nn.Module):
    def __init__(self, init_val):
        super(ScaleNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.relu(self.variance) * x