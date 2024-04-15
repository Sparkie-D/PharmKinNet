import torch
import torch.nn as nn
import torch.nn.functional as F


def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
    

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'silu': nn.SiLU,
    'none': nn.Identity,
    'swish': Swish,
}


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, dim=8):
		super().__init__()
		self.dim = dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
	

class Linear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        act_fn,
        layer_norm=True,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        if act_fn is None:
            self.act_fn = nn.Identity()
        else:
            self.act_fn = act_fn()
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.residual = residual
    
    def forward(self, x):
        y = self.act_fn(self.linear(x))
        y = self.dropout(y)
        if self.residual:
            y = x + y
        y = self.layer_norm(y)
        return y
    
