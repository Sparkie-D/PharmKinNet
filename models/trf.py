import torch
import torch.nn as nn
import transformers

from .utils import SimNorm, Linear
from .trajectory_gpt2 import GPT2Model

class TRFModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        layer_norm=True,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        config = transformers.GPT2Config(**kwargs)
        self.trf_layer = GPT2Model(config)
        self.simnorm = SimNorm(dim=8)
        self.output_layer = Linear(hidden_dim, output_dim, SimNorm, layer_norm, residual=False)
        self.embed_ln = nn.LayerNorm(4*hidden_dim)
        
    def forward(self, input, attention_mask):
        batch_size, num_timesteps, _ = input.shape
        
        transformer_output = self.trf_layer(
            inputs_embeds=self.embed_ln(input),
            attention_mask=attention_mask
        )
        # partition outputs into (state, action, population, compound)
        x = transformer_output['last_hidden_state'].reshape(batch_size, 4, num_timesteps, self.hidden_dim)
        # trf_output = self.simnorm(x[:, 0].reshape(-1, self.hidden_dim)) # decode state
        trf_output = x[:, 0].reshape(-1, self.hidden_dim)
        return self.output_layer(trf_output).view(batch_size, num_timesteps, -1) # predicted state
