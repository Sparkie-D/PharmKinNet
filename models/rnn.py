import torch.nn as nn

from .utils import SimNorm, Linear


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        rnn_num_layers=2,
        layer_norm=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        self.simnorm = SimNorm(dim=8)
        self.output_layer = Linear(hidden_dim, output_dim, SimNorm, layer_norm, residual=False)

    def forward(self, input, h_state=None):
        batch_size, num_timesteps, _ = input.shape
        rnn_output, h_state = self.rnn_layer(input, h_state)
        rnn_output = self.simnorm(rnn_output.reshape(-1, self.hidden_dim))
        output = self.output_layer(rnn_output)
        output = output.view(batch_size, num_timesteps, -1)
        return output, h_state