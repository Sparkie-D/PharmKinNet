from .rnn_world_model import RNNWorldModel
# from .trf_world_model import TRFWorldModel
from .mlp_world_model import MLPWorldModel


WORLD_MODEL = {
    'RNN': RNNWorldModel,
    # 'TRF': TRFWorldModel,
    'MLP': MLPWorldModel,
}