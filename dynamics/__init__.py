from .rnn_world_model import RNNWorldModel
from .trf_world_model import TRFWorldModel

WORLD_MODEL = {
    'RNN': RNNWorldModel,
    'TRF': TRFWorldModel
}