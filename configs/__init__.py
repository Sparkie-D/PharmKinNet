from .rnn_config import *
from .trf_config import *

CONFIG = {
    'pretrain-RNN-XS': pretrain_rnn_config_xs,
    'pretrain-RNN-S': pretrain_rnn_config_s,
    'pretrain-RNN-M': pretrain_rnn_config_m,
    'finetune-RNN-XS': finetune_rnn_config_xs,
    'finetune-RNN-S': finetune_rnn_config_s,
    'finetune-RNN-M': finetune_rnn_config_m
}