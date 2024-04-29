import os
import torch
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('.')

from dynamics import WORLD_MODEL
from data.dataset import load_data
from utils.trainer import Trainer
from utils.logger import Logger, make_log_dirs
from configs import CONFIG


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='RNN')
    parser.add_argument('--model-size', type=str, default='XS')
    parser.add_argument('--task', type=str, default='pretrain')
    parser.add_argument('--obs-mode', type=str, default='ven')
    parser.add_argument('--log-id', type=str, default="default")
    parser.add_argument('--train-path', type=str, default='data/skeleton_train.csv')
    parser.add_argument('--valid-path', type=str, default='data/skeleton_valid.csv')
    parser.add_argument('--normalizer-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--info', type=str, default=None)

    known_args, _ = parser.parse_known_args()
    for arg_key, default_value in CONFIG[known_args.task + '-' + known_args.model_type + '-' + known_args.model_size].items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    
    args = parser.parse_args()

    return args


def main(args=get_args()):
    # set seed
    set_seed_everywhere(args.seed)    
    
    # init dataset
    train_dataset = load_data(args.train_path, args.train_seq_len, args.obs_mode, args.device)
    valid_dataset = load_data(args.valid_path, args.valid_seq_len, args.obs_mode, args.device)
    valid_dataset.set_normalizer(train_dataset.get_normalizer())
    args.max_ep_len = train_dataset._max_ep_len
    if args.normalizer_path is not None:
        train_dataset.load_normalizer(args.normalizer_path)
        valid_dataset.load_normalizer(args.normalizer_path)
    
    for k, v in train_dataset.feature_dims.items():
        args.__dict__[k] = v
    
    # init logger
    log_dirs = make_log_dirs(args.task, args.model_type, args.seed, vars(args), record_params=['model_size', 'log_id'])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "model_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    # init model
    world_model = WORLD_MODEL[args.model_type](args)
    # if args.model_path is not None:
    if args.model_path is not None:
        world_model.load_model(args.model_path)
    
    # init trainer
    trainer = Trainer(train_dataset, valid_dataset, logger, world_model)
    trainer.train(
        train_mode=args.train_mode,
        train_epoch=args.train_epochs, 
        log_interval=args.log_interval, 
        plot_interval=args.plot_interval, 
        save_interval=args.save_interval,
        batch_size=args.batch_size)
    

if __name__ == '__main__':
    main()