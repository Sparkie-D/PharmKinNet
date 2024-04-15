from tqdm import tqdm
import pandas as pd
import torch
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
import sys
sys.path.append('.')

from dynamics import WORLD_MODEL
from data.dataset import load_data
from utils.trainer import Trainer
from utils.logger import Logger, make_log_dirs
from configs import CONFIG


@torch.no_grad()
def rollout(world_model, dataset):
    ven_preds = []
    organ_preds = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    for observations, actions, populations, compounds, organs, timesteps in tqdm(data_loader):
        ven_pred, organ_pred = world_model.rollout_batch(observations, actions, populations, compounds, organs, timesteps, rollout_mode="all")
        ven_preds.append(ven_pred.detach().cpu().numpy())
        organ_preds.append(organ_pred.detach().cpu().numpy())
    ven_preds = np.concatenate(ven_preds)
    organ_preds = np.concatenate(organ_preds)
    
    ven_preds =  dataset.get_normalizer().unnormalize(ven_preds, "observation").clip(min=0).cpu().numpy()
    organ_preds =  dataset.get_normalizer().unnormalize(organ_preds, "organ").clip(min=0).cpu().numpy()
    
    return ven_preds, organ_preds


def save_predict_data(data, ven_preds, organ_preds, save_path):
    pass


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
    parser.add_argument('--task', type=str, default='rollout')
    parser.add_argument('--obs-mode', type=str, default='ven')
    parser.add_argument('--rollout-mode', type=str, default='all')
    parser.add_argument('--data-path', type=str, default='data/data_kfold/35/35_train_data_0.csv')
    parser.add_argument('--save-path', type=str, default='data/data_kfold/35/35_train_data_0_predict.csv')
    parser.add_argument('--normalizer-path', type=str, default='log/pretrain/RNN&model_size=XS/seed_1&timestamp_24-0319-195400/model/normalizer.pkl')
    parser.add_argument('--model-path', type=str, default='log/finetune/RNN&model_size=XS&drug_id=4/seed_1&timestamp_24-0320-150154/model/models_600.pth')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--info', type=str, default=None)

    known_args, _ = parser.parse_known_args()
    for arg_key, default_value in CONFIG[known_args.model_type + '-' + known_args.model_size].items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    
    args = parser.parse_args()

    return args


def main(args=get_args()):
    # set seed
    set_seed_everywhere(args.seed)    
    
    # init dataset
    rollout_dataset = load_data(args.data_path, -1, args.obs_mode, args.device)
    rollout_dataset.load_normalizer(args.normalizer_path)
    
    for k, v in rollout_dataset.feature_dims.items():
        args.__dict__[k] = v
    
    # init logger
    log_dirs = make_log_dirs(args.task, args.model_type, args.seed, vars(args), record_params=['model_size'])
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
    world_model.load_model(args.model_path)
    
    ven_preds, organ_preds = rollout(world_model, rollout_dataset)
    data = pd.read_csv(args.data_path)
    save_predict_data(data, ven_preds, organ_preds, args.save_path)

if __name__ == '__main__':
    main()