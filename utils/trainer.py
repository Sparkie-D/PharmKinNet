import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.data_utils import *

class Trainer(object):
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        logger,
        world_model
    ) -> None:
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.logger = logger
        
        self.world_model = world_model

        
    def train(self, train_mode, train_epoch, log_interval=100, plot_interval=10, save_interval=10, batch_size=256):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        step = 0
        for epoch in range(train_epoch):
            if train_mode == 'schedule':
                prob = epoch / train_epoch
            elif train_mode == "auto-regression":
                prob = 1
            else:
                prob = 0
            # evaluate & plot
            self.world_model._torch_train(False)
            eval_metrics = self.evaluate(batch_size, plot_interval, epoch)
            for k, v in eval_metrics.items():
                self.logger.logkv_mean(k, v)
            self.logger.logkv_mean('epoch', epoch)
            self.logger.set_timestep(step)
            self.logger.dumpkvs()

            if epoch % save_interval == 0:
                self.save_predict_csv(epoch, mode="valid")
                self.save_predict_csv(epoch, mode="train")
                self.save(epoch)
            
            self.world_model._torch_train(True)
            for observations, actions, populations, compounds, organs, timesteps in tqdm(train_loader, desc=f'Epoch #{epoch}/{train_epoch}'):
                metrics = self.world_model.train_batch(observations, actions, populations, compounds, organs, timesteps, prob=prob)
                for k, v in metrics.items():
                    self.logger.logkv_mean(k, v)
                step += 1

                if step % log_interval == 0:
                    self.logger.set_timestep(step)
                    self.logger.dumpkvs()

    @torch.no_grad()
    def save_predict_csv(self, epoch, mode="valid"):
        self.world_model._torch_train(False) 
        if mode == "valid":
            dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=512, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=512, shuffle=False)
        obs_preds, organ_preds = [], []
        for observations, actions, populations, compounds, organs, timesteps in tqdm(dataloader):
            obs_pred, organ_pred = self.world_model.rollout_batch(observations, actions, populations, compounds, organs, timesteps, rollout_mode="all")
            obs_pred = torch.cat([observations[:, 0: 1], obs_pred], dim=1)
            obs_preds.append(self.train_dataset.get_normalizer().unnormalize(obs_pred, "observation").clip(min=0).cpu().numpy())
            organ_pred = torch.cat([organs[:, 0: 1], organ_pred], dim=1)
            organ_preds.append(self.train_dataset.get_normalizer().unnormalize(organ_pred, "organ").clip(min=0).cpu().numpy())
        
        obs_preds = np.concatenate(obs_preds).reshape(-1)
        organ_preds = np.concatenate(organ_preds).reshape(-1, organ_pred.shape[-1])
        
        if mode == "valid":
            data = pd.read_csv(self.valid_dataset._data_path)
        else:
            data = pd.read_csv(self.train_dataset._data_path)
        print(data.shape, obs_pred.shape)
        data['predict_ven_p'] = obs_preds
        for i, key in enumerate(organ_columns):
            data[f'predict_{key}'] = organ_preds[:, i]
        
        data.to_csv(f"{self.logger._result_dir}/mode={mode}_epoch={epoch}.csv")      
            
    @torch.no_grad()
    def evaluate(self, batch_size, plot_interval, epoch, rollout_mode="all"):
        results = {}
        eval_step = 0
        self.world_model._torch_train(False) 
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        
        for observations, actions, populations, compounds, organs, timesteps in tqdm(valid_loader):
            metrics = self.world_model.eval_batch(observations, actions, populations, compounds, organs, timesteps)
            for k, v in metrics.items():
                if k not in results.keys():
                    results[k] = v
                else:
                    results[k] += v
            eval_step += 1
        
        for k, v in results.items():
            results[k] = v / eval_step
        self.world_model._torch_train(True)
        
        if epoch % plot_interval == 0:
            obs_preds, organ_preds = self.world_model.rollout_batch(observations, actions, populations, compounds, organs, timesteps, rollout_mode=rollout_mode)
            observations = self.train_dataset.get_normalizer().unnormalize(observations, "observation").clip(min=0).cpu().numpy()
            obs_preds = self.train_dataset.get_normalizer().unnormalize(obs_preds, "observation").clip(min=0).cpu().numpy()
            
            organs = self.train_dataset.get_normalizer().unnormalize(organs, "organ").clip(min=0).cpu().numpy()
            organ_preds = self.train_dataset.get_normalizer().unnormalize(organ_preds, "organ").clip(min=0).cpu().numpy()
            
            self.plot(observations, obs_preds, "ven_p", epoch)
            
            for i, k in enumerate(organ_columns):
                self.plot(organs[..., i], organ_preds[..., i], k, epoch)
            
        return results
    
    @torch.no_grad()
    def plot(self, targets, preds, name, epoch):
        figure_dir = os.path.join(self.logger._plot_dir, name, "epoch=%d"%epoch)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        
        for i in range(10):
            target = targets[i]
            pred = np.concatenate([targets[i, :1], preds[i]])
            figure_path = os.path.join(figure_dir, "plot_%d.png"%i)
            plt.plot(list(range(target.shape[0])), target, label='real')
            plt.plot(list(range(pred.shape[0])), pred, label='generated')
            plt.legend()
            plt.savefig(f'{figure_path}')
            print(f'figure saved at {figure_path}')
            plt.close()
    
    def save(self, epoch):
        self.world_model.save_model(epoch, self.logger._model_dir)
        self.train_dataset.save_normalizer(os.path.join(self.logger._model_dir,"normalizer.pkl"))
