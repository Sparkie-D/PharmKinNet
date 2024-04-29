import numpy as np
import torch 
import torch.nn as nn
import os 
from tqdm import tqdm
from typing import Iterable

from models.rnn import RNNModel
from models.dense import DenseModel
from models.utils import get_parameters


def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class MLPWorldModel():
    def __init__(self, args):
        args.embed_size = args.modelstate_size * 4
        
        # self.DynamicModel = RNNModel(args.embed_size, args.modelstate_size, **args.dynamic_model).to(args.device)
        self.DynamicModel = DenseModel(args.modelstate_size, args.embed_size, **args.dynamic_model).to(args.device)
        
        
        self.ObsEncoder = DenseModel(args.modelstate_size, args.observation_dim, **args.obs_encoder).to(args.device)
        self.ObsDecoder = DenseModel(args.observation_dim, args.modelstate_size, **args.obs_decoder).to(args.device)
        self.ActionEncoder = DenseModel(args.modelstate_size, args.action_dim, **args.action_encoder).to(args.device)
        self.CompoundEncoder = DenseModel(args.modelstate_size, args.compound_dim, **args.compound_encoder).to(args.device)
        self.PopulationEncoder = DenseModel(args.modelstate_size, args.population_dim, **args.population_encoder).to(args.device)
        
        self.OrganEncoder = DenseModel(args.modelstate_size, args.organ_dim, **args.organ_encoder).to(args.device)
        self.OrganDecoder = DenseModel(args.organ_dim, args.modelstate_size, **args.organ_decoder).to(args.device)
        # self.OrganDynamicModel = RNNModel(args.embed_size, args.modelstate_size, **args.organ_dynamic_model).to(args.device)
        self.OrganDynamicModel = DenseModel(args.modelstate_size, args.embed_size, **args.dynamic_model).to(args.device)
        
        self.world_list = [self.DynamicModel, self.ObsEncoder, self.ObsDecoder, self.ActionEncoder, self.CompoundEncoder, self.PopulationEncoder, self.OrganEncoder, self.OrganDecoder, self.OrganDynamicModel]
        self.world_name = ['DynamicModel', 'ObsEncoder', 'ObsDecoder', 'ActionEncoder', 'CompoundEncoder', 'PopulationEncoder', 'OrganEncoder', 'OrganDecoder', 'OrganDynamicModel']

        self.device = args.device
        self._loss_scale = args.loss_scale
        self._grad_clip_norm = args.grad_clip_norm
        
        self._loss_scale = args.loss_scale
        self._update_organ = args.update_organ
        self._update_obs = args.update_obs
        
        self.update_list = []
        if args.update_list is not None:
            for k in args.update_list:
                self.update_list.append(self.__dict__[k])
        else:
            if self._update_obs:
                self.update_list.extend([self.DynamicModel, self.ObsEncoder, self.ObsDecoder, self.ActionEncoder, self.CompoundEncoder, self.PopulationEncoder])
            if self._update_organ:
                self.update_list.extend([self.OrganEncoder, self.OrganDecoder, self.OrganDynamicModel])
        self.model_optimizer = torch.optim.Adam(get_parameters(self.update_list), lr=args.model_lr)

        self.model_summary()

    def _torch_train(self, mode=True):
        for model in self.world_list:
            model.train(mode)
    
    def model_summary(self):
        total_param = 0
        print('-' * 50)
        for name, model in zip(self.world_name, self.world_list):
            param_num = sum(p.numel() for p in model.parameters())
            total_param += param_num  # Update total_param by adding param_num
            print("%s Params: %.2f M" % (name, param_num / (1024 * 1024)))
            print("Estimated %s Size: %.2f MB" % (name, param_num * 4 / (1024 * 1024)))
            print('-' * 50)

        model_size = total_param * 4 / (1024 * 1024)
        print("Total Params: %.2f M" % (total_param / (1024 * 1024)))
        print("Estimated Total Size: %.2f MB" % model_size)
        print('-' * 50)
        print('-' * 50)
    
    def get_auc(self, obs):
        auc = (obs[:-1] + obs[:1]).mean() / 2
        return auc
    
    def get_tmax(self, obs):
        tmax = torch.argmax(obs, dim=1)
        return tmax
        
    def train_batch(self, observations, actions, populations, compounds, organs, timesteps, prob=0.):
        """ 
        trains the world model
        """
        assert self._update_obs or self._update_organ
        
        train_metrics = {}
        
        obs_embeds = self.ObsEncoder(observations)
        action_embeds = self.ActionEncoder(actions)
        population_embeds = self.PopulationEncoder(populations)
        compound_embeds = self.CompoundEncoder(compounds)
        organ_embeds = self.OrganEncoder(organs)

        z_preds, o_preds = [], []
        for i in range(1, observations.shape[1]):
            stacked_inputs = torch.cat([obs_embeds[:, i-1: i] if ((np.random.random() > prob) or (i==1)) else z_state, 
                                        action_embeds[:, i-1: i], 
                                        population_embeds[:, i-1: i], 
                                        compound_embeds[:, i-1: i]], -1)
            z_state = self.DynamicModel(stacked_inputs)
            z_preds.append(z_state)
            
            organ_stacked_inputs = torch.cat([organ_embeds[:, i-1: i] if ((np.random.random() > prob) or (i==1)) else o_state, 
                                              z_state, 
                                              population_embeds[:, i-1: i], 
                                              compound_embeds[:, i-1: i]], -1)
            o_state = self.OrganDynamicModel(organ_stacked_inputs)
            o_preds.append(o_state)
            

        z_preds = torch.cat(z_preds, 1)
        obs_preds = self.ObsDecoder(z_preds)
        
        o_preds = torch.cat(o_preds, 1)
        organ_preds = self.OrganDecoder(o_preds)

        # ven loss
        reconstruction_loss = self._obs_loss(obs_preds, observations[:, 1:]) * self._loss_scale["reconstruction"]
        consistency_loss = self._obs_loss(z_preds, obs_embeds[:, 1:]) * self._loss_scale["consistency"] 
        obs_loss = reconstruction_loss + consistency_loss

        # organ loss
        organ_reconstruction_loss = self._obs_loss(organ_preds, organs[:, 1:]) * self._loss_scale["reconstruction"]
        organ_consistency_loss = self._obs_loss(o_preds, organ_embeds[:, 1:]) * self._loss_scale["consistency"]
        organ_loss = organ_reconstruction_loss + organ_consistency_loss

        # auc loss
        auc = self.get_auc(observations)
        auc_pred = self.get_auc(obs_preds)
        auc_loss = ((auc_pred - auc) ** 2).mean() * self._loss_scale["auc"]

        # cmax loss
        tmax = self.get_tmax(observations)
        cmax = torch.gather(observations.squeeze(), dim=1, index=tmax)
        tmax_pred = self.get_tmax(obs_preds)
        cmax_pred = torch.gather(obs_preds.squeeze(), dim=1, index=tmax_pred)
        cmax_loss = self._obs_loss(cmax_pred, cmax) * self._loss_scale["cmax"]
        
        model_loss = 0
        if self._update_obs:
            model_loss += obs_loss
        if self._update_organ:
            model_loss += organ_loss
        model_loss += auc_loss
        model_loss += cmax_loss
        
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self._grad_clip_norm)
        self.model_optimizer.step()

        train_metrics['train/model_loss'] = model_loss.item()
        train_metrics['train/obs_loss'] = obs_loss.item()
        train_metrics['train/reconstruction_loss'] = reconstruction_loss.item()
        train_metrics['train/consistency_loss'] = consistency_loss.item()
        train_metrics['train/organ_loss'] = organ_loss.item()
        train_metrics['train/organ_reconstruction_loss'] = organ_reconstruction_loss.item()
        train_metrics['train/organ_consistency_loss'] = organ_consistency_loss.item()
        train_metrics['train/auc_loss'] = auc_loss.item()
        train_metrics['train/cmax_loss'] = cmax_loss.item()

        return train_metrics

    def _obs_loss(self, obs_pred, obs):
        obs_loss = ((obs_pred - obs) ** 2).mean()
        return obs_loss

    def train(self):
        for model in self.world_list:
            model.train()
    
    def eval(self):
        for model in self.world_list:
            model.eval()

    @torch.no_grad()
    def eval_batch(self, observations, actions, populations, compounds, organs, timesteps):
        obs_preds, organ_preds = self.rollout_batch(observations, actions, populations, compounds, organs, timesteps)
        rollout_loss = self._obs_loss(obs_preds, observations[:, 1:])
        organ_rollout_loss = self._obs_loss(organ_preds, organs[:, 1:])

        metrics = {
            "rollout_loss": rollout_loss.item(),
            "organ_rollout_loss": organ_rollout_loss.item(),
        }
        
        return metrics
        
    @torch.no_grad()
    def rollout_batch(self, observations, actions, populations, compounds, organs, timesteps, rollout_mode="all"):
        """ 
        trains the world model
        """
        self.eval()
        
        obs_embeds = self.ObsEncoder(observations)
        action_embeds = self.ActionEncoder(actions)
        population_embeds = self.PopulationEncoder(populations)
        compound_embeds = self.CompoundEncoder(compounds)
        organ_embeds = self.OrganEncoder(organs)

        z_preds, o_preds = [], []
        for i in range(1, observations.shape[1]):
            stacked_inputs = torch.cat([obs_embeds[:, i-1: i] if i == 1 else z_state, 
                                        action_embeds[:, i-1: i], 
                                        population_embeds[:, i-1: i], 
                                        compound_embeds[:, i-1: i]], -1)
            z_state = self.DynamicModel(stacked_inputs)
            z_preds.append(z_state)
            
            if rollout_mode == "organ":
                z_state = obs_embeds[:, i: i+1]
            organ_stacked_inputs = torch.cat([organ_embeds[:, i-1: i] if i == 1 else o_state, 
                                              z_state, 
                                              population_embeds[:, i-1: i], 
                                              compound_embeds[:, i-1: i]], -1)
            o_state = self.OrganDynamicModel(organ_stacked_inputs)
            o_preds.append(o_state)

        z_preds = torch.cat(z_preds, 1)
        obs_preds = self.ObsDecoder(z_preds)
        
        o_preds = torch.cat(o_preds, 1)
        organ_preds = self.OrganDecoder(o_preds)
    
        self.train()

        return obs_preds, organ_preds
        
    def save_model(self, iter, save_dir):
        save_dict = {
            'DynamicModel': self.DynamicModel.state_dict(),
            'ObsEncoder': self.ObsEncoder.state_dict(),
            'ObsDecoder': self.ObsDecoder.state_dict(),
            'ActionEncoder': self.ActionEncoder.state_dict(),
            'CompoundEncoder': self.CompoundEncoder.state_dict(),
            'PopulationEncoder': self.PopulationEncoder.state_dict(),
            'OrganEncoder': self.OrganEncoder.state_dict(),
            'OrganDecoder': self.OrganDecoder.state_dict(),
            'OrganDynamicModel': self.OrganDynamicModel.state_dict(),
        }
        save_path = os.path.join(save_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def load_model(self, load_path):
        save_dict = torch.load(load_path, map_location=self.device)
        for k in self.world_name:
            if k in save_dict.keys():
                try:
                    self.__dict__[k].load_state_dict(save_dict[k])
                except:
                    continue
        print('load model from %s' % load_path)