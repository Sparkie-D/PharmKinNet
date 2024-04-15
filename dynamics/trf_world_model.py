import numpy as np
import torch 
import torch.nn as nn
import os 
from tqdm import tqdm
from typing import Iterable

from models.trf import TRFModel
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
        # print(module)
        model_parameters += list(module.parameters())
    return model_parameters


class TRFWorldModel():
    def __init__(self, args):
        args.embed_size = args.modelstate_size * 4
        
        self.DynamicModel = TRFModel(args.embed_size, args.modelstate_size, **args.dynamic_model).to(args.device)
        
        self.ObsEncoder = DenseModel(args.modelstate_size, args.observation_dim, **args.obs_encoder).to(args.device)
        self.ObsDecoder = DenseModel(args.observation_dim, args.modelstate_size, **args.obs_decoder).to(args.device)
        self.ActionEncoder = DenseModel(args.modelstate_size, args.action_dim, **args.action_encoder).to(args.device)
        self.CompoundEncoder = DenseModel(args.modelstate_size, args.compound_dim, **args.compound_encoder).to(args.device)
        self.PopulationEncoder = DenseModel(args.modelstate_size, args.population_dim, **args.population_encoder).to(args.device)
        self.TimestepEmbedding = nn.Embedding(args.max_ep_len, args.modelstate_size).to(args.device)
        
        self.OrganEncoder = DenseModel(args.modelstate_size, args.organ_dim, **args.organ_encoder).to(args.device)
        self.OrganDecoder = DenseModel(args.organ_dim, args.modelstate_size, **args.organ_decoder).to(args.device)
        self.OrganDynamicModel = TRFModel(args.embed_size, args.modelstate_size, **args.organ_dynamic_model).to(args.device)
        
        self.world_list = [self.DynamicModel, self.ObsEncoder, self.ObsDecoder, self.ActionEncoder, self.CompoundEncoder, self.PopulationEncoder, self.TimestepEmbedding, self.OrganEncoder, self.OrganDecoder, self.OrganDynamicModel]
        self.world_name = ['DynamicModel', 'ObsEncoder', 'ObsDecoder', 'ActionEncoder', 'CompoundEncoder', 'PopulationEncoder', 'TimestepEmbedding', 'OrganEncoder', 'OrganDecoder', 'OrganDynamicModel']

        self.device = args.device
        self._seq_len = args.seq_len
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
                self.update_list.extend([self.DynamicModel, self.ObsEncoder, self.ObsDecoder, self.ActionEncoder, self.CompoundEncoder, self.PopulationEncoder, self.TimestepEmbedding])
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
           
    def train_batch(self, observations, actions, populations, compounds, organs, timesteps, prob=0.):
        """ 
        trains the world model
        """
        assert self._update_obs or self._update_organ
        
        train_metrics = {}
        timestep_embeds = self.TimestepEmbedding(timesteps)
        obs_embeds = self.ObsEncoder(observations)
        action_embeds = self.ActionEncoder(actions)
        population_embeds = self.PopulationEncoder(populations) 
        compound_embeds = self.CompoundEncoder(compounds) 
        organ_embeds = self.OrganEncoder(organs)
        
        obs_embeds, action_embeds, population_embeds, compound_embeds, organ_embeds, timestep_embeds, attention_mask = self.padding_all(
            obs_embeds, action_embeds, population_embeds, compound_embeds, organ_embeds, timestep_embeds
        )
        stacked_inputs = torch.cat([obs_embeds + timestep_embeds, 
                                    action_embeds + timestep_embeds, 
                                    population_embeds + timestep_embeds, 
                                    compound_embeds + timestep_embeds], -1)
        z_preds = self.DynamicModel(stacked_inputs, attention_mask=attention_mask)

        organ_stacked_inputs = torch.cat([organ_embeds + timestep_embeds, 
                                        z_preds + timestep_embeds, 
                                        population_embeds + timestep_embeds, 
                                        compound_embeds + timestep_embeds], -1)
        o_preds = self.OrganDynamicModel(organ_stacked_inputs, attention_mask=attention_mask)

        obs_preds = self.ObsDecoder(z_preds)
        organ_preds = self.OrganDecoder(o_preds)

        reconstruction_loss = self._obs_loss(obs_preds[:, :-1], observations[:, 1:])
        consistency_loss = self._obs_loss(z_preds[:, :-1], obs_embeds[:, 1:])
        obs_loss = self._loss_scale["reconstruction"] * reconstruction_loss + self._loss_scale["consistency"] * consistency_loss

        organ_reconstruction_loss = self._obs_loss(organ_preds[:, :-1], organs[:, 1:])
        organ_consistency_loss = self._obs_loss(o_preds[:, :-1], organ_embeds[:, 1:])
        organ_loss = self._loss_scale["reconstruction"] * organ_reconstruction_loss + self._loss_scale["consistency"] * organ_consistency_loss
    
        model_loss = 0
        if self._update_obs:
            model_loss += obs_loss
        if self._update_organ:
            model_loss += organ_loss
        
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

        return train_metrics

    def _obs_loss(self, obs_pred, obs):
        obs_loss = ((obs_pred - obs) ** 2).sum(dim=-1).mean()
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
        
    def padding(self, data):
        return torch.cat([data, torch.zeros((data.shape[0], self._seq_len - data.shape[1], data.shape[-1]), device=self.device)], dim=1).to(dtype=torch.float32)
    
    def padding_all(self, observations, actions, populations, compounds, organs, timesteps):
        max_length = self._seq_len
        batch_size, num_timesteps, _ = observations.shape
        attention_mask = torch.cat([torch.ones((batch_size, num_timesteps)), torch.zeros((batch_size, max_length-num_timesteps))], dim=1)
        attention_mask = attention_mask.to(dtype=torch.long, device=self.device)

        obs = self.padding(observations)
        act = self.padding(actions)
        pop = self.padding(populations)
        com = self.padding(compounds)
        org = self.padding(organs)
        tim = self.padding(timesteps)
    
        return obs, act, pop, com, org, tim, attention_mask
        
    @torch.no_grad()
    def rollout_batch(self, observations, actions, populations, compounds, organs, timesteps):
        self.eval()
        z_preds = self.ObsEncoder(observations[:, :1])
        o_preds = self.OrganEncoder(organs[:, :1])
        for i in range(1, observations.shape[1]):    
            timestep_embeds = self.TimestepEmbedding(timesteps[:, :i])
            obs_embeds = self.ObsEncoder(observations[:, :i])
            action_embeds = self.ActionEncoder(actions[:, :i]) 
            population_embeds = self.PopulationEncoder(populations[:, :i]) 
            compound_embeds = self.CompoundEncoder(compounds[:, :i])
            organ_embeds = self.OrganEncoder(organs[:, :i])
            
            obs_embeds, action_embeds, population_embeds, compound_embeds, organ_embeds, timestep_embeds, attention_mask = self.padding_all(
                obs_embeds, action_embeds, population_embeds, compound_embeds, organ_embeds, timestep_embeds
            )

            stacked_inputs = torch.cat([self.padding(z_preds) + timestep_embeds,
                                        action_embeds + timestep_embeds, 
                                        population_embeds + timestep_embeds, 
                                        compound_embeds + timestep_embeds], -1)
            z_state = self.DynamicModel(stacked_inputs, attention_mask)[:, i-1:i]
            z_preds = torch.cat([z_preds, z_state], 1) # autoregressive

            organ_stacked_inputs = torch.cat([self.padding(o_preds) + timestep_embeds,
                                              self.padding(z_preds) + timestep_embeds, 
                                              population_embeds + timestep_embeds, 
                                              compound_embeds + timestep_embeds], -1)
            o_state = self.OrganDynamicModel(organ_stacked_inputs, attention_mask)[:, i-1:i]
            o_preds = torch.cat([o_preds, o_state], 1)

        obs_preds = self.ObsDecoder(z_preds)[:, 1:]
        organ_preds = self.OrganDecoder(o_preds)[:, 1:]
        
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
            'TimestepEmbedding': self.TimestepEmbedding.state_dict(),
        }
        save_path = os.path.join(save_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def load_model(self, load_path):
        save_dict = torch.load(load_path, map_location=self.device)
        for k in self.world_name:
            if k in save_dict.keys():
                self.__dict__[k].load_state_dict(save_dict[k])
        print('load model from %s' % load_path)