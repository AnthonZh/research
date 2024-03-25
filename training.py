import copy
import yaml
import os
import math 
import numpy as np
import random
from datetime import datetime

from architectures.openai.unet import UNetModel 
from utils.common_functions import create_output_folders, save_grid, save_log, save_state_dict

import torch 
from torch.utils.data import   DataLoader 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.linalg import matrix_norm
from architectures.UNET.unet import UNET    
from torch import Tensor

DEEP_MODEL='deep'
OPENAI_MODEL='oai' 

class Trainer:
    def __init__(
        self,
        model_name, 
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        total_training_steps:int,
        world_size,
        rho: float = 7.0,    
        final_timesteps: int = 1280,
        initial_timesteps:int= 10,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        eta_min= 1e-5,
        upper_mult=3.55,
        find_unused_parameters=True
        ) -> None:

        self.model_name=model_name
        self.gpu_id = gpu_id
        self.upper_mult=upper_mult
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer 
        self.final_timesteps=final_timesteps
        self.sigma_min=sigma_min
        self.sigma_max=sigma_max
        self.rho=rho   
        self.sigma_data=sigma_data
        self.total_training_steps=total_training_steps
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=find_unused_parameters)
        self.epochs= 0  
        self.world_size=world_size 
        self.sample_shape=(128,3,32,32)
        self.current_training_step= self.gpu_id  
        self.initial_timesteps=initial_timesteps
        self.training_steps_completed=False
        self.eta_min= eta_min:
    def loss(self, current_noisy_data, next_noisy_data, target, current_sigmas, next_sigmas, sigmas, timesteps, num_timesteps):
        next_x = self.model_forward_wrapper(self.model, next_noisy_data, next_sigmas)
        with torch.no_grad():
            current_x = self.model_forward_wrapper(self.model, current_noisy_data, current_sigmas)
        
        ph_loss = self.pseudo_huber_norms(current_x, next_x, target)
        return ph_loss

    def pseudo_huber_one(self, a, b, condition):
        c = .00054 * math.sqrt( math.prod(a.shape[1:]) )
        a_b = torch.sqrt( (a-b)**2 + c**2 ) - c
        a_condition = torch.sqrt( (a-condition)**2 + c**2 ) - c
        b_condition = torch.sqrt( (b-condition)**2 + c**2 ) - c

        return a_b + a_condition + b_condition

    def pseudo_huber_norms(self, a, b, y):
        c = .00054 * math.sqrt( a.dim() - 1 )
        loss_ab = torch.sqrt( (matrix_norm(a-b))**2 + c**2 ) - c
        loss_ay = torch.sqrt( (matrix_norm(a-y))**2 + c**2 ) - c
        loss_by = torch.sqrt( (matrix_norm(b-y))**2 + c**2 ) - c

        return a + b + y

    def model_forward_wrapper(self, model, x, sigma):
        c_skip = self.skip_scaling(sigma )
        c_out = self.output_scaling(sigma) 
        
        c_skip = self.pad_dims_like(c_skip, x)
        c_out = self.pad_dims_like(c_out, x)
        return c_skip  * x + c_out * model(x, sigma)

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)

    def output_scaling(self, sigma):
        return (self.sigma_data * (sigma - self.sigma_min)) / (self.sigma_data**2 + sigma**2) ** 0.5

    def pad_dims_like(self,x, other):
        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))
