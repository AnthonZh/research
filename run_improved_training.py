import copy
import torch 
from torch.utils.data import DataLoader 
from architectures.UNET.unet import UNET    
import yaml

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.linalg import matrix_norm
import os
import math 
import numpy as np
import random
import matplotlib.pyplot as plt
from architectures.openai.unet import UNetModel 
from utils.common_functions import create_output_folders, save_grid, save_log, save_state_dict
from utils.datasetloader import LDCTLoader
from datetime import datetime 
from torch import Tensor
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

DEEP_MODEL='deep'
OPENAI_MODEL='oai'

def ddp_setup():  
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"]))  

class Trainer:
    def __init__(
        self,
        model_name, 
        model: torch.nn.Module,
        train_data: DataLoader,
        dataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        total_training_steps: int,
        world_size,
        rho: float = 7.0,    
        final_timesteps: int = 1280,
        initial_timesteps:int = 10,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        eta_min = 1e-5,
        upper_mult = 3.55,
        find_unused_parameters = True,
        anatomical_threshold: float = 24.,
        anatomical_weight: float = .95,
        background_weight: float = 0.05
    ) -> None:
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.upper_mult = upper_mult
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.dataset = dataset
        self.optimizer = optimizer 
        self.final_timesteps = final_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho   
        self.sigma_data = sigma_data
        self.total_training_steps = total_training_steps
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=find_unused_parameters)
        self.epochs = 0  
        self.world_size = world_size 
        self.sample_shape = (1, 512, 512)
        self.current_training_step = self.gpu_id  
        self.initial_timesteps = initial_timesteps
        self.training_steps_completed = False
        self.eta_min = eta_min
        self.anatomical_threshold = anatomical_threshold
        self.anatomical_weight = anatomical_weight
        self.background_weight = background_weight

    def _run_batch(self, x, condition):
        self.optimizer.zero_grad()
 
        num_timesteps = self.improved_timesteps_schedule(current_training_step=self.current_training_step)  

        boundaries = self.karras_boundaries(num_timesteps).to(device=self.gpu_id)  
        #max_str= 'Huber Loss: {:.4f}.format
        #print(f'max val: {torch.amax(boundaries)}')
        current_timesteps =  self.lognormal_timestep_distribution(x.shape[0], boundaries)
               
        current_sigmas = boundaries[current_timesteps].to(device=self.gpu_id)
        #print('current_sigmas: '+ str(current_sigmas))
        #print('timesteps: '+ str(timesteps))
        next_sigmas = boundaries[current_timesteps + 1].to(device=self.gpu_id)
        #print('next_sigmas: '+ str(next_sigmas))
        #print('timesteps+1 : '+ str(timesteps+1)) 
        current_noisy_data,next_noisy_data = self.add_noise(current_sigmas=current_sigmas,next_sigmas=next_sigmas,x=x)
        loss = self.loss_fun_improved(current_noisy_data=current_noisy_data,current_sigmas=current_sigmas, next_noisy_data=next_noisy_data,next_sigmas=next_sigmas,sigmas=boundaries,timesteps=current_timesteps, num_timesteps=num_timesteps, condition=condition)  
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), num_timesteps,current_timesteps + 1, next_sigmas, current_timesteps, current_sigmas
    
    def _run_epoch(self, epoch): 
        #b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        #pbar = tqdm(self.train_data)
        data_len= len(self.train_data)
        batch_step = 0 
        loss_list=[]
        for x, y in self.train_data: 
            if not self.model.training:
                self.model.train()  

            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            
            loss, num_timesteps, next_timesteps, next_sigmas, current_timesteps, current_sigmas= self._run_batch(x, y) 
            
            batch_step+=1
            now = datetime.now() 
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3] 
            result=  'Huber Loss: {:.4f}\tTraining Step: {:7d}/{:7d}\tNumber of Time Steps: {:7d}\tMin noise index: {:5d}\tMin sigma: {:5f}\tMax noise index: {:5d}\tMax sigma: {:5f}\tBatch Step: {:4d}/{:4d}\tEpoch: {:5d}/{:5d}\tGpu ID: {:3d}\tTime: {}'.format(
                    loss,
                    int(self.current_training_step),
                    int(self.total_training_steps),
                    int(num_timesteps),
                    torch.amin(current_timesteps ).item() ,
                    torch.amin(current_sigmas ).item(),
                    torch.amax(next_timesteps ).item() ,
                    torch.amax(next_sigmas ).item(), 
                    int(batch_step),
                    int(data_len),
                    int(epoch),
                    int(self.epochs),
                    int(self.gpu_id),
                    dt_string
                )
            print(result)

            loss_list.append(loss)

            if self.current_training_step == self.total_training_steps:
                self.training_steps_completed=True 
                break
            
            self.current_training_step = min(self.current_training_step + self.world_size, self.total_training_steps)
            save_log(model_name=self.model_name,record=result) 

        return np.mean(loss_list)

    def save_checkpoint(self,epoch):
        model_state_dict = copy.deepcopy(self.model.module.state_dict())
        
        save_state_dict(state_dict=model_state_dict,epoch=epoch,model_name=self.model_name) 
     
    def train(self):
        if self.gpu_id == 0 :
            
            create_output_folders(self.model_name)

        print('Started.')

        print('Model Name: '+ self.model_name)  
        print('final_timesteps: '+ str(self.final_timesteps))
        print('rho: '+ str(self.rho))
        print('train_data len: '+ str(len(self.train_data)))
        self.epochs= math.ceil(self.total_training_steps / (len(self.train_data)*world_size))
 
        self.show_denoised_image('Pre-Training')

        #self.scheduler = CosineAnnealingLR(self.optimizer,   T_max = self.epochs,     eta_min = self.eta_min)  
        for epoch in range(self.epochs):
            if self.training_steps_completed==False:
                avg_loss= self._run_epoch(epoch)
                #self.scheduler.step()
                now = datetime.now() 
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
                epoch_result=  'Average Loss: {:.4f}\tEpoch: {:5d}/{:5d}\tLR: {:5f}\tGpu ID: {:3d}\tTime: {}'.format(
                        avg_loss, 
                        epoch,
                        self.epochs,
                        self.optimizer.param_groups[0]["lr"],
                        self.gpu_id,
                        dt_string
                    )
                epoch_result= "*"*10 +'\t'+ epoch_result + '\t' +"*"*10 
                print(epoch_result)
                save_log(model_name=self.model_name,record=epoch_result)
                if self.gpu_id == 0: 
                    with torch.no_grad():
                        #self.sample_and_save(current_training_step=self.current_training_step,sample_steps=[2,5]) 
                        if epoch%5==0:
                            self.save_checkpoint(epoch)
                            self.show_denoised_image(epoch)
                
            else:
                if self.gpu_id == 0 : 
                    with torch.no_grad():
                        #self.sample_and_save(current_training_step=self.current_training_step,sample_steps=[2,5]) 
                        self.save_checkpoint(epoch)
                        self.show_denoised_image(epoch)
                break

    def sample(self, model, x, ts): 
        sigma = ts[0:1]
        sigma = torch.full((x.shape[0],), sigma[0], dtype=x.dtype, device=self.gpu_id)
        x = x.to(device=self.gpu_id)
        #sigma = torch.squeeze(sigma,dim=-1) 
        x = self.model_forward_wrapper(model, x, sigma)

        for sigma in ts[1:]:
            z = torch.randn_like(x).to(device=self.gpu_id)
            x = x + math.sqrt(sigma**2 - self.sigma_min**2) * z
            x = self.model_forward_wrapper(model, x, torch.tensor([sigma])) 

        return x
    
    def get_sigmas_linear_reverse(self,n,sigma_min= 0.002,sigma_max=79.999985): 
        sigmas = torch.linspace(sigma_max, sigma_min, n, dtype=torch.float16 ).to(device=self.gpu_id)
        return sigmas

    def sample_and_save(self,current_training_step,sample_steps): 
        self.model.eval()  
        for sample_step in sample_steps:
            
            sigmas = self.get_sigmas_linear_reverse(sample_step,self.sigma_min,self.sigma_max) 
            sample_results = self.sample( model=self.model,ts= sigmas )
            sample_results = (sample_results * 0.5 + 0.5).clamp(0, 1) 
            
            save_grid(tensor=sample_results,epoch=int(current_training_step),model_name=self.model_name,sample_step=sample_step)
        self.model.train() 
    
    def show_denoised_image(self, epoch):
        self.model.eval()

        ldct, fdct = self.dataset[50]
        ldct, fdct = ldct.unsqueeze(0), fdct.unsqueeze(0)

        fdct = torch.squeeze(fdct)
        fdct = fdct.detach().cpu().numpy()
        ldct_pre_denoised = torch.squeeze(ldct)
        ldct_pre_denoised = ldct_pre_denoised.detach().cpu().numpy()

        #noisy_SSIM = structural_similarity(fdct, ldct_pre_denoised, data_range=np.max( [math.ceil(ldct_pre_denoised.max() - ldct_pre_denoised.min()), math.ceil(fdct.max() - fdct.min())] ))
        #noisy_PSNR = peak_signal_noise_ratio(fdct, ldct_pre_denoised, data_range=np.max( [math.ceil(ldct_pre_denoised.max() - ldct_pre_denoised.min()), math.ceil(fdct.max() - fdct.min())] ))

        ldct.to(device=self.gpu_id)
        ldct = self.multiple_step_denoising(1, ldct)

        ldct = torch.squeeze(ldct)
        ldct = ldct.detach().cpu().numpy()

        #denoised_SSIM = structural_similarity(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))
        #denoised_PSNR = peak_signal_noise_ratio(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))

        #print(f'NOISY -- SSIM: {noisy_SSIM}, PSNR: {noisy_PSNR}\nDENOISED -- SSIM: {denoised_SSIM}, PSNR: {denoised_PSNR}')
        
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(ldct_pre_denoised, cmap='gray')
        ax[0].set_title('LDCT Pre Denoising')
        ax[1].imshow(ldct, cmap='gray')
        ax[1].set_title('LDCT Post Denoising')
        ax[2].imshow(fdct, cmap='gray')
        ax[2].set_title('FDCT')

        f.tight_layout()
        
        f.savefig(f'training_samples/epoch_{epoch}.png', dpi=1200)
    
    def multiple_step_denoising(self, sample_steps, x): 
        sigmas = self.get_sigmas_linear_reverse(sample_steps, self.sigma_min, self.sigma_max) 
        #print(sigmas)   
        sample_results = self.sample(model=self.model, x=x, ts=sigmas)
        return sample_results

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
    
    def output_scaling(self, sigma):
        return (self.sigma_data * (sigma - self.sigma_min)) / (self.sigma_data**2 + sigma**2) ** 0.5
    
    def model_forward_wrapper(self, model, x, sigma):
        c_skip = self.skip_scaling(sigma )
        c_out = self.output_scaling(sigma) 
            
        c_skip = self.pad_dims_like(c_skip, x).to(device=self.gpu_id)
        c_out = self.pad_dims_like(c_out, x).to(device=self.gpu_id)
        return c_skip  * x + c_out * model(x, sigma)

    def loss_fun_improved(self, current_noisy_data, next_noisy_data, condition, current_sigmas, next_sigmas, sigmas, timesteps, num_timesteps):         
        next_x= self.model_forward_wrapper(self.model,next_noisy_data,next_sigmas)
        with torch.no_grad():
            current_x = self.model_forward_wrapper(self.model,current_noisy_data,current_sigmas)
        
        loss_weights = self.pad_dims_like(self.improved_loss_weighting(sigmas)[timesteps], next_x) 

        #ph_loss= self.pseudo_huber_anatomical(current_x, next_x, condition)
        ph_loss = self.psuedo_huber_condition(current_x, next_x, condition)

        #loss = self.pseudo_huber_norms(current_x, next_x, condition)
        loss = (loss_weights * ph_loss).mean()

        return loss

    def improved_loss_weighting(self,sigmas):
        return 1 / (sigmas[1:] - sigmas[:-1])
 
    def pseudo_huber_loss(self,input, target): 
        c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
        return torch.sqrt((input - target) ** 2 + c**2) - c 
    
    def pseudo_huber_anatomical(self, a, b, y):
        a_anatomical = torch.where(a > self.anatomical_threshold, a, 0).to(device=self.gpu_id)
        a_background = torch.where(a <= self.anatomical_threshold, a, 0).to(device=self.gpu_id)

        b_anatomical = torch.where(b > self.anatomical_threshold, b, 0).to(device=self.gpu_id)
        b_background = torch.where(b <= self.anatomical_threshold, b, 0).to(device=self.gpu_id)

        y_anatomical = torch.where(y > self.anatomical_threshold, y, 0).to(device=self.gpu_id)
        y_background = torch.where(y <= self.anatomical_threshold, y, 0).to(device=self.gpu_id)

        a_b_anatomical = self.pseudo_huber_loss(a_anatomical, b_anatomical)
        a_y_anatomical = self.pseudo_huber_loss(a_anatomical, y_anatomical)
        b_y_anatomical = self.pseudo_huber_loss(b_anatomical, y_anatomical)

        a_b_background = self.pseudo_huber_loss(a_background, b_background)
        a_y_background = self.pseudo_huber_loss(a_background, y_background)
        b_y_background = self.pseudo_huber_loss(b_background, y_background)

        return self.anatomical_weight * ( (a_b_anatomical * 2/3) + ((a_y_anatomical + b_y_anatomical) * 1/3) ) + self.background_weight * ( (a_b_background * 2/3) + ((a_y_background + b_y_background) * 1/3) )
    
    def psuedo_huber_condition(self, a, b, y):
        a_b = self.pseudo_huber_loss(a, b)
        a_y = self.pseudo_huber_loss(a, y)
        b_y = self.pseudo_huber_loss(b, y)

        return a_b * 2/3 + (a_y + b_y) * 1/3

    def ph_function(self, x, y):
        assert x.dim() == y.dim(), "X and Y don't have the same amount of dimensions"
        c = .00054 * math.sqrt( x.dim() - 1 )
        return torch.sqrt( (matrix_norm(x-y))**2 + c**2 ) - c 

    def pad_dims_like(self, x, other):
        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))
    
    def add_noise(self,x,current_sigmas,next_sigmas):
        #noise = torch.randn_like(x).to(device=self.gpu_id)  

        current_noisy_data = torch.where(x > self.anatomical_threshold, x + current_sigmas * torch.randn(1).item(), self.anatomical_threshold).to(device=self.gpu_id)
        next_noisy_data = torch.where(x > self.anatomical_threshold, x + next_sigmas * torch.randn(1).item(), self.anatomical_threshold).to(device=self.gpu_id)

        #current_noisy_data = x + self.pad_dims_like(current_sigmas,x) * noise
        #next_noisy_data = x + self.pad_dims_like(next_sigmas,x)  * noise 

        #current_noisy_data = x + current_sigmas[:,:,None,None] * noise
        #next_noisy_data = x + next_sigmas[:,:,None,None]  * noise 

        return current_noisy_data,next_noisy_data
    
    def improved_timesteps_schedule(self, current_training_step: int) -> int: 
        total_training_steps_prime = math.floor(
            self.total_training_steps
            / (math.log2(math.floor(self.final_timesteps / self.initial_timesteps)) + 1)
        )
        num_timesteps = self.initial_timesteps * math.pow(
            2, math.floor(current_training_step / total_training_steps_prime)
        )
        num_timesteps = min(num_timesteps, self.final_timesteps) + 1

        return num_timesteps

    def improved_timesteps_schedule(self, current_training_step: int) -> int: 
        total_training_steps_prime = math.floor(
            self.total_training_steps
            / (math.log2(math.floor(self.final_timesteps / self.initial_timesteps)) + 1)
        )
        num_timesteps = self.initial_timesteps * math.pow(
            2, math.floor(current_training_step / total_training_steps_prime)
        )
        num_timesteps = min(num_timesteps, self.final_timesteps) + 1

        return num_timesteps

    def karras_boundaries(self, num_timesteps):
        # This will be used to generate the boundaries for the time discretization
        rho_inv = 1.0 / self.rho
        # Clamp steps to 1 so that we don't get nans
        steps = torch.arange(num_timesteps, device=self.gpu_id) / max(num_timesteps - 1, 1)
        sigmas = self.sigma_min**rho_inv + steps * (
            self.sigma_max**rho_inv - self.sigma_min**rho_inv
        )
        sigmas = sigmas**self.rho

        return sigmas
    
    def rayleigh_distribution(self, N, dim, scale=2):
        values = np.random.rayleigh(scale, 10000)  
        choices= random.choices(values, k=dim)
        
        #print(len(weights))
        #print(len(values))
        upper_bound= scale * self.upper_mult
        choices= np.clip(a=choices,a_min=0,a_max=upper_bound)
        
        #choices_scaled = (choices - np.amin(choices)) / (np.amax(choices) - np.amin(choices))  
        choices_scaled = choices*( (N-1)/ upper_bound )
        choices_scaled= np.ceil(choices_scaled*10) /10
        #choices_scaled *= N-1
        return choices_scaled.astype(int)

    def lognormal_timestep_distribution(self, num_samples: int, sigmas: Tensor, mean: float = -1.1, std: float = 2.0) -> Tensor: 
        pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
            (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
        )
        pdf = pdf / pdf.sum()

        timesteps = torch.multinomial(pdf, num_samples, replacement=True)

        return timesteps

def main(world_size):
    with open("parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    #print(parameters)
    ddp_setup() 
    batch_size= parameters['batch_size'] // world_size
    loader = LDCTLoader(root=parameters['root'], batch_size=batch_size)
    train_data = loader.dataloader 
    dataset = loader.dataset
    gpu_id = int(os.environ["RANK"])
    if parameters['model_type'] == DEEP_MODEL: 
        model = UNET( img_channels=parameters['img_channels'],  device=gpu_id,groupnorm=parameters['groupnorm'], attention_resolution=parameters['attention_resolutions'], 
        num_heads=parameters['num_heads'], dropout=parameters['dropout'],base_channels=parameters['base_channels'],
        num_res_blocks=parameters['num_res_blocks'],  use_flash_attention=parameters['use_flash_attention'], emb_time_multiplier=parameters['emb_time_multiplier'],
                    num_head_channels=parameters['num_head_channels'], use_new_attention_order=parameters['use_new_attention_order'],
                    use_scale_shift_norm=parameters['use_scale_shift_norm'],use_conv=parameters['use_conv'], use_conv_up_down =parameters['use_conv_up_down']).to(device=gpu_id)

    else:
        model= UNetModel(attention_resolutions=parameters['attention_resolutions'], use_scale_shift_norm=parameters['use_scale_shift_norm'],
                         model_channels=parameters['base_channels'],num_head_channels=parameters['num_head_channels'],dropout=parameters['dropout'],
                         num_res_blocks=parameters['num_res_blocks'],resblock_updown=True,image_size=parameters['image_size'],in_channels=parameters['img_dimension'],out_channels=parameters['img_dimension'])
    
    optimizer= torch.optim.AdamW(model.parameters(), lr=parameters['lr']) 
    
    trainer = Trainer(model_name=parameters['model_name'],model=model, train_data=train_data, dataset=dataset, optimizer=optimizer, gpu_id=gpu_id,rho = parameters['rho'],  
        find_unused_parameters=parameters['use_flash_attention'],final_timesteps  = parameters['final_timesteps'], 
        initial_timesteps=parameters['initial_timesteps'], total_training_steps=parameters['total_training_steps'], world_size=world_size)
    trainer.train()
    destroy_process_group()

if __name__ == "__main__": 
    
    world_size =int(os.environ['WORLD_SIZE'])
    local_world_size =int(os.environ['LOCAL_WORLD_SIZE'])
    print('local_world_size :'+ str(local_world_size))
    print('world_size :'+ str(world_size))
    print('RANK  :'+ str(int(os.environ["RANK"]))) 

    #mp.spawn(main, args=(world_size, args.model_name, args.batch_size, args.epochs), nprocs=world_size)
    main(world_size=world_size)
