import torch
import numpy as np
import yaml
import os
import math

from torch.distributed import init_process_group, destroy_process_group

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from utils.datasetloader import LDCTLoader

from architectures.UNET.unet import UNET
from architectures.openai.unet import UNetModel

import matplotlib.pyplot as plt

import gc

DEEP_MODEL='deep'
OPENAI_MODEL='oai'

class Tester:
    def __init__(self, model, test_root, gpu_id, denoising_steps: int = 2, sigma_min: float = 0.002, sigma_data: float = 0.5, sigma_max: float = 80.):
        self.model = model
        self.model.eval()

        self.gpu_id = gpu_id
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoising_steps = denoising_steps
        
        self.dataloader_wrapper = LDCTLoader(test_root, 1)
        self.dataset = self.dataloader_wrapper.dataset
        self.dataloader = self.dataloader_wrapper.dataloader

    def test_mult_step(self):
        self.model.eval()
        
        mean_ssim = 0
        mean_psnr = 0
        n = len(self.dataloader)
        i = 1

        print(f'Test data length: {n}')

        for ldct, fdct in self.dataloader:
            ldct.to(device=self.gpu_id)
            ldct = self.multiple_step_denoising(self.denoising_steps, ldct)
            
            ldct, fdct = torch.squeeze(ldct), torch.squeeze(fdct)
            ldct, fdct = ldct.detach().cpu().numpy(), fdct.detach().cpu().numpy()

            #gc.collect()
            #torch.cuda.empty_cache()

            mean_ssim += structural_similarity(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))
            mean_psnr += peak_signal_noise_ratio(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))

            print(f'Slice {i}/{n}:\nMean SSIM: {mean_ssim/i}, Mean PSNR: {mean_psnr/i}')
            
            i += 1

        return (mean_ssim/n, mean_psnr/n)

    def show_denoised_image(self):
        self.model.eval()

        ldct, fdct = self.dataset[50]
        ldct, fdct = ldct.unsqueeze(0), fdct.unsqueeze(0)

        fdct = torch.squeeze(fdct)
        fdct = fdct.detach().cpu().numpy()
        ldct_pre_denoised = torch.squeeze(ldct)
        ldct_pre_denoised = ldct_pre_denoised.detach().cpu().numpy()

        noisy_SSIM = structural_similarity(fdct, ldct_pre_denoised, data_range=np.max( [math.ceil(ldct_pre_denoised.max() - ldct_pre_denoised.min()), math.ceil(fdct.max() - fdct.min())] ))
        noisy_PSNR = peak_signal_noise_ratio(fdct, ldct_pre_denoised, data_range=np.max( [math.ceil(ldct_pre_denoised.max() - ldct_pre_denoised.min()), math.ceil(fdct.max() - fdct.min())] ))

        ldct.to(device=self.gpu_id)
        ldct = self.multiple_step_denoising(self.denoising_steps, ldct)

        ldct = torch.squeeze(ldct)
        ldct = ldct.detach().cpu().numpy()

        denoised_SSIM = structural_similarity(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))
        denoised_PSNR = peak_signal_noise_ratio(fdct, ldct, data_range=np.max( [math.ceil(ldct.max() - ldct.min()), math.ceil(fdct.max() - fdct.min())] ))

        print(f'NOISY -- SSIM: {noisy_SSIM}, PSNR: {noisy_PSNR}\nDENOISED -- SSIM: {denoised_SSIM}, PSNR: {denoised_PSNR}')
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(ldct_pre_denoised, cmap='gray')
        ax[0].set_title('LDCT Pre Denoising')
        ax[1].imshow(ldct, cmap='gray')
        ax[1].set_title('LDCT Post Denoising')
        ax[2].imshow(fdct, cmap='gray')
        ax[2].set_title('FDCT')

        f.tight_layout()
        
        f.savefig('test.png', dpi=1200) 

    def model_forward_wrapper(self, model, x, sigma):
        x = x.to(device=self.gpu_id)
        sigma = sigma.to(device=self.gpu_id)

        c_skip = self.skip_scaling(sigma)
        c_out = self.output_scaling(sigma) 
            
        c_skip = self.pad_dims_like(c_skip, x).to(device=self.gpu_id)
        c_out = self.pad_dims_like(c_out, x).to(device=self.gpu_id)

        return c_skip  * x + c_out * model(x, sigma)
    
    def skip_scaling(self, sigma):
        return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
    
    def output_scaling(self, sigma):
        return (self.sigma_data * (sigma - self.sigma_min)) / (self.sigma_data**2 + sigma**2) ** 0.5
    
    def pad_dims_like(self, x, other):
        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))

    def sample(self, model, x, ts): 
        sigma = ts[0:1]
        sigma = torch.full((x.shape[0],), sigma[0], dtype=x.dtype, device=self.gpu_id)
        #sigma = torch.squeeze(sigma,dim=-1) 
        x = self.model_forward_wrapper(model, x, sigma)

        for sigma in ts[1:]:
            z = torch.randn_like(x).to(device=self.gpu_id)
            x = x + math.sqrt(sigma**2 - self.sigma_min**2) * z
            x = self.model_forward_wrapper(model, x, torch.tensor([sigma])) 

        return x
    
    def get_sigmas_linear_reverse(self,n,sigma_min= 0.002,sigma_max=80): 
        sigmas = torch.linspace(sigma_max, sigma_min, n, dtype=torch.float16).to(device=self.gpu_id)
        #print(sigmas)
        return sigmas

    def multiple_step_denoising(self, sample_steps, x): 
        sigmas = self.get_sigmas_linear_reverse(sample_steps, self.sigma_min, self.sigma_max) 
        #print(sigmas)   
        sample_results = self.sample(model=self.model, x=x, ts=sigmas)
        return sample_results

    def test(self):
        multi_step = self.test_mult_step()
        print(f'Multi step denoising:\nSSIM: {multi_step[0]} PSNR: {multi_step[1]}')

def ddp_setup():  
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"])) 

def main():
    with open("parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    ddp_setup()
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
    model.load_state_dict(torch.load('checkpoints/consistency_test_one/consistency_test_one_100_ckpt.pt'))
    model.to(device=gpu_id)

    tester = Tester(model=model, test_root = parameters['test_root'], denoising_steps=1, gpu_id=gpu_id)
    #tester.test()
    tester.show_denoised_image()
    
    destroy_process_group()

if __name__ == "__main__":
    main()