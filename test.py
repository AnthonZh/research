import torch
import numpy as np
import yaml
import os

from torch.distributed import init_process_group, destroy_process_group

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

from utils.dataset import LDCTLoader

class Tester:
    def __init__(self, model, test_root, gpu_id, sigma_min: float = 0.002, sigma_data: float = 0.5, sigma_max: float = 80.):
        self.model = model
        self.model.eval()

        self.gpu_id = gpu_id
        self.sigma_max = sigma_max
        
        self.dataloader = LDCTLoader(test_root, 1).dataloader
    
    def test_one_step():
        self.model.eval()

        mean_ssim = 0.
        mean_psnr = 0.
        n = len(self.dataloader)

        for ldct, fdct in self.dataloader:
            ldct.to(device=self.gpu_id)
            ldct = self.model_forward_wrapper(self.model, ldct, self.sigma_max)
            ldct = torch.squeeze(ldct)
            fdct = torch.squeeze(fdct)

            ldct = ldct.detach().cpu().numpy()
            fdct = fdct.detach().cpu().numpy()

            mean_ssim += structural_similarity(fdct, ldct, data_range=np.max( ldct.max() - ldct.min(), fdct.max() - fdct.min() ))
            mean_psnr += peak_signal_noise_ratio(fdct, ldct, data_range=np.max( ldct.max() - ldct.min(), fdct.max() - fdct.min() ))
        
        return (mean_ssim/n, mean_psnr/n)

    def test_mult_step():
        self.model.eval()
        
        mean_ssim = 0
        mean_psnr = 0
        n = len(self.dataloader)

        for ldct, fdct in self.dataloader:
            ldct.to(device=self.gpu_id)
            ldct = self.multiple_step_denoising(ldct)
            
            ldct, fdct = torch.squeeze(ldct), torch.squeeze(fdct)
            ldct, fdct = ldct.detach().cpu().numpy(), fdct.detach().cpu().numpy()

            mean_ssim += structural_similarity(fdct, ldct, data_range=np.max( ldct.max() - ldct.min(), fdct.max() - fdct.min() ))
            mean_psnr += peak_signal_noise_ratio(fdct, ldct, data_range=np.max( ldct.max() - ldct.min(), fdct.max() - fdct.min() ))

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
    
    def pad_dims_like(self, x, other):
        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))

    def sample(self, model, x, ts): 
        sigma = ts[0]
        x = self.model_forward_wrapper(model, x, sigma)

        for sigma in ts[1:]:
            z = torch.randn_like(x).to(device=self.gpu_id)
            x = x + math.sqrt(sigma**2 - self.sigma_min**2) * z
            x = self.model_forward_wrapper(model, x, sigma) 

        return x
    
    def get_sigmas_linear_reverse(self,n,sigma_min= 0.002,sigma_max=80): 
        sigmas = torch.linspace(sigma_max, sigma_min, n, dtype=torch.float16).to(device=self.gpu_id)
        return sigmas

    def multiple_step_denoising(self, x): 
        sigmas = self.get_sigmas_linear_reverse(sample_step,self.sigma_min,self.sigma_max) 
        sample_results = self.sample(model=self.model, x, ts=sigmas)
        return sample_results

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
    model.load_state_dict(torch.load('_______________________________________________________________________'))

    tester = Tester(model=model, test_root = parameters['test_root'], gpu_id=gpu_id)
    
    one_step = tester.test_one_step()
    print(f'One step denoising\nSSIM: {one_step[0]} PSNR: {one_step[1]}')

    multi_step = tester.test_mult_step()
    print(f'Multi step denoising\nSSIM: {multi_step[0]} PSNR: {multi_step[1]}')
    
    destroy_process_group()

if __name__ == "__main__":
    main()