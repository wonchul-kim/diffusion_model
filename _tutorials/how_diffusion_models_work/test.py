import torch 
import numpy as np 

from utils import denoise

@torch.no_grad()
def sample_wo_context(model, n_samples, time_steps, imgsz, 
                device, a_t, b_t, ab_t, save_rate=20):

    samples = torch.randn(n_samples, 3, imgsz, imgsz).to(device)
    
    intermediate = []
    for i in range(time_steps, 0, -1):
        print(f'sampling time step {i:3d}', end='\r')
        
        t = torch.tensor([i / time_steps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0

        pred_noise = model(samples, t) # e_(x_t, t)
        samples = denoise(samples, i, pred_noise, a_t, b_t, ab_t, z)
        if i % save_rate ==0 or i == time_steps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

@torch.no_grad()
def sample_w_context(model, n_samples, context, time_steps, imgsz, 
                device, a_t, b_t, ab_t, save_rate=20):

    samples = torch.randn(n_samples, 3, imgsz, imgsz).to(device)
    
    intermediate = []
    for i in range(time_steps, 0, -1):
        print(f'sampling time step {i:3d}', end='\r')
        
        t = torch.tensor([i / time_steps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0

        pred_noise = model(samples, t, c=context) # e_(x_t, t, c_tx)
        samples = denoise(samples, i, pred_noise, a_t, b_t, ab_t, z)
        if i % save_rate ==0 or i == time_steps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
