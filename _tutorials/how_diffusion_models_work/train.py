import torch 
import torch.nn.functional as F

import os
import os.path as osp 
from tqdm import tqdm 

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

from utils import perturb_input

def train_wo_context(model, dataloader, lr, device, n_epochs, optimizer, time_steps, ab_t, save_dir):
    model.train()
    for epoch in range(n_epochs):
        optimizer.param_groups[0]['lr'] = lr*(1-epoch/n_epochs)
        
        pbar = tqdm(dataloader, mininterval=2, desc=f"Epoch: {epoch}")
        for x, _ in pbar:
            optimizer.zero_grad()

            x = x.to(device)
            
            noise = torch.randn_like(x)
            t = torch.randint(1, time_steps + 1, (x.shape[0], )).to(device)
            x_pert = perturb_input(x, t, noise, ab_t)
            
            pred_noise = model(x_pert, t/time_steps)
            
            loss = F.mse_loss(pred_noise, noise)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        

        # save model periodically
        if epoch%9==0 or epoch == int(n_epochs - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + f"model_{epoch}.pth")
            print('saved model at ' + save_dir + f"model_{epoch}.pth")
            
            
def train_w_context(model, dataloader, lr, device, n_epochs, optimizer, time_steps, ab_t, save_dir):
    model.train()
    for epoch in range(n_epochs):
        optimizer.param_groups[0]['lr'] = lr*(1-epoch/n_epochs)
        
        pbar = tqdm(dataloader, mininterval=2, desc=f"Epoch: {epoch}")
        for x, c in pbar:
            optimizer.zero_grad()

            x = x.to(device)
            c = c.to(device)
            
            # randomly mask out c
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)
            
            noise = torch.randn_like(x)
            t = torch.randint(1, time_steps + 1, (x.shape[0], )).to(device)
            x_pert = perturb_input(x, t, noise, ab_t)
            
            pred_noise = model(x_pert, t/time_steps, c=c)
            
            loss = F.mse_loss(pred_noise, noise)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        

        # save model periodically
        if epoch%4==0 or epoch == int(n_epochs-1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + f"context_model_{epoch}.pth")
            print('saved model at ' + save_dir + f"context_model_{epoch}.pth")