import os.path as osp 
import os
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from utils import *

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

from model import ContextUnet
from test import sample_wo_context, sample_w_context
from dataset import CustomDataset, transform
from train import train_wo_context, train_w_context

save_dir = '/HDD/outputs/ddpm'

if not osp.exists(save_dir):
    os.mkdir(save_dir)

n_epochs = 32
lr = 1e-3
batch_size = 32

time_steps = 500
beta_1 = 1e-4
beta_2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
n_features = 64
n_cfeatures = 5
imgsz = 16
    
### DDPM noise scheduler
b_t = (beta_2 - beta_1)*torch.linspace(0, 1, time_steps + 1, device=device) + beta_1
a_t = 1 - b_t 
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

plt.plot(a_t.cpu().detach().numpy(), label='a_t')
plt.plot(b_t.cpu().detach().numpy(), label='b_t')
plt.plot(ab_t.cpu().detach().numpy(), label='ab_t')
plt.legend()
plt.savefig(osp.join(save_dir, 'noise.png'))
plt.close()

model = ContextUnet(in_channels=3, n_feat=n_features, n_cfeat=n_cfeatures,
                    height=imgsz).to(device)

# ########## Sampling 1
# model.load_state_dict(torch.load(osp.join(str(ROOT), 'weights/model_trained.pth'), map_location=device))
# model.eval()

# plt.clf()
# n_samples = 4
# samples, inter_ddpm = sample_wo_context(model, n_samples, time_steps, imgsz, 
#                 device, a_t, b_t, ab_t, save_rate=20)
# n_rows = 2
# animation_ddpm = plot_sample(inter_ddpm, n_samples, n_rows, save_dir, 
#                              "ani_run", None, save=True)
        
########## Sampling 2
model.load_state_dict(torch.load(osp.join(str(ROOT), 'weights/model_trained.pth'), map_location=device))
model.eval()

# plt.clf()
# n_samples = 4
# ctx = F.one_hot(torch.randint(0, n_cfeatures, (n_samples,)), n_cfeatures).to(device=device).float()
# samples, inter_ddpm = sample_w_context(model, n_samples, ctx, time_steps, imgsz, 
#                 device, a_t, b_t, ab_t, save_rate=20)
# n_rows = 2
# animation_ddpm = plot_sample(inter_ddpm, n_samples, n_rows, save_dir, 
#                              "ani_run", None, save=True)
        
        
# user defined context
ctx = torch.tensor([
    # hero, non-hero, food, spell, side-facing
    [1,0,0,0,0],  
    [1,0,0,0,0],    
    [0,0,0,0,1],
    [0,0,0,0,1],    
    [0,1,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
]).float().to(device)
samples, _ = sample_w_context(model, ctx.shape[0], ctx, time_steps, imgsz, 
                device, a_t, b_t, ab_t, save_rate=20)

show_images(samples, save_dir=save_dir)

# # mix of defined context
# ctx = torch.tensor([
#     # hero, non-hero, food, spell, side-facing
#     [1,0,0,0,0],      #human
#     [1,0,0.6,0,0],    
#     [0,0,0.6,0.4,0],  
#     [1,0,0,0,1],  
#     [1,1,0,0,0],
#     [1,0,0,1,0]
# ]).float().to(device)
# samples, _ = sample_ddpm_context(ctx.shape[0], ctx)
# show_images(samples)


# ############### Training
# dataset = CustomDataset(osp.join(str(ROOT), "sprites/sprites_1788_16x16.npy"), 
#                         osp.join(str(ROOT), "sprites/sprite_labels_nc_1788_16x16.npy"), 
#                         transform=transform, null_context=False)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # train_wo_context(model, dataloader, lr, device, n_epochs, optimizer, time_steps, ab_t, save_dir)
# # train_w_context(model, dataloader, lr, device, n_epochs, optimizer, time_steps, ab_t, save_dir)

