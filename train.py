import os
import os.path as osp 
from tqdm.auto import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from sprite_dataset import SpriteDataset
from ddpm import DDPM
from sprites import get_sprites
from debug_dataset import debug_dataset

output_dir = '/mnt/HDD/etc/sprites'
if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
debug = False
device = 'cuda'
checkpoint_path = None
sprites = get_sprites(output_dir)
print(sprites.shape, sprites.min(), sprites.max())

transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

train_dataset = SpriteDataset(sprites, transform)
train_loader = DataLoader(train_dataset, batch_size=m, shuffle=True)

m = 128

if debug:
    train_loader_iter = iter(train_loader)
    samples = next(train_loader_iter)
    debug_dataset(samples)

model = DDPM()
model.to(device)
output = model(samples[0].to(device), samples[3].reshape(-1,1).float().to(device))
print(output.shape)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 20
losses = []

for e in range(epochs):
    epoch_loss = 0.0
    epoch_mae = 0.0

    for i, data in enumerate(tqdm(train_loader)):
        x_0, x_t, eps, t = data
        x_t = x_t.to(device)
        eps = eps.to(device)
        t = t.to(device)

        optimizer.zero_grad()
        eps_theta = model(x_t, t.reshape(-1,1).float())
        loss = loss_func(eps_theta, eps)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item()
            epoch_mae += torch.nn.functional.l1_loss(eps_theta, eps)

    epoch_loss /= len(train_loader)
    epoch_mae /= len(train_loader)

    print(f"Epoch: {e+1:2d}: loss:{epoch_loss:.4f}, mae:{epoch_mae:.4f}")
    losses.append(epoch_loss)
    torch.save(model.state_dict(), osp.join(output_dir, 'last.pth'))
