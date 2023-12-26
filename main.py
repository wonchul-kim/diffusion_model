import random
import numpy as np
import os.path as osp
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from ddpm.vis import show_first_batch, show_images, show_forward, generate_new_images
from ddpm.ddpm import DDPM
from ddpm.unet import UNet
from ddpm.train import training_loop

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
input_dir = '/home/wonchul/HDD/datasets/public/mnist'
output_dir = './outputs'

if not osp.exists(output_dir):
    os.mkdir(output_dir)

STORE_PATH_MNIST = osp.join(output_dir, f"ddpm_model_mnist.pt")
STORE_PATH_FASHION = osp.join(output_dir, f"ddpm_model_fashion.pt")

# Notice that `lr=0.001` is the hyper-parameter used by the authors.
no_train = True
fashion = False
batch_size = 128
n_epochs = 2
lr = 0.001


""" Loading data
**NOTE**: It is important to normalize images in range `[-1,1]` and not `[0,1]` as 
one might usually do. This is because the DDPM network predicts normally distributed 
noises throughout the denoising process.
"""

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)
ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn(osp.join(input_dir), download=True, train=True, transform=transform)
loader = DataLoader(dataset, batch_size, shuffle=True)
show_first_batch(loader)


# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

"""# Defining the DDPM module
- `image_chw`: tuple contining dimensionality of images.

The `forward` process of DDPMs benefits from a nice property: 
We don't actually need to slowly add noise step-by-step, 
but we can directly skip to whathever step $t$ we want using coefficients $\alpha_bar$.

For the `backward` method instead, we simply let the network do the job.

Note that in this implementation, $t$ is assumed to be a `(N, 1)` tensor, 
where `N` is the number of images in tensor `x`. 
We thus support different time-steps for multiple images.
"""

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
optim=Adam(ddpm.parameters(), lr)

sum([p.numel() for p in ddpm.parameters()])

"""# Optional visualizations"""

# Optionally, load a pre-trained model that will be further trained
# ddpm.load_state_dict(torch.load(store_path, map_location=device))

# Optionally, show the diffusion (forward) process
show_forward(ddpm, loader, device)

# Optionally, show the denoising (backward) process
generated = generate_new_images(ddpm, gif_name=osp.join(output_dir, "before_training_fashion.gif") if fashion 
                else osp.join(output_dir, "before_training_mnist.gif"))
show_images(generated, "Images generated before training")


# Training
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=optim, device=device, store_path=STORE_PATH_MNIST)

"""# Testing the trained model

Time to check how well our model does. We re-store the best performing model according to our training loss and set it to evaluation mode. Finally, we display a batch of generated images and the relative obtained and nice GIF.
"""

# Loading the trained model
best_model = DDPM(UNet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(STORE_PATH_MNIST, map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name=osp.join(output_dir, "after_training_fashion.gif") if fashion 
                else osp.join(output_dir, "after_training_mnist.gif")
    )
show_images(generated, "Final result")

"""# Visualizing the diffusion"""

from IPython.display import Image

Image(open(osp.join(output_dir, 'after_training_fashion.gif') if fashion else 
           osp.join(output_dir, 'after_training_mnist.gif'),'rb').read())

