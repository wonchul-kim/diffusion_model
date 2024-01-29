import gdown 
import os.path as osp 
import numpy as np 


def get_sprites(output_dir, url='https://drive.google.com/uc?id=1gADYmo2UXlr24dUUNaqyPF2LZXk1HhrJ'):
    file_path = osp.join(output_dir, 'sprites_1788_16x16.npy')
    if not osp.exists(file_path):
        gdown.download(url, file_path, quiet=False)

    sprites = np.load(file_path)
    
    return sprites