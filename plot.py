



import torch
import torchvision
from matplotlib import pyplot as plt

from helpers import * 
from train import *  

##### plotting #####

def plot_data(x, data=False, corrupted =False, predicted=False):
    
    if data == True: 
        fig, axs = plt.subplots(1, 1, figsize=(10,5))
        axs.set_title('Input data')
        axs.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

    if corrupted == True:

        amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
        corrupted_x = gaussian_noise_corruption(x, amount)
        
        fig, axs = plt.subplots(1, 1, figsize=(10,5))
        axs.set_title('Corrupted data (amount increases -->)')
        axs.imshow(torchvision.utils.make_grid(corrupted_x)[0], cmap='Greys')
        
    if predicted == True:
        
        with torch.no_grad():
            preds = net(corrupted_x.to(device)).detach().cpu()
        fig, axs = plt.subplots(1, 1, figsize=(10,5))
        axs.set_title('Predicted data')
        axs.imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')