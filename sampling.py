
import torch
from train import *

def sampling(steps):
    x = torch.rand(8, 1, 28, 28).to(device) # Start from random
    step_history = [x.detach().cpu()]
    pred_output_history = []
    for i in range(steps):
        with torch.no_grad(): # No need to track gradients during inference
            pred = net(x) # Predict the denoised x0
        pred_output_history.append(pred.detach().cpu()) # Store model output for plotting
        mix_factor = 1/(steps - i) # How much we move towards the prediction
        x = x*(1-mix_factor) + pred*mix_factor # Move part of the way there
        step_history.append(x.detach().cpu()) # Store step for plotting
    return step_history, pred_output_history