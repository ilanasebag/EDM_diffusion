import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import tqdm 
from tqdm import * 
from utils import UNet
from helpers import * 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet()
net.to(device)

def train(dataset, epochs=5, learning_rate=1e-3, batch_size=128) : 

    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    #net = UNet()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Keeping a record of the losses
    losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):

        for x, y in train_dataloader:

            # Get some data and prepare the corrupted version
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(
                device)  # Pick random noise amounts
            noisy_x = gaussian_noise_corruption(x, noise_amount)  # Create our noisy x

            # Get the model prediction
            pred = net(noisy_x)

            # Calculate the loss
            loss = loss_fn(pred, x)

            # Backprop and update the params:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the loss for later
            losses.append(loss.item())

        # Print out the average of the loss values for this epoch:
        avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
        print(
            f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

    # View the loss curve
    plt.plot(losses)
    plt.ylim(0, 0.1)
    return losses 

