import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.optim as optim
from models.ddpm_basic import ddpm_simple
from models.unet import UNET
from models.utils import DDPM_Scheduler, set_seed
from timm.utils import ModelEmaV3
import numpy as np
import random
import math
import pdb
from tqdm import tqdm
from typing import List
from datetime import datetime

# Custom dataset class to load CelebA tensor files saved as .pt
class CelebaTensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.files = sorted([
            f for f in os.listdir(tensor_dir)
            if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.tensor_dir, self.files[idx]))


# Training function for DDPM with UNet backbone

# Supports checkpoint loading, EMA tracking, and periodic saving
def train(batch_size: int=128,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None, output_dir: str='output'):
    # Set random seed for reproducibility
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    # Load training dataset (CelebA tensor images)
    train_dataset = CelebaTensorDataset('./data/celeba_tensor64')
    subset = torch.utils.data.Subset(train_dataset, range(30000)) 
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Initialize DDPM scheduler for noise schedule
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    
    #model = nn.DataParallel(UNET(input_channels=3, output_channels=3)).cuda()
    model = UNET(input_channels=3, output_channels=3).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    # Load checkpoint if available
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')
    epoch_losses = []

    for i in range(num_epochs):
        total_loss = 0
        for bidx, x in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}",mininterval=15.0)):
            x = x.cuda()
            # Sample random timestep t for each image
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)  # True noise
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)  # Add noise to image
            # Model predicts noise from noisy image
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)  # Compare prediction to actual noise
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)  # Update EMA model
        avg_epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch {i+1} | Loss {avg_epoch_loss:.5f}')
        if (i + 1) % 10 == 0 or (i + 1) == num_epochs:
            torch.save({
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }, f'checkpoints/ddpm_checkpoint_epoch_{i+1}.pth')
    # checkpoint = {
    #     'weights': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'ema': ema.state_dict()
    # }
    # torch.save(checkpoint, 'checkpoints/ddpm_checkpoint_150')
    loss_figure(num_epochs,epoch_losses, output_dir)

# Plot loss after training
def loss_figure(num_epochs:int, epoch_losses: List, output_dir: str):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()


# Inference function to generate images from random noise
def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999, output_dir: str='output'):
    # load checkpoint
    # checkpoint = torch.load(checkpoint_path)
    # Because we use DataParallel, the model's state_dict keys have "module." prefix in the checkpoint. 
    # We need to remove it before loading the state_dict.
    # new_state_dict = {}
    # for k, v in checkpoint['weights'].items():
    #     print(k)
    #     new_key = k.replace('module.', '')  
    #     print(new_key)
    #     new_state_dict[new_key] = v

    # model = UNET(input_channels=3, output_channels=3).cuda()
    # model.load_state_dict(new_state_dict)

    checkpoint = torch.load(checkpoint_path)
    model = UNET(input_channels=3, output_channels=3).cuda()
    model.load_state_dict(checkpoint['weights'])

    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    # Key timesteps to visualize
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []
    print("Starting inference process...")
    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 3, 32, 32)
            # # Add some truth image in the beginning
            # real_x = test_dataset[i][0].unsqueeze(0).cuda()
            # real_x = F.pad(real_x, (2,2,2,2))
            # images = [real_x.clone().cpu()]
            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
                z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())
                if t[0] in times:
                    images.append(z.clone())
                e = torch.randn(1, 3, 32, 32)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())

            images.append(x.clone())

            # Save reverse process timeline
            display_reverse(images, i, output_dir)


            # x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            # x = x.numpy()
            # plt.imshow(x)
            # plt.show()
            # display_reverse(images)
            images = []

def display_reverse(images: List, index: int,output_dir: str):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = (x + 1) / 2 # Normalize to [0, 1]
        x = x.clamp(0, 1)  
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timestep_{index}.png', bbox_inches='tight')
    plt.close()


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = f'output/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    #train(checkpoint_path='checkpoints/ddpm_checkpoint_epoch_10.pth', lr=2e-6, num_epochs=50, output_dir=output_dir)
    train(checkpoint_path=None, lr=2e-6, num_epochs=50, output_dir=output_dir)
    #print("Training finished. Starting inference...")
    #inference('checkpoints/ddpm_checkpoint_150', output_dir=output_dir)
    #print("Inference finished. Congradulations!")

if __name__ == '__main__':
    main()
