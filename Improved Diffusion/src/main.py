import copy
from datetime import datetime
import os

import cv2 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from tqdm import tqdm

from diffusion import Diffusion
from dataset import ImageDataset
from model import MiniUNet
from train import train

BATCH_SIZE = 32
PATH = './images'
EPOCHS = 20
DATE = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def main():
    
    dataset = ImageDataset(PATH, (256, 256))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    cv2.imwrite('./foo.png', dataset[0].permute(1, 2, 0).numpy() * 127.5 + 127.5)
    model = MiniUNet(3, 128, 6, 64)
    model.to(DEVICE)    

    ema_model = copy.deepcopy(model)
    optimizer = Adam(model.parameters(), lr=2e-4)

    diffusion_model = Diffusion(model, 4000)
    
    train(diffusion_model, ema_model, dataloader, optimizer, DEVICE, EPOCHS, f'./models/{DATE}', 4, f'./images/{DATE}')

if __name__ == '__main__':
    main()