import copy
from datetime import datetime
import os

import cv2 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from diffusion import Diffusion
from dataset import Text8Dataset
from model import Bert, BertConfig
from model2 import DynamicsTransformer
from train import train

BATCH_SIZE = 32
PATH = './text8'
EPOCHS = 20
DATE = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def main():
    
    dataset = Text8Dataset(PATH, 256)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f'{dataset.vocab_size=}')
    # config = BertConfig(vocab_size=dataset.vocab_size, n_embd=256, block_size=256, n_heads=4, n_layers=2)
    # model = Bert(config)
    model = DynamicsTransformer(dataset.vocab_size,)
    model.to(DEVICE)    

    ema_model = copy.deepcopy(model)
    ema_model = ema_model.eval()
    optimizer = Adam(model.parameters(), lr=2e-4)

    diffusion_model = Diffusion(model, 1000, num_classes=dataset.vocab_size, schedule='cosine', use_log=True)
    
    train(diffusion_model, ema_model, dataloader, optimizer, DEVICE, EPOCHS, f'./models/{DATE}', 32, f'./texts/{DATE}')

if __name__ == '__main__':
    main()