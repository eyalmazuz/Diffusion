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
from dataset import Text8Dataset, MoleculeDataset
from model import Bert, BertConfig
from model2 import DynamicsTransformer
from train import train

BATCH_SIZE = 32
PATH = './gdb17.smi'
EPOCHS = 300
NUM_TIMESTEPS = 4000
MAX_SEQ_LEN = 128
DATE = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def main():
    
    dataset = MoleculeDataset(PATH, MAX_SEQ_LEN, return_scaffold=False, return_guidance=False)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f'{dataset.vocab_size=}')
    # config = BertConfig(vocab_size=dataset.vocab_size, n_embd=256, block_size=256, n_heads=4, n_layers=2)
    # model = Bert(config)
    model = DynamicsTransformer(dataset.vocab_size, dim=128, heads=8, depth=4, n_blocks=1,
                                max_seq_len=MAX_SEQ_LEN, num_timesteps=NUM_TIMESTEPS, ff_dropout=0.3,
                                attn_layer_dropout=0.3, n_local_attn_heads=4, local_attn_window_size=64,
                                use_context=False, use_guidance=False)
    model.to(DEVICE)    

    #ema_model = copy.deepcopy(model)
    #ema_model = ema_model.eval()
    optimizer = Adam(model.parameters(), lr=1e-4)

    diffusion_model = Diffusion(model, NUM_TIMESTEPS, num_classes=dataset.vocab_size, schedule='cosine', use_log=True)
    
    train(diffusion_model, dataloader, optimizer, DEVICE, EPOCHS, f'./models/{DATE}', 32, f'./texts/{DATE}', use_context=False, use_guidance=False)

if __name__ == '__main__':
    main()
