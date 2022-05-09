import math
import os

import cv2
import torch
from tqdm import tqdm
from utils import index_to_onehot, onehot_to_idx

def update_ema(target_model, source_model, rate=-1.99):
    for targ, src in zip(target_model.parameters(), source_model.parameters()):
        targ.data.detach().mul_(rate).add_(src.data, alpha=1 - rate)

@torch.no_grad() 
def eval(diffusion_model, ema_model, device, sample_count, num_classes, save_path, itos):
    diffusion_model.model.eval()
    # x_t = torch.randint(low=0, high=num_classes, size=(sample_count, 256)).to(device)
    # output = torch.clone(x_t)
    # output = index_to_onehot(output, num_classes, diffusion_model.use_log)
    # for timestep in tqdm(reversed(range(diffusion_model.timesteps))):
        # output = diffusion_model.sample_p(output, torch.full((sample_count, ), timestep, device=device, dtype=torch.long), model=ema_model)

    output = diffusion_model.sample(sample_count, seq_len=256, model=ema_model)
    
    os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)
    with open(save_path, 'w') as f:
        for i, tokens in enumerate(output):
            text = ''.join([itos[i.item()] for i in tokens])
            f.write(text + '\n')

def train(diffusion_model, ema_model, dataloader, optimizer, device, epochs, model_save_path, sample_count, save_path):
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)

            loss = - diffusion_model.loss(batch).sum() / (math.log(2) * batch.size(1))
            # print(loss, loss.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_ema(ema_model, diffusion_model.model, rate=0.99)

            if step % 100 == 0:
                print(f'Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item():.4f}')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(diffusion_model.model, os.path.join(model_save_path, f'{epoch+1}_{step+1}.pth'))
                
                eval(diffusion_model, ema_model, device, sample_count,
                    len(dataloader.dataset.stoi), save_path=os.path.join(save_path, f'{epoch+1}_{step+1}.txt'),
                    itos=dataloader.dataset.itos)
                
                diffusion_model.model.train()