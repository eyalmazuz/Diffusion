import os

import cv2
import torch
from tqdm import tqdm

def update_ema(target_model, source_model, rate=-1.99):
    for targ, src in zip(target_model.parameters(), source_model.parameters()):
        targ.data.detach().mul_(rate).add_(src.data, alpha=1 - rate)

@torch.no_grad() 
def eval(diffusion_model, ema_model, device, sample_count, save_path):
    diffusion_model.model.eval()
    x_t = torch.randn(sample_count, 3, 256, 256).to(device)
    output = torch.clone(x_t)
    for timestep in tqdm(reversed(range(diffusion_model.timesteps))):
        output = diffusion_model.p_sample(output, torch.Tensor([timestep]).long().to(device), model=ema_model)

    output = output.cpu().numpy().transpose(0, 2, 3, 1)

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(output):
        cv2.imwrite(os.path.join(save_path, f'target_{i}.png'), image * 127.5 + 127.5)

def train(diffusion_model, ema_model, dataloader, optimizer, device, epochs, model_save_path, sample_count, save_path):
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)

            loss = diffusion_model.loss(batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_ema(ema_model, diffusion_model.model, rate=0.99)

            if step % 100 == 0:
                print(f'Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item():.4f}')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(diffusion_model.model, os.path.join(model_save_path, f'{epoch+1}_{step+1}.pth'))
                
                eval(diffusion_model, ema_model, device, sample_count, save_path=os.path.join(save_path, f'{epoch+1}_{step+1}'))
                
                diffusion_model.model.train()