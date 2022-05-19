import math
import os

import cv2
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, _LRScheduler
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from utils import index_to_onehot, onehot_to_idx

class LinearWarmupScheduler(_LRScheduler):
    """ Linearly warm-up (increasing) learning rate, starting from zero.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epoch: target learning rate is reached at total_epoch.
    """

    def __init__(self, optimizer, total_epoch, last_epoch=-1):
        self.total_epoch = total_epoch
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1, (self.last_epoch / self.total_epoch)) for base_lr in self.base_lrs]


def update_ema(target_model, source_model, rate=0.99):
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

def train(diffusion_model, dataloader, optimizer, device, epochs, model_save_path, sample_count, save_path):
    # scaler = GradScaler()

    epoch_lr = ExponentialLR(optimizer, gamma=0.99)
    # step_lr = LinearWarmupScheduler(optimizer, 500)
    ema_model = AveragedModel(diffusion_model.model, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: 0.99 * averaged_model_parameter + 0.01 * model_parameter)

    diffusion_model.model.train()
    for epoch in range(epochs):
        loss_moving = None
        loss_count = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            if isinstance(batch, tuple):
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
            
            else:
                input_ids = batch.to(device)
                input_mask = None

            #with autocast():
            loss = - diffusion_model.loss(input_ids, input_mask=input_mask).sum() / (math.log(2) * torch.prod(torch.tensor(batch.size()[:2])).to(device))
            # print(loss, loss.mean())
            loss.backward()
            optimizer.step()
            #step_lr.step()

            ema_model.update_parameters(diffusion_model.model)
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            loss_count += len(batch)



            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            #update_ema(ema_model, diffusion_model.model, rate=0.99)

            print('Training. Epoch: {}/{}, Step: {}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, epochs, step, loss_count, len(dataloader.dataset), loss_moving), end='\r')
            if step % 100 == 0 and step != 0:
                #print(f'Epoch: {epoch+1}, Step: {step+1}, Loss: {loss_moving:.4f}')
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(diffusion_model.model, os.path.join(model_save_path, f'{epoch+1}_{step+1}.pth'))
                
            if step % 2500 == 0 and step != 0:
                print('Training. Epoch: {}/{}, Step: {}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, epochs, step, loss_count, len(dataloader.dataset), loss_moving))
                eval(diffusion_model, ema_model, device, sample_count,
                    len(dataloader.dataset.stoi), save_path=os.path.join(save_path, f'{epoch+1}', f'{step+1}.txt'),
                    itos=dataloader.dataset.itos)
                
                diffusion_model.model.train()

        epoch_lr.step()
