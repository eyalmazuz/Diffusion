import math
import os
import random

import cv2
import torch

from rdkit import Chem
from rdkit.Chem.QED import qed

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
def eval(diffusion_model, ema_model, device, sample_count, num_classes, save_path, itos, context=None, context_mask=None, guidance=None):
    diffusion_model.model.eval()
    # x_t = torch.randint(low=0, high=num_classes, size=(sample_count, 256)).to(device)
    # output = torch.clone(x_t)
    # output = index_to_onehot(output, num_classes, diffusion_model.use_log)
    # for timestep in tqdm(reversed(range(diffusion_model.timesteps))):
        # output = diffusion_model.sample_p(output, torch.full((sample_count, ), timestep, device=device, dtype=torch.long), model=ema_model)

    output = diffusion_model.sample(sample_count, seq_len=diffusion_model.model.decoder.max_seq_len,
                                    model=ema_model, context=context, context_mask=context_mask, guidance=guidance)
    
    os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)
    with open(save_path, 'w') as f:
        if context is not None:
            scaffold = ''.join([itos[idx] for idx in context.cpu().numpy().tolist()])
            f.write(f'Scaffold: {scaffold}\n')
        
        if guidance is not None:
            f.write(f'QED: {guidance.cpu().item()}\n')
        for i, tokens in enumerate(output):
            text = ''.join([itos[i.item()] for i in tokens])
            f.write(text + '\n')

def train(diffusion_model, dataloader, optimizer, device, epochs, model_save_path, sample_count, save_path, use_context=True, use_guidance=True):
    # scaler = GradScaler()

    epoch_lr = ExponentialLR(optimizer, gamma=0.99)
    # step_lr = LinearWarmupScheduler(optimizer, 500)
    ema_model = AveragedModel(diffusion_model.model, avg_fn=lambda averaged_model_parameter, model_parameter, _: 0.99 * averaged_model_parameter + 0.01 * model_parameter)

    diffusion_model.model.train()
    for epoch in range(epochs):
        loss_moving = None
        loss_count = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()


            for k, v in batch.items():
                batch[k] = v.to(device)

            #with autocast():
            loss = - diffusion_model.loss(**batch).sum() / (math.log(2) * torch.prod(torch.tensor(batch['x_start'].size()[:2])).to(device))
            # print(loss, loss.mean())
            loss.backward()
            optimizer.step()
            #step_lr.step()

            ema_model.update_parameters(diffusion_model.model)
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            loss_count += len(batch['x_start'])



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
                if use_context:
                    scaffold = random.choice(dataloader.dataset.data)
                    scaffold_tokens = [dataloader.dataset.stoi[c] if c in dataloader.dataset.stoi else dataloader.dataset.stoi['[UNK]'] for c in scaffold]

                    mask = [dataloader.dataset.stoi['[PAD]']] * (diffusion_model.model.decoder.max_seq_len - len(scaffold_tokens))
                    context_mask = [True] * len (scaffold_tokens) + [False] * (diffusion_model.model.decoder.max_seq_len - len(scaffold_tokens))

                if use_guidance:
                    qed_score = qed(Chem.MolFromSmiles(scaffold))

                eval(diffusion_model, ema_model, device, sample_count,
                    len(dataloader.dataset.stoi), save_path=os.path.join(save_path, f'{epoch+1}', f'{step+1}.txt'),
                    itos=dataloader.dataset.itos,
                    context=torch.tensor(scaffold_tokens + mask, device=device) if use_context else None,
                    context_mask=torch.tensor(context_mask, device=device) if use_context else None,
                    guidance=torch.tensor(qed_score, device=device) if use_guidance else None)
                
                diffusion_model.model.train()

        epoch_lr.step()
