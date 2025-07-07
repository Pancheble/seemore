import csv
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from core.networks import SeemoRe
from core.datasets import PairedImageDataset
from core.config import config
from core.utils import calculate_psnr, calculate_ssim, setting_csv

csv_path = 'training_log.csv'
accelerator = Accelerator()
device = accelerator.device

model = SeemoRe()
dataset = PairedImageDataset(config['dataroot_gt'], config['dataroot_lq'])
dataloader = DataLoader(dataset,
                        shuffle=config['shuffle'],
                        pin_memory=config['pin_memory'],
                        num_workers=config['num_workers'],
                        batch_size=config['batch_size'])

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    total_steps=len(dataloader) * config['epochs'],
    pct_start=0.4,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4
)

model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

model.train()

best_psnr = 0
best_ssim = 0
best_loss = float('inf')

setting_csv(csv_path=csv_path)

for epoch in range(config['epochs']):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for i, (lq, gt) in loop:
        optimizer.zero_grad()
        
        with accelerator.autocast():
            output = model(lq)
            loss = criterion(output, gt)
        
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        
        psnr_value = calculate_psnr(output[0], gt[0], crop_border=4)
        ssim_value = calculate_ssim(output[0], gt[0], crop_border=4)
        loss_value = loss.item()
        current_lr = scheduler.get_last_lr()[0]
        
        loop.set_postfix({
            'Loss': f'{loss_value:.4f}',
            'PSNR': f'{psnr_value:.4f}',
            'SSIM': f'{ssim_value:.4f}',
            'LR': f'{current_lr:.2e}'
        })

    if accelerator.is_main_process:
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                i + 1,
                f"{loss_value:.4f}",
                f"{psnr_value:.4f}",
                f"{ssim_value:.4f}",
                f"{current_lr:.2e}"
            ])
            
        if loss_value < best_loss:
            try:
                os.remove(save_path)
            except (FileNotFoundError, NameError):
                pass
            best_psnr = psnr_value
            best_ssim = ssim_value
            best_loss = loss_value

            save_path = f'Epoch{epoch+1}_Loss{loss_value:.4f}_PSNR{psnr_value:.4f}_SSIM{ssim_value:.4f}_model.pth'
            accelerator.save(model.state_dict(), save_path)
