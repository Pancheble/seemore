import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from core.networks import SeemoRe
from accelerate import Accelerator
from core.datasets import PairedImageDataset
from core.config import config
from core.utils import calculate_psnr, calculate_ssim


# parser = argparse.ArgumentParser()
# parser.add_argument('--dir01', type=str, default=config['valid_dataroot_gt'])
# parser.add_argument('--dir02', type=str, default=config['valid_dataroot_lq'])
# args = parser.parse_args()

accelerator = Accelerator()

model = SeemoRe()
dataset = PairedImageDataset(config['valid_dataroot_gt'], config['valid_dataroot_lq']) # todo: argument 작성 + valid dataset 만들기
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

model, dataloader = accelerator.prepare(model, dataloader)

model_path = 'Epoch146_Loss0.2017_PSNR22.5198_SSIM0.9993_model.pth'
checkpoint = torch.load(model_path, map_location=accelerator.device)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    for imgs_lq, imgs_gt in dataloader:
        
        output = model(imgs_lq)
        print(imgs_gt.shape, imgs_lq.shape, output.shape)

        for i in range(output.size(0)):
            psnr = calculate_psnr(output[i], imgs_gt[i], crop_border=4)
            ssim = calculate_ssim(output[i], imgs_gt[i], crop_border=4)

            total_psnr += psnr
            total_ssim += ssim
            num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

accelerator.print(f'PSNR: {avg_psnr:.4f}')
accelerator.print(f'SSIM: {avg_ssim:.4f}')

# todo: argument작성
# accelerate launch infer.py --dir1(gt) --dir2(lq) --model_path(model.pth)