from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

from PIL import Image
import torchvision.transforms.functional as F
import random

class PairedImageDataset(Dataset):
    def __init__(self, dataroot_gt, dataroot_lq, train=bool, gt_crop_size=64, scale=4):
        super().__init__()
        self.train = train
        self.dataroot_gt = dataroot_gt
        self.dataroot_lq = dataroot_lq

        self.gt_images = sorted(os.listdir(dataroot_gt))
        self.lq_images = sorted(os.listdir(dataroot_lq))

        self.gt_crop_size = gt_crop_size
        self.scale = scale
        self.lq_crop_size = gt_crop_size // scale

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.dataroot_gt, self.gt_images[idx])
        lq_path = os.path.join(self.dataroot_lq, self.lq_images[idx])

        gt_img = Image.open(gt_path).convert('RGB')
        lq_img = Image.open(lq_path).convert('RGB')

        i, j, h, w = transforms.RandomCrop.get_params(gt_img, output_size=(self.gt_crop_size, self.gt_crop_size))
        gt_crop = F.crop(gt_img, i, j, h, w)

        i_lq, j_lq = i // self.scale, j // self.scale
        lq_crop = F.crop(lq_img, i_lq, j_lq, self.lq_crop_size, self.lq_crop_size)

        if random.random() > 0.5:
            gt_crop = F.hflip(gt_crop)
            lq_crop = F.hflip(lq_crop)
        if random.random() > 0.5:
            gt_crop = F.vflip(gt_crop)
            lq_crop = F.vflip(lq_crop)
        if random.random() > 0.5:
            angle = random.choice([0, 90, 180, 270])
            gt_crop = F.rotate(gt_crop, angle)
            lq_crop = F.rotate(lq_crop, angle)

        gt_tensor = self.to_tensor(gt_crop)
        lq_tensor = self.to_tensor(lq_crop)
        
        return lq_tensor, gt_tensor
