import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'device': device,
    
    'epochs': 500000,
    'shuffle': True,
    'pin_memory': True,
    'num_workers': 18,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'dataroot_gt': 'data/DF2K_train_HR',
    'dataroot_lq': 'data/DF2K_train_LR_bicubic/X4',
    'valid_dataroot_gt': 'data/DF2K_valid_HR',
    'valid_dataroot_lq': 'data/DF2K_valid_LR_bicubic/X4'
    
}