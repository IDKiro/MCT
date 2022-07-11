import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import GeneratorResNet
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--task', type=str, default='day2dusk')
parser.add_argument('--dataset_dir', type=str, default='./data/')
parser.add_argument('--result_dir', type=str, default='./results/base/')
parser.add_argument('--save_dir', type=str, default='./pretrained_models/')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    netG_A2B = GeneratorResNet()
    netG_B2A = GeneratorResNet()
    netG_A2B.cuda()
    netG_B2A.cuda()

    netG_A2B.load_state_dict(torch.load(os.path.join(args.save_dir, args.task, 'G_A2B.pth')))
    netG_B2A.load_state_dict(torch.load(os.path.join(args.save_dir, args.task, 'G_B2A.pth')))

    netG_A2B.eval()
    netG_B2A.eval()

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    dataloader = DataLoader(ImageDataset(os.path.join(args.dataset_dir, args.task), transforms_=transforms_, mode='test'), 
                            batch_size=1, shuffle=False, num_workers=args.workers)

    os.makedirs(os.path.join(args.result_dir, args.task, 'A2B'), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, args.task, 'B2A'), exist_ok=True)

    for i, batch in tqdm(enumerate(dataloader)):
        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()

        filename_A = batch['filename_A'][0]
        filename_B = batch['filename_B'][0]

        with torch.no_grad():
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)

        save_image(0.5 * (fake_B + 1.0), os.path.join(args.result_dir, args.task, 'A2B', filename_A))
        save_image(0.5 * (fake_A + 1.0), os.path.join(args.result_dir, args.task, 'B2A', filename_B))
