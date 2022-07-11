import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from models import Generator, PatchDiscriminator
from utils import ReplayBuffer, LambdaLR, set_requires_grad
from datasets import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--G_lr', type=float, default=2e-4)
parser.add_argument('--D_lr', type=float, default=2e-4)
parser.add_argument('--resize_size', type=int, default=288)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--sample_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--task', type=str, default='day2dusk')
parser.add_argument('--dataset_dir', type=str, default='./data/')
parser.add_argument('--pretrained_dir', type=str, default='./pretrained_models/')
parser.add_argument('--save_dir', type=str, default='./saved_models/')
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    netG_A2B = Generator()
    netG_B2A = Generator()
    netD_A = PatchDiscriminator()
    netD_B = PatchDiscriminator()

    netG_A2B.load_pretrain(os.path.join(args.pretrained_dir, args.task, 'G_A2B.pth'))
    netG_B2A.load_pretrain(os.path.join(args.pretrained_dir, args.task, 'G_B2A.pth'))
    netD_A.load_state_dict(torch.load(os.path.join(args.pretrained_dir, args.task, 'D_A.pth')))
    netD_B.load_state_dict(torch.load(os.path.join(args.pretrained_dir, args.task, 'D_B.pth')))

    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

    criterion_GAN = nn.MSELoss()
    criterion_identity = nn.L1Loss()
    criterion_cycle = nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=args.D_lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [ transforms.Resize(args.resize_size, Image.BICUBIC), 
                    transforms.RandomCrop(args.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    dataloader = DataLoader(ImageDataset(os.path.join(args.dataset_dir, args.task), transforms_=transforms_, unaligned=True), 
                            batch_size=args.batchSize, shuffle=True, drop_last=True, num_workers=args.workers)

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))

    os.makedirs(os.path.join(args.save_dir, args.task), exist_ok=True)

    step = 0
    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].cuda()
            real_B = batch['B'].cuda()

            ''' G update '''
            set_requires_grad([netD_A, netD_B], False)

            # forward
            fake_B_base, fake_B = netG_A2B(real_A)
            fake_A_base, fake_A = netG_B2A(real_B)

            same_B_base, same_B = netG_A2B(real_B)
            same_A_base, same_A = netG_B2A(real_A)

            recovered_A_base, recovered_A = netG_B2A(fake_B)
            recovered_B_base, recovered_B = netG_A2B(fake_A)

            pred_fake_B = netD_B(fake_B)
            pred_fake_A = netD_A(fake_A)

            # GAN loss
            loss_GAN = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B)) + \
                       criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Identity loss
            loss_identity = criterion_identity(same_B, real_B) + \
                            criterion_identity(same_A, real_A)

            # Cycle loss
            loss_cycle = criterion_cycle(recovered_A, real_A) + \
                         criterion_cycle(recovered_B, real_B)

            # Base loss
            loss_base = criterion_cycle(recovered_A_base, real_A) + \
                        criterion_cycle(recovered_B_base, real_B)

            # Total loss
            idt_weight = 5.0
            cyc_weight = 10.0
            bas_weight = 1.0
            loss_G = loss_GAN + loss_identity * idt_weight + loss_cycle * cyc_weight  + loss_base * bas_weight
            
            # backward
            if step > 200:              # warmup for D
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            # save result
            if step % args.sample_freq == 0:
                with torch.no_grad():
                    show_img = torch.cat([
                        torch.cat([real_A, recovered_A, fake_B_base, fake_B], dim=3), 
                        torch.cat([real_B, recovered_B, fake_A_base, fake_A], dim=3)
                    ], dim=2).clamp(-1, 1)[[0], :, :, :]

                    show_img = make_grid(show_img, normalize=True, scale_each=True)
                    writer.add_image('image', show_img, step)

            ''' D update '''
            set_requires_grad([netD_A, netD_B], True)

            # forward
            fake_A = fake_A_buffer.push_and_pop(fake_A)

            pred_fake_A = netD_A(fake_A.detach())
            pred_real_A = netD_A(real_A)

            fake_B = fake_B_buffer.push_and_pop(fake_B)

            pred_fake_B = netD_B(fake_B.detach())
            pred_real_B = netD_B(real_B)

            loss_D_A = (criterion_GAN(pred_real_A, torch.ones_like(pred_real_A)) + \
                        criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))) * 0.5

            loss_D_B = (criterion_GAN(pred_real_B, torch.ones_like(pred_real_B)) + \
                        criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))) * 0.5

            # backward
            optimizer_D.zero_grad()
            loss_D_A.backward()
            loss_D_B.backward()
            optimizer_D.step()

            writer.add_scalar('loss_D', (loss_D_A + loss_D_B).item(), step)
            writer.add_scalar('loss_D_A', loss_D_A.item(), step)
            writer.add_scalar('loss_D_B', loss_D_B.item(), step)
            writer.add_scalar('loss_G', loss_G.item(), step)
            writer.add_scalar('loss_G_GAN', loss_GAN.item(), step)
            writer.add_scalar('loss_G_identity', loss_identity.item(), step)
            writer.add_scalar('loss_G_cycle', loss_cycle.item(), step)
            writer.add_scalar('loss_G_base', loss_base.item(), step)

            step += 1

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if (epoch + 1) % args.save_freq == 0:
            torch.save(netG_A2B.state_dict(), os.path.join(args.save_dir, args.task, 'MCT_G_A2B_{}.pth'.format(epoch + 1)))
            torch.save(netG_B2A.state_dict(), os.path.join(args.save_dir, args.task, 'MCT_G_B2A_{}.pth'.format(epoch + 1)))
            torch.save(netD_A.state_dict(), os.path.join(args.save_dir, args.task, 'MCT_D_A_{}.pth'.format(epoch + 1)))
            torch.save(netD_B.state_dict(), os.path.join(args.save_dir, args.task, 'MCT_D_B_{}.pth'.format(epoch + 1)))
