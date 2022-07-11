import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cyclegan import GeneratorResNet
from utils import set_requires_grad


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.l = 8
		self.dims = 9 * self.l

		self.ts = 256
		self.m_shift = 1

		basenet = GeneratorResNet(3, 3)
		base_layers = list(basenet.model.children())

		self.backbone = nn.Sequential(*base_layers[:-3])
		self.outlayer_base = nn.Sequential(*base_layers[-3:-1])

		self.outlayer = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='replicate'),
			nn.ReLU(True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='replicate'),
			nn.ReLU(True),
			nn.Conv2d(64, self.dims, kernel_size=3, padding=1, padding_mode='replicate')
		)
		
	def load_pretrain(self, pretrained_model):
		basenet = GeneratorResNet(3, 3)
		basenet.load_state_dict(torch.load(pretrained_model))
		base_layers = list(basenet.model.children())

		self.backbone = nn.Sequential(*base_layers[:-3])
		self.outlayer_base = nn.Sequential(*base_layers[-3:-1])

		# set_requires_grad([self.backbone], False)

	def get_coord(self, x):
		B, _, H, W = x.size()

		# smooth grid
		coordh, coordw = torch.meshgrid([torch.linspace(-1,1,H), torch.linspace(-1,1,W)])
		coordh = coordh.unsqueeze(0).unsqueeze(1).repeat(B,1,1,1).cuda()
		coordw = coordw.unsqueeze(0).unsqueeze(1).repeat(B,1,1,1).cuda()

		if self.training:
			coordh = coordh + (torch.rand_like(coordh) - 0.5) * self.m_shift / H
			coordw = coordw + (torch.rand_like(coordw) - 0.5) * self.m_shift / W

		return coordw.detach(), coordh.detach()

	def mapping(self, x, param):
		# grid: x, y, z -> w, h, d  ~[-1 ,1]
		coordw, coordh = self.get_coord(x)
		curve = torch.stack(torch.chunk(param, 3, dim=1), dim=1)

		curve_list = list(torch.chunk(curve, 3, dim=2))
		x_list = list(torch.chunk(x.detach(), 3, dim=1))
		grid_list = [torch.stack([coordw, coordh, x_i], dim=4) for x_i in x_list]
		out = sum([F.grid_sample(curve_i, grid_i, 'bilinear', 'border', True) \
					for curve_i, grid_i in zip(curve_list, grid_list)]).squeeze(2)

		return out

	def forward(self, x):
		x_d = F.interpolate(x, (self.ts, self.ts))
		backbone_feature = self.backbone(x_d)
		base_out = self.outlayer_base(backbone_feature)
		base_param = self.outlayer(backbone_feature)
		param = base_param + base_out.repeat_interleave(self.l * 3, dim=1) / 3
		out = self.mapping(x, param)

		if self.training:
			return torch.tanh(base_out), torch.tanh(out)
		else:
			return torch.tanh(out)
