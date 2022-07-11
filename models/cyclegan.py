import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
			nn.ReLU(inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
		)

	def forward(self, x):
		return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, channels=3, outchannels=3, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, outchannels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        _, _, H, W = x.size()

        padding_h = math.ceil(H / 8) * 8 - H
        padding_w = math.ceil(W / 8) * 8 - W

        x_pad = F.pad(x, (0, padding_w, 0, padding_h), mode='reflect')

        return self.model(x_pad)[:, :, :H ,:W]


class PatchDiscriminator(nn.Module):
	def __init__(self):
		super(PatchDiscriminator, self).__init__()
		
		def discriminator_block(in_filters, out_filters, normalize=True):
			"""Returns downsampling layers of each discriminator block"""
			layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
			if normalize:
				layers.append(nn.InstanceNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*discriminator_block(3, 64, normalize=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 1, 4, padding=1)
		)

	def forward(self, input):
		return self.model(input)
		