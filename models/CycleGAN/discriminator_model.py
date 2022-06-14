import torch
import torch.nn as nn

class Block(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode='reflect'),
			nn.InstanceNorm1d(out_channels),
			nn.LeakyReLU(0.2),
		)
	def forward(self, x):
		return self.conv(x)

class Discriminator(nn.Module):
	def __init__(self, in_channels=12, features=[64,128,256,512]):
		super().__init__()
		self.initial = nn.Sequential(
			nn.Conv1d(
				in_channels,
				features[0],
				kernel_size=4,
				stride=2,
				padding=1,
				padding_mode='reflect',
			),
			nn.LeakyReLU(0.2),
		)

		layers = []
		in_channels = features[0]
		for feature in features[1:]:
			stride = 2
			if feature == features[-1]:
				stride = 1
			layers.append(Block(in_channels, feature, stride))
			in_channels = feature
		layers.append(nn.Conv1d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode = 'reflect'))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		x = self.initial(x)
		return torch.sigmoid(self.model(x))