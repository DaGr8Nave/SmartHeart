import torch
import sys
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms
from dataset import XYDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm 
from discriminator_model import Discriminator
from generator_model import Generator

def train_fc(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
	loop = tqdm(loader, leave=True)

def main():
	disc_X = Discriminator().to(config.DEVICE)
	disc_Y = Discriminator().to(config.DEVICE)
	gen_X = Generator().to(config.DEVICE)
	gen_Y = Generator().to(config.DEVICE)
	opt_disc = optim.Adam(
		list(disc_H.parameters()) + list(disc_Z.parameters()),
		lr=config.LEARNING_RATE,
		betas=(0.5, 0.999)
	)
	opt_gen = optim.Adam(
		list(gen_Z.parameters()) + list(disc_H.parameters()),
		lr=config.LEARNING_RATE,
		betas=(0.5, 0.999)
	)
	L1 = nn.L1Loss()
	mse = nn.MSELoss()
	if config.LOAD_MODEL:
		load_checkpoint(
			config.CHECKPOINT_GEN_X, gen_X, opt_gen, config.LEARNING_RATE,
		)
		load_checkpoint(
			config.CHECKPOINT_GEN_Y, gen_Y, opt_gen, config.LEARNING_RATE,
		)
		load_checkpoint(
			config.CHECKPOINT_disc_X, disc_X, opt_gen, config.LEARNING_RATE,
		)
		load_checkpoint(
			config.CHECKPOINT_disc_Y, disc_Y, opt_gen, config.LEARNING_RATE,
		)
	g_scaler = torch.cuda.amp.GradScaler()
	d_scaler = torch.cuda.amp.GradScaler()
	dataset = XYDataset(21, 1)
	loader = DataLoader(
		)
	for epoch in range(config.NUM_EPOCHS):
		train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

		if config.SAVE_MODEL:
			save_checkpoint(gen_X, opt_gen, filename=config.CHECKPOINT_GEN_X)
			save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
			save_checkpoint(disc_X, opt_disc, filename=config.CHECKPOINT_CRITIC_X)
			save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITIC_Y)
if __name__ == '__main__':
	main()