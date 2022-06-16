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

def train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
	loop = tqdm(loader, leave=True)

	for idx, (x, y) in enumerate(loop):
		x = x.to(config.DEVICE)
		y = y.to(config.DEVICE)

		#Train Discriminator
		with torch.cuda.amp.autocast():
			fake_y = gen_Y(x)
			D_Y_real = disc_Y(y)
			D_Y_fake = disc_Y(fake_y.detach())
			D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
			D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
			D_Y_loss = D_Y_real_loss + D_Y_fake_loss

			fake_x = gen_X(y)
			D_X_real = disc_X(x)
			D_X_fake = disc_X(fake_x.detach())
			D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
			D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))
			D_X_loss = D_X_real_loss + D_X_fake_loss

			D_loss = (D_X_loss + D_Y_loss)/2
		opt_disc.zero_grad()
		d_scaler.scale(D_loss).backward()
		d_scaler.step(opt_disc)
		d_scaler.update()

		#Train Generators

		with torch.cuda.amp.autocast():
			#adversarial loss
			D_Y_fake = disc_Y(fake_y)
			D_X_fake = disc_X(fake_x)
			loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))
			loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))

			#cycle loss
			cycle_X = gen_X(fake_y)
			cycle_Y = gen_Y(fake_x)
			cycle_X_loss = l1(cycle_X, x)
			cycle_Y_loss = l1(cycle_Y, y)

			#identity loss
			identity_X = gen_X(x)
			identity_Y = gen_Y(y)
			identity_X_loss = l1(x, identity_X)
			identity_Y_loss = l1(y, identity_Y)
			G_loss = (
				loss_G_X + loss_G_Y + cycle_X_loss * config.LAMBDA_CYCLE + cycle_Y_loss * config.LAMBDA_CYCLE + identity_X_loss * config.LAMBDA_IDENTITY + identity_Y_loss * config.LAMBDA_IDENTITY
			)
		opt_gen.zero_grad()
		g_scaler.scale(G_loss).backward()
		g_scaler.step(opt_gen)
		g_scaler.update()


def main():

	disc_X = Discriminator().to(config.DEVICE)
	disc_Y = Discriminator().to(config.DEVICE)
	gen_X = Generator().to(config.DEVICE)
	gen_Y = Generator().to(config.DEVICE)
	opt_disc = optim.Adam(
		list(disc_X.parameters()) + list(disc_Y.parameters()),
		lr=config.LEARNING_RATE,
		betas=(0.5, 0.999)
	)
	opt_gen = optim.Adam(
		list(gen_X.parameters()) + list(disc_Y.parameters()),
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

	dataset = XYDataset(21, 1)
	loader = DataLoader(
		dataset,
		batch_size = config.BATCH_SIZE,
		shuffle=True,
		pin_memory=True
	)
	g_scaler = torch.cuda.amp.GradScaler()
	d_scaler = torch.cuda.amp.GradScaler()

	for epoch in range(config.NUM_EPOCHS):
		train_fn(disc_X, disc_Y, gen_X, gen_Y, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

		if config.SAVE_MODEL:
			save_checkpoint(gen_X, opt_gen, filename=config.CHECKPOINT_GEN_X)
			save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
			save_checkpoint(disc_X, opt_disc, filename=config.CHECKPOINT_CRITIC_X)
			save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITIC_Y)
if __name__ == '__main__':
	main()
