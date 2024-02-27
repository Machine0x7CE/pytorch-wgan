# main.py

import torch
import torch.optim as optim
from tqdm import tqdm
from model import Generator, DiscriminatorWithAutoencoder
from dataloader import get_dataloader
from trainergan import train_loop
# Hyperparameters
img_size = 50  # Image size
latent_dim = 128  # Latent dimension for the generator
batch_size = 64
epochs = 350
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999
n_critic = 5  # Number of discriminator updates per generator update
lambda_gp = 10  # Gradient penalty lambda hyperparameter

# DataLoader
dataset = get_dataloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device)
discriminator = DiscriminatorWithAutoencoder().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

# Call train loop function with all necessary arguments
train_loop(generator, discriminator, optimizer_G, optimizer_D, dataset, device, epochs, n_critic, lambda_gp, latent_dim)

# Save trained models
torch.save(generator.state_dict(), 'generatorwithEnc.pth')
torch.save(discriminator.state_dict(), 'discriminatorwithEnc.pth')