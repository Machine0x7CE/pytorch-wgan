import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_size=50):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 50, 50)
        return img

class DiscriminatorWithAutoencoder(nn.Module):
    def __init__(self, img_size=50, latent_dim=128):
        super(DiscriminatorWithAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_size * img_size)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, img):
        encoded = self.encoder(img.view(img.size(0), -1))
        decoded = self.decoder(encoded)
        validity = self.discriminator(encoded)
        return validity, encoded, decoded

class Encoder(nn.Module):
    def __init__(self, img_size=50, latent_dim=128):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, img):
        latent = self.model(img.view(img.size(0), -1))
        return latent
