import torch
from model import Generator, DiscriminatorWithAutoencoder, Encoder
from trainer import display_images_and_mse, train
from dataloader import get_dataloader

def main():
    # Load data
    dataloader = get_dataloader()

    # Initialize models
    generator = Generator()
    discriminator = DiscriminatorWithAutoencoder()
    encoder = Encoder()

    generator.load_state_dict(torch.load('generatorwithEnc.pth'))
    discriminator.load_state_dict(torch.load('discriminatorwithEnc.pth'))

    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    # Train and display
    train(dataloader, generator, discriminator, encoder)
    #display_images_and_mse(generator, discriminator, encoder, dataloader)

    # Save trained encoder
    torch.save(encoder.state_dict(), 'trained_encoder.pth')

if __name__ == "__main__":
    main()
