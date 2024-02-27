import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Generator, DiscriminatorWithAutoencoder, Encoder
from dataloader import get_dataloader

# Constants
img_size = 50  # Image size (50x50)
latent_dim = 128  # Latent dimension for Generator and Encoder

def display_images_and_mse(generator, discriminator, encoder, dataloader):
    with torch.no_grad():
        data_sample = next(iter(dataloader))
        if isinstance(data_sample, list):
            data_sample = data_sample[0]  # Extract the tensor from the list

        data_sample = data_sample.view(data_sample.size(0), -1)

        latent_encoder_sample = encoder(data_sample)
        images_from_encoder = generator(latent_encoder_sample).detach()

        latent_random_sample = torch.randn(data_sample.size(0), 128)
        images_from_gen = generator(latent_random_sample).detach()

        output_discriminator_encoder, _, _ = discriminator(images_from_encoder)
        output_discriminator_gen, _, _ = discriminator(images_from_gen)

        output_discriminator_encoder = output_discriminator_encoder.detach()
        output_discriminator_gen = output_discriminator_gen.detach()

        mse_images = nn.MSELoss()(images_from_encoder, images_from_gen).item()

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(images_from_encoder[i].squeeze(0), cmap='gray')
            axes[0, i].set_title(f"Enc: {output_discriminator_encoder[i].item():.2f}")
            axes[0, i].axis('off')

            axes[1, i].imshow(images_from_gen[i].squeeze(0), cmap='gray')
            axes[1, i].set_title(f"Gen: {output_discriminator_gen[i].item():.2f}")
            axes[1, i].axis('off')

        plt.suptitle(f'MSE Between Images: {mse_images:.4f}')
        plt.show()

# Training Loop with Checkpoints
def train(dataloader, generator, discriminator, encoder, num_epochs=300):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.99))

    for epoch in range(num_epochs):
        with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for data_batch in tepoch:
                # Extract data from the batch
                data_batch = data_batch[0]  # Assuming the batch is a list of tensors, get the first element

                # Flatten the images for input to the encoder
                data_batch_flattened = data_batch.view(data_batch.size(0), -1)

                # Generate latent vectors using the Encoder
                latent_vectors = encoder(data_batch_flattened)

                # Generate images using the Generator
                generated_images = generator(latent_vectors)

                # Reshape the generated images to match the original image size
                generated_images_reshaped = generated_images.view(-1, 1, img_size, img_size)

                # Compute reconstruction loss with respect to the original images
                recon_loss = criterion(generated_images_reshaped, data_batch)

                # Pass the original image through the Encoder and Generator
                latent_representation = encoder(data_batch_flattened)
                generated_img = generator(latent_representation)

                # Get feature vectors from the Discriminator
                _, original_features, _ = discriminator(data_batch)
                _, generated_features, _ = discriminator(generated_img)

                # Calculate the MSE loss between the original and generated feature vectors
                mse_loss = criterion(original_features, generated_features)

                # Combine the reconstruction loss and the new MSE loss
                loss = recon_loss + mse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

            # Display images and MSE at 50%, 75%, and 100% of epochs
            '''if epoch == num_epochs // 2 - 1 or epoch == int(num_epochs * 0.75) - 1 or epoch == num_epochs - 1:
                display_images_and_mse(generator, discriminator, encoder, dataloader)'''




if __name__ == "__main__":
    dataloader = get_dataloader('data.npy')
    train(dataloader)
