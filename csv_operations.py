import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Generator, DiscriminatorWithAutoencoder, Encoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load pre-trained models
gen = Generator()
dis = DiscriminatorWithAutoencoder()
gen.load_state_dict(torch.load('generatorwithEnc.pth'))
dis.load_state_dict(torch.load('discriminatorwithEnc.pth'))

encoder = Encoder()
encoder.load_state_dict(torch.load('trained_encoder.pth'))

gen.eval()
dis.eval()
encoder.eval()
criterion = nn.MSELoss()
mse_losses = []
image_losses = []
feature_vectors = []
indices = []
total_losses = []

# Load data
data = np.load('data.npy')
data = data.reshape(8565, 50, 50)

for idx, original_img in enumerate(tqdm(data, desc="Processing images")):
    original_img = torch.Tensor(original_img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Pass the original image through the Encoder and Generator
    latent_representation = encoder(original_img)
    generated_img = gen(latent_representation)

    # Get feature vectors from the Discriminator
    _, original_features, _ = dis(original_img)
    _, generated_features, _ = dis(generated_img)

    # Calculate the feature loss between the original and generated feature vectors
    feature_loss = criterion(original_features, generated_features).item()
    mse_losses.append(feature_loss)

    # Calculate the loss between the original and generated images
    image_loss = criterion(original_img, generated_img).item()
    image_losses.append(image_loss)

    feature_vectors.append(original_features.detach().numpy())
    indices.append(idx)

    # Calculate total loss
    total_loss = feature_loss + image_loss
    total_losses.append(total_loss)

# Convert list of numpy arrays to a single numpy array
feature_vectors = np.concatenate(feature_vectors, axis=0)

# Concatenate feature vectors into a single column
feature_vectors_str = [",".join(map(str, vec)) for vec in feature_vectors]

# Create a DataFrame with the collected information
data_dict = {'Index': indices, 'Feature Loss': mse_losses, 'Image Loss': image_losses, 'Total Loss': total_losses, 'Feature Vectors': feature_vectors_str}
df = pd.DataFrame(data_dict)

# Save the DataFrame as a CSV file
df.to_csv('updated-output-img-final.csv', index=False)
