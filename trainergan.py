import torch
from tqdm import tqdm

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    device = real_samples.device
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)[0]  # Only consider the first returned value
    fake = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_loop(generator, discriminator, optimizer_G, optimizer_D, dataset, device, epochs, n_critic, lambda_gp,latent_dim):
    # Training loop with visualization using tqdm
    generated_samples = []

    for epoch in range(epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        epoch_samples = []

        for i, (real_imgs,) in enumerate(tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_imgs = real_imgs.to(device)
            batch_size_actual = real_imgs.size(0)

            # Train Discriminator
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size_actual, latent_dim).to(device)
                gen_imgs = generator(z)
                d_real = discriminator(real_imgs)[0]
                d_fake = discriminator(gen_imgs.detach())[0]
                gradient_penalty = lambda_gp * compute_gradient_penalty(discriminator, real_imgs, gen_imgs)
                d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gradient_penalty
                d_loss.backward()
                optimizer_D.step()
                d_loss_total += d_loss.item()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            gen_imgs = generator(z)
            g_loss = -torch.mean(discriminator(gen_imgs)[0])
            g_loss.backward()
            optimizer_G.step()
            g_loss_total += g_loss.item()
            epoch_samples.append(gen_imgs.detach().cpu().numpy())

        # Save generated samples for this epoch
        generated_samples.append(epoch_samples)