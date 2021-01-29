"""
Gradient Penalty implementation for WGAN-GP
"""

import torch
import torch.nn


# Define the gradient penalty for Wasserstein GAN
# Implementation as is in paper
def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape

    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (real * epsilon) + (fake * (1 - epsilon))

    # Calculate the critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
