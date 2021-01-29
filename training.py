"""
Training of WGAN-GP for Pokemon Dataset
"""

import gc
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialise_weights
from utils import gradient_penalty

if __name__ == '__main__':
    # Initialise GPU if available and clear cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Hyperparameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 100
    NUM_EPOCHS = 500
    FEATURES_DISC = 64
    FEATURES_GEN = 64
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    # Define image transforms and augmentations
    transforms = transforms.Compose([  # transforms.ToPILImage(),
                                     transforms.Resize(IMAGE_SIZE),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         [0.5 for _ in range(CHANNELS_IMG)],
                                         [0.5 for _ in range(CHANNELS_IMG)])
                                     ])

    # Load dataset and model
    dataset = datasets.ImageFolder(root='pokemon_dataset_augmentation/', transform=transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialise_weights(gen)
    initialise_weights(critic)

    # Define optimisers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # Setup torchvision
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    writer_real = SummaryWriter('logs/real')
    writer_fake = SummaryWriter('logs/fake')
    step = 0

    gen.train()
    critic.train()

    # Network training
    for epoch in range(NUM_EPOCHS):
        # Clear CPU and GPU cache each epoch
        gc.collect()
        torch.cuda.empty_cache()

        print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))

        # Save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(gen, 'gen_epoch{}'.format(epoch+1))

        # For each batch_idx...
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)

            # Train the critic (5 to 1 ratio against Generator)
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic=critic, real=real, fake=fake, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + (LAMBDA_GP * gp)
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: min -E[critic(gen_fake)]
            output = critic(fake).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Export to torchvision every 100 batch_idx iterations
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # Take out up to 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)

                    writer_real.add_image('Real', img_grid_real, global_step=step)
                    writer_fake.add_image('Fake', img_grid_fake, global_step=step)

                step += 1
