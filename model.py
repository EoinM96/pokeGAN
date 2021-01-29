"""
Discriminator and Generator Implementation from DCGAN Paper
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    # Model implementation for GAN using blocks as defined below
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        # Input: N * channels_img * 64 * 64
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img,
                      features_d,
                      kernel_size=4,
                      stride=2,
                      padding=1
                      ),
            # Currently size 32 * 32
            nn.LeakyReLU(0.2),

            self._block(features_d, features_d*2, 4, 2, 1),  # 16 * 16
            self._block(features_d*2, features_d*4, 4, 2, 1),  # 8 * 8
            self._block(features_d*4, features_d*8, 4, 2, 1),  # 4 * 4
            self._block(features_d*8, features_d*16, 4, 2, 1),  # 4 * 4

            nn.Conv2d(features_d*16, 1, kernel_size=4, stride=2, padding=0),  # 1 * 1
        )

    # Block to contain layers within the overall neural network, used to build Critic model
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False
                      ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    # Forward propagation
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    # Model implementation for GAN using blocks as defined below
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N * z_dim * 1 * 1
            self._block(z_dim, features_g*16, 4, 1, 0),  # N * features_g*16 * 4 * 4
            self._block(features_g*16, features_g*8, 4, 2, 1),  # 8 * 8
            self._block(features_g*8, features_g*4, 4, 2, 1),  # 16 * 16
            self._block(features_g*4, features_g*2, 4, 2, 1),  # 32 * 32
            nn.ConvTranspose2d(features_g*2,
                               channels_img,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()  # [-1, 1] Images normalised to be in this range, models to output the same
        )

    # Block to contain layers within the overall neural network, used to build Generator model
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # Forward propagation
    def forward(self, x):
        return self.gen(x)

# Function for weight initialization
def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Following is a test to ensure model implementation works as expected, please ignore
if '__name__' == '__main__':
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(in_channels, 8)
    initialise_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialise_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)

