from torch import nn
import torch
from SpectralFunctions import SpectralLoss, SRMfilter
import numpy as np
import torch.nn.functional as F

"""
The code is based on: https://github.com/steffen-jung/SpectralGAN
"""


class Identity(nn.Module):
    """
    Identity mapping
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _get_norm_layer_2d(norm):
    """
    Get 2D normalization layer
    """
    if norm == "none":
        return Identity
    elif norm == "batch_norm":
        return nn.BatchNorm2d
    elif norm == "instance_norm":
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == "layer_norm":
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class ImageD(nn.Module):
    """Discriminator for image-based real/fake classification
    Return 1D probability vector instead of 2D one-hot encoding vector
    """

    ###########################################################################
    def __init__(
        self, img_size: int = 128, ndf: int = 128, nc: int = 3, norm="batch_norm"
    ):

        super(ImageD, self).__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False or Norm == Identity,
                ),
                Norm(out_dim),
                nn.LeakyReLU(0.2),
            )

        layers = []

        layers.append(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        blocks = int(np.log2(img_size)) - 3  # blocks = 4 if img_size = 128
        for i in range(blocks):
            f_in = ndf * (2 ** i)
            f_in = min(f_in, 512)
            f_out = ndf * (2 ** (i + 1))
            f_out = min(f_out, 512)
            layers.append(
                conv_norm_lrelu(f_in, f_out, kernel_size=4, stride=2, padding=1)
            )

        f_in = min(ndf * (2 ** blocks), 512)
        layers.append(nn.Conv2d(f_in, 1, kernel_size=4, stride=1, padding=0))

        self._forward = nn.Sequential(*layers)
        self.apply(weights_init)

    ###########################################################################
    def forward(self, x):
        y = self._forward(x)

        return y


class SpectrumD(nn.Module):
    """Discriminator for spectrum-based real/fake classification
    Return 1D probability vector instead of 2D one-hot encoding vector
    """

    ###########################################################################
    def __init__(
        self,
        img_size=128,
        ndf: int = 128,
        is_gray: bool = True,
        is_mag: bool = False,
        norm="batch_norm",
    ):

        super(SpectrumD, self).__init__()

        if is_gray:
            nc = 1
        else:
            nc = 3
        if is_mag:
            nc *= 2

        self.is_gray = is_gray
        self.is_mag = is_mag
        self.spectral_transform = SpectralLoss(rows=img_size, cols=img_size)
        Norm = _get_norm_layer_2d(norm)

        # self.register_buffer("spectral_transform", SpectralLoss(rows=img_size, cols=img_size))
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False or Norm == Identity,
                ),
                Norm(out_dim),
                nn.LeakyReLU(0.2),
            )

        layers = []

        layers.append(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        blocks = int(np.log2(img_size)) - 3  # blocks = 4 if img_size = 128
        for i in range(blocks):
            f_in = ndf * (2 ** i)
            f_in = min(f_in, 512)
            f_out = ndf * (2 ** (i + 1))
            f_out = min(f_out, 512)
            layers.append(
                conv_norm_lrelu(f_in, f_out, kernel_size=4, stride=2, padding=1)
            )

        f_in = min(ndf * (2 ** blocks), 512)
        layers.append(nn.Conv2d(f_in, 1, kernel_size=4, stride=1, padding=0))

        self._forward = nn.Sequential(*layers)
        self.apply(weights_init)

    ###########################################################################
    def forward(self, x):
        x_magphase = self.spectral_transform.magphase(
            x, is_gray=self.is_gray, is_log=True, is_mag=self.is_mag
        )
        y = self._forward(x_magphase)
        return y


class SpectrumDistributionD(nn.Module):
    """Discriminator for spectrum distribution-based real/fake classification
    Return 1D probability vector instead of 2D one-hot encoding vector
    """

    ###########################################################################
    #! nonlinear -> linear
    def __init__(self, img_size=128, spectral="linear"):
        super(SpectrumDistributionD, self).__init__()

        self.img_size = img_size
        self.spectral = spectral

        self.spectral_transform = SpectralLoss(rows=img_size, cols=img_size)
        self.frequency_thre = int(0.00 * self.spectral_transform.vector_length)
        # print(self.frequency_thre)
        self._add_spectral_layers(spectral)

    ###########################################################################
    def _add_spectral_layers(self, spectral):
        if spectral == "none":
            self.forward = self.forward_none

        else:
            layers = nn.Sequential()

            # if "unnormalize" in spectral:
            #     layers.add_module("Unnormalize", Unnormalize())

            if "dropout" in spectral:
                layers.add_module("Dropout", nn.Dropout())

            if "linear" in spectral and not "nonlinear" in spectral:
                layers.add_module(
                    "LinearSpectral",
                    nn.Linear(
                        self.spectral_transform.vector_length - self.frequency_thre, 1
                    ),
                )

            if "nonlinear" in spectral:
                layers.add_module(
                    "Linear1Spectral",
                    nn.Linear(
                        self.spectral_transform.vector_length - self.frequency_thre,
                        self.spectral_transform.vector_length - self.frequency_thre,
                    ),
                )
                layers.add_module("ReLU1Spectral", nn.LeakyReLU(0.2))
                layers.add_module(
                    "Linear2Spectral",
                    nn.Linear(
                        self.spectral_transform.vector_length - self.frequency_thre, 1
                    ),
                )

            self._forward_spectral = layers
            self.apply(weights_init)

    ###########################################################################
    def forward(self, x):
        x_profiles = self.spectral_transform.spectral_vector(x)
        # print(x_profiles.shape, x_profiles[:, frequency_thre:].shape)

        #! adjsut x_profile length
        y = self._forward_spectral(x_profiles[:, self.frequency_thre :])

        return y

    ###########################################################################
    def forward_none(self, x):
        return torch.tensor(0.0)


class FingerprintD(nn.Module):
    """This discriminator is used for classifiy the spectrum distributions of real/fake images
    The code is based on: https://github.com/steffen-jung/SpectralGAN
    """

    ###########################################################################
    def __init__(
        self, img_size: int = 128, ndf: int = 128, nc: int = 3, norm="batch_norm"
    ):

        super(FingerprintD, self).__init__()

        self.SRM = SRMfilter(inc=3)

        Norm = _get_norm_layer_2d(norm)

        # self.register_buffer("spectral_transform", SpectralLoss(rows=img_size, cols=img_size))
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False or Norm == Identity,
                ),
                Norm(out_dim),
                nn.LeakyReLU(0.2),
            )

        layers = []

        layers.append(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        blocks = int(np.log2(img_size)) - 3  # blocks = 4 if img_size = 128
        for i in range(blocks):
            f_in = ndf * (2 ** i)
            f_in = min(f_in, 512)
            f_out = ndf * (2 ** (i + 1))
            f_out = min(f_out, 512)
            layers.append(
                conv_norm_lrelu(f_in, f_out, kernel_size=4, stride=2, padding=1)
            )

        f_in = min(ndf * (2 ** blocks), 512)
        layers.append(nn.Conv2d(f_in, 1, kernel_size=4, stride=1, padding=0))

        self._forward = nn.Sequential(*layers)
        self.apply(weights_init)

    ###########################################################################
    def forward(self, x):
        x_fingerprint = self.SRM(x)
        y = self._forward(x_fingerprint)
        return y
