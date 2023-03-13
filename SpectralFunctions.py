# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:10:37 2020

@author: Steff
This module contains spectral transformations include AI, FFT etc.
"""

import torch
import numpy as np
import os
import torchaudio
import torch.nn.functional as F
from torch import nn
import numbers
import math

################################################################
class SpectralLoss(torch.nn.Module):

    f_cache = "spectralloss.{}.cache"

    ############################################################
    def __init__(self, rows=128, cols=128, eps=1e-8, device=None, cache=False):
        super(SpectralLoss, self).__init__()
        self.img_size = rows
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((rows, cols_onesided)) - np.array([[[shift_rows]], [[0]]])
        r = np.sqrt(r[0, :, :] ** 2 + r[1, :, :] ** 2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r, axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(r_max + 1, -1, -1).to(torch.float)
        radius_to_slice = torch.arange(r_max + 1).view(-1, 1, 1)
        # generate mask for each radius
        mask = torch.where(
            r == radius_to_slice,
            torch.tensor(1, dtype=torch.float),
            torch.tensor(0, dtype=torch.float),
        )
        # how man entries for each radius?
        mask_n = torch.sum(mask, axis=(1, 2))
        mask = mask.unsqueeze(0)  # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1 / mask_n.to(torch.float)).unsqueeze(0)
        self.criterion_l1 = torch.nn.L1Loss(reduction="sum")
        self.r_max = r_max
        self.vector_length = r_max + 1

        self.register_buffer("mask", mask)
        self.register_buffer("mask_n", mask_n)

        if cache and os.path.isfile(SpectralLoss.f_cache.format(self.img_size)):
            self._load_cache()
        else:
            self.is_fitted = False
            self.register_buffer("mean", None)

        if device is not None:
            self.to(device)
        self.device = device

    ############################################################
    def _save_cache(self):
        torch.save(self.mean, SpectralLoss.f_cache.format(self.img_size))
        self.is_fitted = True

    ############################################################
    def _load_cache(self):
        mean = torch.load(
            SpectralLoss.f_cache.format(self.img_size), map_location=self.mask.device
        )
        self.register_buffer("mean", mean)
        self.is_fitted = True

    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################

    ############################################################
    def fft(self, data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data = (
                0.299 * data[:, 0, :, :]
                + 0.587 * data[:, 1, :, :]
                + 0.114 * data[:, 2, :, :]
            )

        fft = torch.rfft(data, signal_ndim=2, onesided=True)
        # fft = torch.fft.rfft(data)
        # abs of complex
        fft_abs = torch.sum(fft ** 2, dim=3)
        fft_abs = fft_abs + self.eps
        fft_abs = 20 * torch.log(fft_abs)
        return fft_abs

    ############################################################
    def magphase(self, data, is_gray, is_log, is_mag):
        """Assumes first dimension to be batch size."""
        # fft = torch.fft.fft(data)
        # print(fft)
        if is_gray:
            data = (
                0.299 * data[:, 0, :, :]
                + 0.587 * data[:, 1, :, :]
                + 0.114 * data[:, 2, :, :]
            )
            data = data.unsqueeze(1)
        fft = torch.rfft(data, 2, onesided=False)
        if is_mag:
            fft = torch.cat(torchaudio.functional.magphase(fft), dim=1)

        if is_log:
            fft = torch.sum(fft ** 2, dim=-1)
            fft = fft + self.eps
            fft = torch.log(fft)
        return fft

    ############################################################

    ############################################################
    def spectral_vector(self, data):
        """Assumes first dimension to be batch size."""
        fft = (
            self.fft(data).unsqueeze(1).expand(-1, self.vector_length, -1, -1)
        )  # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2, 3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1, 1)
        profile = profile / profile.max(1)[0].view(-1, 1)

        return profile

    ############################################################
    def avg_spectrum(self, data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data = (
                0.299 * data[:, 0, :, :]
                + 0.587 * data[:, 1, :, :]
                + 0.114 * data[:, 2, :, :]
            )

        fft = torch.rfft(data, signal_ndim=2, onesided=False)
        # fft = torch.fft.rfft(data)
        # abs of complex
        fft_abs = torch.sum(fft ** 2, dim=3)
        fft_abs = fft_abs + self.eps
        fft_abs = 20 * torch.log(fft_abs)
        fft_mean = fft_abs.mean(0).numpy()
        fft_mean = np.fft.fftshift(fft_mean)
        return fft_mean

    ############################################################
    def avg_profile(self, data):
        profile = self.spectral_vector(data)
        return profile.mean(0)

    ############################################################
    def avg_profile_batched(self, data, batch_size=1024, dtype=torch.double):
        i = 0
        v_total = torch.zeros(1, self.vector_length, dtype=dtype, device=self.device)
        while i < len(data):
            i_next = i + batch_size
            v = torch.sum(self.spectral_vector(data[i:i_next]), dim=0)
            v_total += v
            i = i_next
        return v_total / len(data)

    ############################################################
    def avg_profile_and_sd(self, data, batch_size=1024):
        if len(data) < batch_size:
            profile = self.spectral_vector(data)
            return profile.mean(0), profile.std(0)
        else:
            i = 0
            v_total = []
            while i < len(data):
                i_next = i + batch_size
                v_total.append(self.spectral_vector(data[i:i_next]))
                i = i_next

            v = torch.cat(v_total)
            return v.mean(0), v.std(0)

    ############################################################
    def fit_batch(self, batch):
        if not hasattr(self, "batches"):
            self.batches = []
            self.batches_size = 0

        v = np.sum(self.spectral_vector(batch).detach().cpu().numpy(), axis=0).reshape(
            (1, -1)
        )

        self.batches_size += len(batch)
        self.batches.append(v)

    ############################################################
    def complete_fit(self):
        total = np.sum(np.concatenate(self.batches), axis=0)
        mean = torch.from_numpy(total / self.batches_size)
        if self.device is not None:
            mean = mean.to(self.device)
        del self.batches
        del self.batches_size

        return mean

    ############################################################
    def complete_fit_real(self, cache=False):
        self.mean = self.complete_fit()

        if cache:
            self._save_cache()

    ############################################################
    def fit(self, data, batch_size=1024, cache=False):
        self.mean = self.avg_profile_batched(data, batch_size)
        self.register_buffer("mean", self.mean)

        if cache:
            self._save_cache()

    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################

    ############################################################
    def calc_from_profile(self, profile):
        batch_size = profile.shape[0]
        target = self.mean.expand(batch_size, -1)

        return self.criterion_l1(profile, target)


class SRMfilter(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMfilter, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        """
        x: imgs (Batch, H, W, 3)
        """
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]
        # filter2：KV
        filter2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]
        # filter3：hor 2rd
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        filter1 = np.asarray(filter1, dtype=float) / 4.0
        filter2 = np.asarray(filter2, dtype=float) / 12.0
        filter3 = np.asarray(filter3, dtype=float) / 2.0
        # statck the filters
        filters = [
            [filter1],  # , filter1, filter1],
            [filter2],  # , filter2, filter2],
            [filter3],
        ]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.01, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


# class Butter(Function):
#     @staticmethod
#     def forward(ctx, input, filter, bias):
#         # detach so we can cast to NumPy
#         input, filter, bias = input.detach(), filter.detach(), bias.detach()
#         result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
#         result += bias.numpy()
#         ctx.save_for_backward(input, filter, bias)
#         return torch.as_tensor(result, dtype=input.dtype)

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_output = grad_output.detach()
#         input, filter, bias = ctx.saved_tensors
#         grad_output = grad_output.numpy()
#         grad_bias = np.sum(grad_output, keepdims=True)
#         grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
#         # the previous line can be expressed equivalently as:
#         # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
#         grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
#         return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


# class ScipyConv2d(Module):
#     def __init__(self, filter_width, filter_height):
#         super(ScipyConv2d, self).__init__()
#         self.filter = Parameter(torch.randn(filter_width, filter_height))
#         self.bias = Parameter(torch.randn(1, 1))

#     def forward(self, input):
#         return ScipyConv2dFunction.apply(input, self.filter, self.bias)
