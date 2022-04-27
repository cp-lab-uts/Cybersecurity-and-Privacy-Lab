# This script is borrowed from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
# which is also used by Durall et al. (https://ieeexplore.ieee.org/document/9157579)

from cv2 import GaussianBlur
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from SpectralFunctions import SpectralLoss, GaussianSmoothing, GaussianNoise
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
import lpips


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_gan_losses_fn():
    bce = torch.nn.BCEWithLogitsLoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(r_logit, torch.ones_like(r_logit))
        f_loss = bce(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
        f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = torch.nn.MSELoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(r_logit, torch.ones_like(r_logit))
        f_loss = mse(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = -r_logit.mean()
        f_loss = f_logit.mean()
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = -f_logit.mean()
        return f_loss

    return d_loss_fn, g_loss_fn


def get_contr_adv_loss():
    bce = torch.nn.BCEWithLogitsLoss()
    crt_l1 = torch.nn.TripletMarginLoss(margin=1, p=2)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(r_logit, torch.ones_like(r_logit))
        f_loss = bce(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(r_logit, f_logit, s_logit):
        s_loss = crt_l1(s_logit, r_logit, f_logit)
        return s_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == "gan":
        return get_gan_losses_fn()
    elif mode == "hinge_v1":
        return get_hinge_v1_losses_fn()
    elif mode == "hinge_v2":
        return get_hinge_v2_losses_fn()
    elif mode == "lsgan":
        return get_lsgan_losses_fn()
    elif mode == "wgan":
        return get_wgan_losses_fn()
    elif mode == "contr_adv":
        return get_contr_adv_loss()


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 10.0 * (1.0 - super(SSIM_Loss, self).forward(img1, img2))


def sim_loss(ts1: torch.Tensor, ts2: torch.Tensor, basemode, highmode) -> torch.Tensor:
    """
    @Desc: L1 loss for 2 image tensors;
    @Params:
        ts1: Image tensor 1;
        ts2: Image tensor 2;
    @Return:
        loss_sim: Similarity loss;
    """
    if basemode == "l1":
        criterion = nn.L1Loss()
    elif basemode == "ssim":
        criterion = SSIM_Loss(
            data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True
        )
    elif basemode == 'perceptual':
        criterion = lpips.LPIPS(net='vgg').to(device)

    if highmode == 'GaussianNoise':
        noising = GaussianNoise().to(device)
        ts1 = noising(ts1)
        ts2 = noising(ts2)
    elif highmode == 'GaussianFilter':
        # filter = GaussianSmoothing(3, 3, 0.5).to(device)
        # pad = nn.ReflectionPad2d(1)
        # ts1 = filter(pad(ts1))
        # ts2 = filter(pad(ts2))
        filter = GaussianBlur(3, 1).to(device)
        ts1 = filter(ts1)
        ts2 = filter(ts2)
    else:
        pass
    loss_sim = criterion(ts1, ts2)
    if basemode == 'perceptual':
        loss_sim = loss_sim.mean()
    return loss_sim


def fre_regu_loss(ts1: torch.Tensor, ts2: torch.Tensor, is_avg=True) -> torch.Tensor:
    """[summary]

    Args:
        ts1 (torch.Tensor): serves as label (require_grad = False) in BCE loss
        ts2 (torch.Tensor): serves as prediction (require_grad = True) in BCE loss
        is_avg (bool, optional): whether using the average of real images. Defaults to True.

    Returns:
        torch.Tensor: loss
    """
    # filter = GaussianSmoothing(3, 3, 0.5).to(device)
    # pad = nn.ReflectionPad2d(1)
    # ts1 = ts1 - filter(pad(ts1))
    # ts2 = ts2 - filter(pad(ts2))

    filter = GaussianBlur(3, 0.2)
    ts1 = ts1 - filter(ts1)
    ts2 = ts2 - filter(ts2)
    spectral_transform = SpectralLoss(device=device)
    frequency_thre = int(0.1 * spectral_transform.vector_length)
    ts1_profiles = spectral_transform.spectral_vector(ts1)[:, frequency_thre:]
    ts2_profiles = spectral_transform.spectral_vector(ts2)[:, frequency_thre:]
    if is_avg:
        ts1_profiles_avg = ts1_profiles.mean(dim=0)
        ts1_profiles = torch.zeros_like(ts1_profiles)
        for t in range(ts1_profiles.shape[0]):
            ts1_profiles[t, :] = ts1_profiles_avg
    ts1_profiles = Variable(ts1_profiles, requires_grad=False).to(device)
    ts2_profiles = Variable(ts2_profiles, requires_grad=True).to(device)

    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    return criterion(ts2_profiles, ts1_profiles)

