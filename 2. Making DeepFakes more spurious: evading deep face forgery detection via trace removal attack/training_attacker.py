"""
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-05 13:58:47
@LastEditTime : 2022-01-24 20:34:28
"""
"""
@Author: MaraPapMann
@Desc: To train an attack model that makes an arbitrary fake image undifferentiable.
@Coding: UTF-8
"""


import math
import os
from re import A
import numpy as np
import torch as T
import torch.nn as nn
import tqdm

from skimage import img_as_ubyte
from skimage.io import imsave
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import marapapmann as M
import marapapmann.imlib as im
import marapapmann.pylib as py
from discriminator import FingerprintD, ImageD, SpectrumD, SpectrumDistributionD
from generator import UNet
from marapapmann.imgLoader import imgLoader

import gan_loss
from gp import gradient_penalty
import functools
import warnings

warnings.filterwarnings("ignore")
# ===============================================
# =               Parse arguments               =
# ===============================================

# Set up arguments
py.arg("--n_ep", type=int, default=6, help="Epoch numbers.")
py.arg("--bs", type=int, default=150, help="Batch size.")
py.arg("--dir_exp", type=str, default="exp/test", help="Experiment directory.")
py.arg("--ext", type=str, default="png", help="File extension.")
py.arg(
    "--n_max_keep", type=int, default=5, help="Maximum number of checkpoints to keep."
)
py.arg("--dir_data", type=str, default="data/dataset", help="Data directory.")
py.arg("--size", type=int, default=128, help="Image size.")
py.arg("--n_imgs", type=int, default=0, help="Image numbers.")

#! lr -> G_lr and D_lr
py.arg("--G_lr", type=float, default=1.6e-4, help="Learning rate.")
py.arg("--D1_lr", type=float, default=1.6e-3, help="Learning rate.")
py.arg("--D2_lr", type=float, default=1.6e-3, help="Learning rate.")
py.arg("--D3_lr", type=float, default=1.6e-3, help="Learning rate.")
py.arg("--D4_lr", type=float, default=1.6e-3, help="Learning rate.")
# py.arg('--lr', type=float, default=1e-3, help='Learning rate.')

py.arg("--beta1", type=float, default=0.5, help="Beta1 for Adam.")
py.arg(
    "--iter_save",
    type=int,
    default=10,
    help="Number of iterations for saving a checkpoint.",
)
py.arg(
    "--iter_rdc",
    type=int,
    default=400,
    help="Number of iterations for reducing learning rate.",
)

py.arg(
    "--paras",
    type=list,
    default=[1, 1, 0, 1, 0.1, 0.8],
    help="Number of iterations for saving a checkpoint.",
)

py.arg("--pth_ckpt", type=str, default=None, help="Path to the saved checkpoint.")
py.arg("--cln_space", type=bool, default=True, help="Clean test space.")
py.arg(
    "--debug",
    type=bool,
    default=False,
    help="Debug with error log and error image outputs for nan loss.",
)

py.arg("--loss_mode", type=str, default="wgan", help="loss mode.")
py.arg("--gradient_penalty_mode", type=str, default="0-gp")
py.arg("--gradient_penalty_sample_mode", type=str, default="line")
py.arg("--gradient_penalty_weight", type=float, default=1.0)
py.arg("--gradient_penalty_d_norm", type=str, default="layer_norm")

# Parse arguments
args = py.args()

##############################################################################
#
#                              Losses, Optimiziers
#
##############################################################################
d_loss, g_loss = gan_loss.get_adversarial_losses_fn(args.loss_mode)
sim_loss = gan_loss.sim_loss
fre_regu_loss = gan_loss.fre_regu_loss


def get_d_norm(gradient_penalty_mode):
    # Setup discriminator norm (Cannot use batch normalization with gradient penalty)
    if gradient_penalty_mode == "none":
        d_norm = "batch_norm"
    else:
        d_norm = gradient_penalty_d_norm
    return d_norm


gradient_penalty_mode = args.gradient_penalty_mode
gradient_penalty_sample_mode = args.gradient_penalty_sample_mode
gradient_penalty_weight = args.gradient_penalty_weight
gradient_penalty_d_norm = args.gradient_penalty_d_norm
d_norm = get_d_norm(gradient_penalty_mode)
##############################################################################
#
#                              Training function
#
##############################################################################


def train_D(
    real,
    fake,
    rec,
    D,
    optimizer,
    loss_fun,
    loss_mode,
    gradient_penalty_mode,
    gradient_penalty_sample_mode,
    gradient_penalty_weight,
):
    # train D with real samples
    y_real = D(real)
    if loss_mode == "contr_adv":
        y_fake = D(fake)
    else:
        y_fake = D(rec)

    err_real, err_fake = loss_fun(y_real, y_fake)
    gp = gradient_penalty(
        functools.partial(D),
        real,
        fake,
        gp_mode=gradient_penalty_mode,
        sample_mode=gradient_penalty_sample_mode,
    )
    err = (err_real + err_fake) + gp * gradient_penalty_weight

    err.backward(retain_graph=True)

    # optimize
    optimizer.step()

    # return stats
    return (
        y_real.detach().mean().item(),
        y_fake.detach().mean().item(),
        err.item(),
        err_real.item(),
        err_fake.item(),
        gp.item(),
    )


def train_G(
    x_real,
    x_rec,
    x_fake,
    para_set,
    D1,
    D2,
    D3,
    D4,
    optimizer,
    loss_fun,
    loss_mode,
    sim_loss_fuc,
    fre_reg_fuc,
    **kwargs,
):
    #! change adv loss to contrastive adv loss
    a, b, c, d, lambda1, lambda2 = para_set

    err_list = []
    y_list = []
    for D in [D1, D2, D3, D4]:
        if loss_mode == "contr_adv":
            y_real = D(x_real)
            y_fake = D(x_fake)
            y_rec = D(x_rec)
            err = loss_fun(y_real, y_fake, y_rec)
            err_list.append(err)
            y_list.append(
                (
                    y_real.detach().mean().item(),
                    y_fake.detach().mean().item(),
                    y_rec.detach().mean().item(),
                )
            )
        else:
            y = D(x_rec)
            err = loss_fun(y)
            y_list.append(y.detach().mean().item())
            err_list.append(err)

    err_adv = 0
    for e, weight in zip(err_list, [a, b, c, d]):
        err_adv += e * weight

    # train with similarity loss
    err_sim = sim_loss_fuc(x_fake, x_rec, basemode="ssim", highmode='GaussianFilter')

    # add a frequency regularizer
    err_reg = fre_reg_fuc(x_real, x_rec)
    # err_sim = sim_loss_fuc(x_fake, x_rec, mode="ssim")

    err_list = [i.item() for i in err_list]
    err = (1 - lambda1 - lambda2) * err_adv + lambda1 * err_sim + lambda2 * err_reg
    err.backward()

    # optimize
    optimizer.step()

    return (
        y_list,
        err_list,
        err_adv.item(),
        err_sim.item(),
        err_reg.item(),
        err.item(),
    )


##############################################################################
#
#                              I/O function
#
##############################################################################


def save_result_image(x_real, x_fake, x_rec, dir_img, n_iter, is_debug):
    x_real = x_real.detach().cpu().numpy()
    x_real = np.transpose(x_real, (0, 2, 3, 1))
    x_real = im.immerge(x_real, n_rows=1).squeeze()

    x_fake = x_fake.detach().cpu().numpy()
    x_fake = np.transpose(x_fake, (0, 2, 3, 1))
    x_fake = im.immerge(x_fake, n_rows=1).squeeze()

    x_rec = x_rec.detach().cpu().numpy()
    x_rec = np.transpose(x_rec, (0, 2, 3, 1))
    x_rec = im.immerge(x_rec, n_rows=1).squeeze()

    x_2save = np.concatenate((x_real, x_fake, x_rec), 0)

    if not is_debug:
        imsave(
            py.join(dir_img, "wm_training_iter_%d.png" % (n_iter)),
            img_as_ubyte(x_2save),
        )
    else:
        imsave(py.join(dir_img, "debug_iter_%d.png" % (n_iter)), img_as_ubyte(x_2save))


##############################################################################
#
#                             Training loop
#
##############################################################################

stop_training = False  # debug mode flag

while not stop_training:

    # =======================================================
    # =                Set global parameters                =
    # =======================================================

    if T.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    py.mkdir(args.dir_exp)
    dir_ckpt = py.join(args.dir_exp, "ckpt")
    dir_img = py.join(args.dir_exp, "img")
    pth_log = py.join(args.dir_exp, "training_log.txt")
    err_log = py.join(args.dir_exp, "err_log.txt")
    py.mkdir(dir_ckpt)
    py.mkdir(dir_img)

    if args.cln_space:
        rm_count = 0
        for root, _, files in os.walk(args.dir_exp):
            for file in files:
                os.remove(os.path.join(root, file))
                rm_count += 1
        T.cuda.empty_cache()

        print(f"Done. {rm_count} File Removed")

    # ===================================================
    # =                Load dataset                 =
    # Note1: The generator and discriminators have build-in frequency
    # transformtions. So no need to do frequency transformations here.
    # Note2: Expect sewed images consisted of  most similar real/fake pair
    #        shape as (batch_size, n_channels, H, W*2), eg, (64,3,128,256)
    # ===================================================

    img_loader = imgLoader(0, 0, args.bs, args.dir_data, args.ext)
    #! adjust dataset size
    img_loader.get_files_in_dir(num_file=args.n_imgs)
    img_loader.set_data_transform()
    img_loader.load_img_in_dir()

    # ===================================================
    # =                load untrained model             =
    # ===================================================

    print("Create generator.")
    # Create the generator
    netG = UNet().to(device)
    print(summary(netG, (3, 128, 128)))

    print("Create discriminator.")
    netD1 = ImageD(norm=d_norm).to(device)
    # print(summary(netD1, (3, 128, 128)))
    # netD1 = DeepCNN().xception.to(device)
    # summary = summary(netD1, (3, 128, 128))
    netD2 = SpectrumD(norm=d_norm).to(device)
    # summary = summary(netD2, (3, 128, 128))
    netD3 = SpectrumDistributionD(spectral="nonlinear").to(device)
    # summary = summary(netD3, (3, 128, 128))
    netD4 = FingerprintD(norm=d_norm).to(device)
    # summary = summary(netD4, (3, 128, 128))

    # ===================================================
    # =                 loss and optims                 =
    # ===================================================
    # #! lr -> G_lr and D_lr
    # optim_G = T.optim.Adam(netG.parameters(),
    #                        lr=args.G_lr,
    #                        betas=(args.beta1, 0.999))
    # #! modification: Adam -> sgd
    # optim_D1 = T.optim.Adam(netD1.parameters(),
    #                         lr=args.D1_lr,
    #                         betas=(args.beta1, 0.999))
    # optim_D2 = T.optim.Adam(netD2.parameters(),
    #                         lr=args.D2_lr,
    #                         betas=(args.beta1, 0.999))
    # optim_D3 = T.optim.Adam(netD3.parameters(),
    #                         lr=args.D3_lr,
    #                         betas=(args.beta1, 0.999))
    # optim_D4 = T.optim.Adam(netD4.parameters(),
    #                         lr=args.D4_lr,
    #                         betas=(args.beta1, 0.999))

    optim_G = T.optim.RMSprop(netG.parameters(), lr=args.G_lr)
    optim_D1 = T.optim.RMSprop(netD1.parameters(), lr=args.D1_lr,)

    optim_D2 = T.optim.RMSprop(netD2.parameters(), lr=args.D2_lr,)

    optim_D3 = T.optim.RMSprop(netD3.parameters(), lr=args.D3_lr,)

    optim_D4 = T.optim.RMSprop(netD4.parameters(), lr=args.D4_lr,)

    # optim_G = T.optim.SGD(netG.parameters(), lr=args.G_lr)
    # optim_D1 = T.optim.SGD(netD1.parameters(), lr=args.D1_lr,)

    # optim_D2 = T.optim.SGD(netD2.parameters(), lr=args.D2_lr,)

    # optim_D3 = T.optim.SGD(netD3.parameters(), lr=args.D3_lr,)

    # optim_D4 = T.optim.SGD(netD4.parameters(), lr=args.D4_lr,)

    sch_G = ReduceLROnPlateau(optim_G, mode="min", factor=0.5, patience=5)
    sch_D1 = ReduceLROnPlateau(optim_D1, mode="min", factor=0.5, patience=5)
    sch_D2 = ReduceLROnPlateau(optim_D2, mode="min", factor=0.5, patience=5)
    sch_D3 = ReduceLROnPlateau(optim_D3, mode="min", factor=0.5, patience=5)
    sch_D4 = ReduceLROnPlateau(optim_D4, mode="min", factor=0.5, patience=5)
    # ===================================================
    # =                 Load checkpoint                 =
    # ===================================================
    if args.pth_ckpt != None:
        state_dict = T.load(args.pth_ckpt)
        ep = state_dict["ep"]
        n_iter = state_dict["n_iter"]
        netG.load_state_dict(state_dict["netG"])
        netD1.load_state_dict(state_dict["netD1"])
        netD2.load_state_dict(state_dict["netD2"])
        netD3.load_state_dict(state_dict["netD3"])
        netD4.load_state_dict(state_dict["netD4"])
        optim_G.load_state_dict(state_dict["optim_G"])
        optim_D1.load_state_dict(state_dict["optim_D1"])
        optim_D2.load_state_dict(state_dict["optim_D2"])
        optim_D3.load_state_dict(state_dict["optim_D3"])
        optim_D4.load_state_dict(state_dict["optim_D4"])
    else:
        ep = 0
        n_iter = 0

    # ===================================================
    # =                 Strat training                 =
    # ===================================================
    last_n_iter = 0  # the last usable ckpt before loss turns to nan
    find_error = False
    # min_error = 100.0

    for ep_ in tqdm.trange(args.n_ep, desc="Epoch Loop"):
        min_epoch_error = 50.0
        # Inside epoch loop
        if ep_ < ep:
            ep_ = ep
        ep_ += 1

        for step in tqdm.trange(img_loader.num_step, desc="Iter Loop"):
            # Inside iteration loop
            n_iter += 1
            x_real = img_loader.img[:, :, :, : args.size].to(device)
            x_fake = img_loader.img[:, :, :, args.size :].to(device)

            # ===============================
            # =           Train D           =
            # ===============================

            netG.eval()
            netD1.train()
            netD2.train()
            netD3.train()
            netD4.train()

            netD1.zero_grad()
            netD2.zero_grad()
            netD3.zero_grad()
            netD4.zero_grad()

            x_rec = netG(x_fake)

            #! train D with x_fake

            y1_real, y1_rec, err1, err1_real, err1_rec, gp1 = train_D(
                real=x_real,
                fake=x_fake,
                rec=x_rec,
                D=netD1,
                optimizer=optim_D1,
                loss_fun=d_loss,
                loss_mode=args.loss_mode,
                gradient_penalty_mode=gradient_penalty_mode,
                gradient_penalty_sample_mode=gradient_penalty_sample_mode,
                gradient_penalty_weight=gradient_penalty_weight,
            )
            y2_real, y2_rec, err2, err2_real, err2_rec, gp2 = train_D(
                real=x_real,
                fake=x_fake,
                rec=x_rec,
                D=netD2,
                optimizer=optim_D2,
                loss_fun=d_loss,
                loss_mode=args.loss_mode,
                gradient_penalty_mode=gradient_penalty_mode,
                gradient_penalty_sample_mode=gradient_penalty_sample_mode,
                gradient_penalty_weight=gradient_penalty_weight,
            )
            y3_real, y3_rec, err3, err3_real, err3_rec, gp3 = train_D(
                real=x_real,
                fake=x_fake,
                rec=x_rec,
                D=netD3,
                optimizer=optim_D3,
                loss_fun=d_loss,
                loss_mode=args.loss_mode,
                gradient_penalty_mode=gradient_penalty_mode,
                gradient_penalty_sample_mode=gradient_penalty_sample_mode,
                gradient_penalty_weight=gradient_penalty_weight,
            )
            y4_real, y4_rec, err4, err4_real, err4_rec, gp4 = train_D(
                real=x_real,
                fake=x_fake,
                rec=x_rec,
                D=netD4,
                optimizer=optim_D4,
                loss_fun=d_loss,
                loss_mode=args.loss_mode,
                gradient_penalty_mode=gradient_penalty_mode,
                gradient_penalty_sample_mode=gradient_penalty_sample_mode,
                gradient_penalty_weight=gradient_penalty_weight,
            )

            T.cuda.empty_cache()

            # ===============================
            # =           Train G           =
            # ===============================

            netG.train()
            netD1.eval()
            netD2.eval()
            netD3.eval()
            netD4.eval()
            netG.zero_grad()

            #! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #! adjust weights
            _, _, err_adv, err_sim, err_reg, errG = train_G(
                x_real=x_real,
                x_rec=x_rec,
                x_fake=x_fake,
                para_set=args.paras,
                D1=netD1,
                D2=netD2,
                D3=netD3,
                D4=netD4,
                optimizer=optim_G,
                loss_fun=g_loss,
                loss_mode=args.loss_mode,
                sim_loss_fuc=sim_loss,
                fre_reg_fuc=fre_regu_loss,
            )
            #! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            T.cuda.empty_cache()

            # ==================================
            # =           Checkpoint           =
            # ==================================

            if n_iter % args.iter_save == 0:
                # Log
                f_log = open(pth_log, "a")

                log = (
                    "Ep: %d, Iter: %d, D1: %.5f, D2: %.5f, D3: %.5f,  D4: %.5f,  G: %.5f, G_adv: %.5f, G_sim: %.5f, G_reg: %.5f\n"
                    % (
                        ep_,
                        n_iter,
                        float(err1),
                        float(err2),
                        float(err3),
                        float(err4),
                        float(errG),
                        float(err_adv),
                        float(err_sim),
                        float(err_reg),
                    )
                )

                f_log.write(log)
                f_log.close()

                # Save image
                save_result_image(
                    x_real, x_fake, x_rec, dir_img, n_iter, is_debug=False
                )

                # Save checkpoint
                ckpt = {
                    "ep": ep_,
                    "n_iter": n_iter,
                    "netG": netG.state_dict(),
                    "netD1": netD1.state_dict(),
                    "netD2": netD2.state_dict(),
                    "netD3": netD3.state_dict(),
                    "netD4": netD4.state_dict(),
                    "optim_G": optim_G.state_dict(),
                    "optim_D1": optim_D1.state_dict(),
                    "optim_D2": optim_D2.state_dict(),
                    "optim_D3": optim_D3.state_dict(),
                    "optim_D4": optim_D4.state_dict(),
                }

                if abs(errG) < min_epoch_error:
                    min_epoch_error = errG
                    is_best = True
                else:
                    is_best = False
                M.torchlib.save_checkpoint(
                    ckpt,
                    py.join(dir_ckpt, "iter_%d.dict" % (n_iter)),
                    is_best=is_best,
                    max_keep=args.n_max_keep,
                    best_name=f"best_of_ep{ep_}.ckpt",
                )

            # ==========================================
            # =          Adjust learning rate          =
            # ==========================================

            if n_iter % args.iter_rdc == 0:
                sch_G.step(errG)
                # if n_iter % (args.iter_rdc) == 0:
                sch_D1.step(err1)
                sch_D2.step(err2)
                sch_D3.step(err3)
                sch_D4.step(err4)
            # ==========================================
            # =          Debug mode         =
            # ==========================================
            if math.isnan(errG):
                if args.debug:
                    e_log = open(err_log, "a")
                    err_info = (
                        f"x_real: {x_real}"
                        + "\n"
                        + f"x_fake: {x_fake}"
                        + "\n"
                        + f"x_rec: {x_rec}"
                        + "\n"
                        + f"{x_real.shape}"
                        + "\n"
                        + f"y_list: {y_list}"
                        + "\n"
                    )
                    e_log.write(err_info)
                    e_log.close()
                    save_result_image(
                        x_real, x_fake, x_rec, dir_img, n_iter, is_debug=args.debug
                    )
                    print("\n\n!!Error: found nan loss values. Training stopped!!\n\n")
                    find_error = True
                else:
                    if errG > 8 and n_iter > 100:
                        path_last_ckpt = py.join(dir_ckpt, "best.dict")
                        print(
                            f"\n!!Warning: iter = {n_iter}: found abnormal loss values. Load the best checkpoint.\n Training continued!!"
                        )
                        _state_dict = T.load(path_last_ckpt)
                        # ep = _state_dict['ep']
                        # n_iter = _state_dict['n_iter']
                        netG.load_state_dict(_state_dict["netG"])
                        netD1.load_state_dict(_state_dict["netD1"])
                        netD2.load_state_dict(_state_dict["netD2"])
                        netD3.load_state_dict(_state_dict["netD3"])
                        netD4.load_state_dict(_state_dict["netD4"])
                        optim_G.load_state_dict(_state_dict["optim_G"])
                        optim_D1.load_state_dict(_state_dict["optim_D1"])
                        optim_D2.load_state_dict(_state_dict["optim_D2"])
                        optim_D3.load_state_dict(_state_dict["optim_D3"])
                        optim_D4.load_state_dict(_state_dict["optim_D4"])
                    else:
                        try:
                            if last_n_iter == 0:
                                if n_iter % args.iter_save == 0:
                                    last_n_iter = (
                                        n_iter // args.iter_save - 1
                                    ) * args.iter_save
                                else:
                                    last_n_iter = (
                                        n_iter // args.iter_save - 0
                                    ) * args.iter_save
                            else:
                                last_n_iter -= args.iter_save
                            path_last_ckpt = py.join(
                                dir_ckpt, "iter_%d.dict" % (last_n_iter)
                            )
                            print(
                                f"\n!!Warning: iter = {n_iter}: found nan loss values. Load the most previous checkpoint ({path_last_ckpt}).\n Training continued!!"
                            )
                            _state_dict = T.load(path_last_ckpt)
                            ep = _state_dict["ep"]
                            n_iter = _state_dict["n_iter"]
                            netG.load_state_dict(_state_dict["netG"])
                            netD1.load_state_dict(_state_dict["netD1"])
                            netD2.load_state_dict(_state_dict["netD2"])
                            netD3.load_state_dict(_state_dict["netD3"])
                            netD4.load_state_dict(_state_dict["netD4"])
                            optim_G.load_state_dict(_state_dict["optim_G"])
                            optim_D1.load_state_dict(_state_dict["optim_D1"])
                            optim_D2.load_state_dict(_state_dict["optim_D2"])
                            optim_D3.load_state_dict(_state_dict["optim_D3"])
                            optim_D4.load_state_dict(_state_dict["optim_D4"])

                        except FileNotFoundError:
                            print(path_last_ckpt, n_iter)
                            print(
                                "\n!!Error: not available checkpoint found. Try: Restart training. \n"
                            )
                            find_error = True
            else:
                last_n_iter = 0

            if find_error:
                break
            # Next step
            img_loader.next_step()

        if find_error:
            break

        # Next epoch
        img_loader.next_epoch()

    if args.debug and find_error:
        stop_training = True
    elif (not args.debug) and find_error:
        stop_training = False
    else:
        stop_training = True
