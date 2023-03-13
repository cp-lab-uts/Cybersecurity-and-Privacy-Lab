'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-13 16:29:20
@LastEditTime : 2022-04-02 20:15:18
'''
import torch as T
from detection.DeepCNN import DeepCNN
import marapapmann.pylib as py
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import tqdm
import os
from marapapmann.imgLoader import imgLoader
from generator import UNet
from PIL import Image as I
import glob
import pandas as pd
import numpy as np
import pickle

# ===============================================
# =               Parse arguments               =
# ===============================================
#!
py.arg(
    '--dir_exp',
    type=str,
    default='./detection/detection_dataset/sample_2000/attack/',
    help='Experiment directory.',
)

py.arg(
    '--dir_data',
    type=str,
    default='./detection/detection_dataset/sample_2000/',
    help='Test Data directory.',
)
py.arg('--ext', type=str, default='png', help='File extension.')
py.arg('--cln_space', type=bool, default=True, help='Clean test space.')
py.arg('--new_attack', type=bool, default=False, help='Start a new attack.')
--
#!
py.arg(
    '--detector_ckpt',
    type=str,
    default='./detection/exp_xception_def/ckpt/epoch_10.dict',
    help='detector checkpoint path.',
)
#!
py.arg(
    '--attacker_ckpt',
    type=str,
    default='./exp/test_full_wgan_lowfresim_frereq/ckpt/best_model.ckpt',
    help='attacker checkpoint path.',
)
# Parse arguments
args = py.args()

# =======================================================
# =                Set global parameters                =
# =======================================================

if T.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

py.mkdir(args.dir_exp)

results_path = py.join(args.dir_exp, 'attack_results.csv')
att_sample_path = py.join(args.dir_exp, 'attack_samples')
py.mkdir(att_sample_path)

if args.cln_space:
    rm_count = 0
    for root, _, files in os.walk(args.dir_exp):
        for file in files:
            os.remove(py.join(root, file))
            rm_count += 1
    T.cuda.empty_cache()

    print(f'Done. {rm_count} File Removed')

##############################################################################
#
#                             Load target fake data
#
##############################################################################
target_img_dir = py.join(args.dir_data, 'fake')
target_img_loader = imgLoader(0, 0, batch_size=200, dir=target_img_dir, ext=args.ext)
target_img_loader.get_files_in_dir(shuffle=False, num_file=0)
target_img_loader.set_data_transform()
target_img_loader.load_img_in_dir()

##############################################################################
#
#                             Load detector and attacker
#
##############################################################################
print("========> Loading detector and attacker")
if 'resnet' in args.detector_ckpt:
    D = DeepCNN().resnet.to(device)
elif 'xception' in args.detector_ckpt:
    D = DeepCNN().xception.to(device)

detector_state_dict = T.load(args.detector_ckpt)
D.load_state_dict(detector_state_dict['model'])
print("========> Load detector successfully.")

A = UNet().to(device)
attacker_state_dict = T.load(args.attacker_ckpt)
A.load_state_dict(attacker_state_dict['netG'])
print("========> Load attacker successfully.")


##############################################################################
#
#                             Create attack samples
#
##############################################################################
def save_batch_image(ims, from_name_list, to_path):
    ims = ims.detach().cpu()
    for im, name in zip(ims, from_name_list):
        name = name.split('/')[-1]
        im_to_path = py.join(to_path, name)
        T = transforms.ToPILImage()
        T(im).save(im_to_path)


if args.new_attack:
    print("========> Creating attack samples.")
    print(f"---save to {att_sample_path}")
    with T.no_grad():
        A.eval()
        for step in tqdm.trange(target_img_loader.num_step, desc='Iter Loop'):
            x_target = target_img_loader.img.to(device)
            from_name_list = target_img_loader.cur_files
            # print(from_name_list)
            x_attack = A(x_target)
            #         # print("==========", x_attack.shape)
            save_batch_image(x_attack, from_name_list, att_sample_path)
            target_img_loader.next_step()
    print("========> Creation over.")

##############################################################################
#
#                             Load test datasets (real/fake/attack)
#
##############################################################################


class TestDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(py.join(path, '*.png'))
        self.transform = transform

    def __getitem__(self, index):
        x = I.open(self.image_paths[index])
        if self.transform is not None:
            x = self.transform(x)

        return x, self.image_paths[index].split('/')[-1]

    def __len__(self):
        return len(self.image_paths)


def detection_acc(test_results):
    accs = []
    for col in ['real-results', 'fake-results', 'attack-results']:
        outputs = test_results[col]
        if col == 'real-results':
            acc = np.sum(outputs) / len(outputs)
        else:
            acc = 1 - np.sum(outputs) / len(outputs)
        accs.append(acc)
    return accs


trans_funs = []
trans_funs.append(transforms.ToTensor())
trans_funs.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
trans_funs = transforms.Compose(trans_funs)

real_img_dir = py.join(args.dir_data, 'real')
fake_img_dir = target_img_dir
attack_img_dir = att_sample_path

test_map = {'real': real_img_dir, 'fake': fake_img_dir, 'attack': attack_img_dir}
test_results = {}
for key in test_map.keys():
    dataset = TestDataset(test_map[key], transform=trans_funs)
    loader = T.utils.data.DataLoader(
        dataset, batch_size=200, shuffle=False, num_workers=2
    )
    results = {'filenames': [], 'outputs': [], 'results': []}
    with T.no_grad():
        D.eval()
        softmax = nn.Softmax(dim=1).to(device)
        for data, filename in tqdm.tqdm(loader, desc='Iter Loop'):
            data = data.to(device)
            re = softmax(D(data)).detach().cpu().numpy()
            results['outputs'].append(re)
            results['filenames'].append(filename)
            results['results'].append(np.argmax(re, -1))
    test_results[key + '-filenames'] = [j for i in results['filenames'] for j in i]
    test_results[key + '-outputs'] = [j for i in results['outputs'] for j in i]
    test_results[key + '-results'] = [j for i in results['results'] for j in i]
    # print(test_results.keys())
test_results['acc'] = detection_acc(test_results)
with open(results_path, 'wb') as f:
    pickle.dump(test_results, f)
print(test_results['acc'])
