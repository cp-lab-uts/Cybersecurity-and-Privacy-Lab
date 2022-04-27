"""
@Author: MaraPapMann
@Desc: A personal toolkit class that contains basic functions supporting miscellaneous operations.
@Encoding: UTF-8
"""
import os
import numpy as np
import cv2
from glob import glob
from typing import Any
import marapapmann.pylib as py
import random as rd
import tqdm
from shutil import copy, copyfile


def divide_train_test_data(dir_data:str, dir_train:str, dir_test:str, ext:str, n_train:int, n_test:int) -> None:
    """
    @Desc: To randomly sample a certain amount of data, and divide them into training and test datasets.
    @Params:
        dir_data: Data directory;
        dir_train: Training data directory;
        dir_test: Test data directory;
        ext: File extension;
        n_train: Number of training data;
        n_test: Number of test data;
    @Return:
    """
    lst_files = get_files_in_dir_by_key(dir_data, ext)
    lst_files = rd.sample(lst_files, n_train + n_test)
    lst_train_data = lst_files[:n_train]
    lst_test_data = lst_files[n_train:]

    for f in tqdm.tqdm(lst_train_data, desc='Copying training data...'):
        os.system('cp %s %s'%(f, dir_train))
    
    for f in tqdm.tqdm(lst_test_data, desc='Copying test data...'):
        os.system('cp %s %s'%(f, dir_test))
    return


def get_files_in_dir_by_key(dir:str, ext:str) -> list:
    """
    @Desc: To get all files in directory with a extension.
    @Params:
        dir: string, path to the directory;
        ext: string, the extension name;
    @Return:
        files: list of strings, all files in the directory with the extension name.
    """
    files = []
    if dir[-1] != '/':
        dir = dir + '/'
    files += [dir + each for each in os.listdir(dir) if each.endswith(ext)]
    return files


def parse_1_img_to_ndarray(img_pth:str) -> np.ndarray:
    """
    @Desc:
    @Params:
        img_pth: Path of the image;
    @Return:
        img: Image in an np.ndarray;
    """
    img = cv2.imread(img_pth)
    return img


def RGB2gray(rgb:np.ndarray) -> np.ndarray:
    """
    @Desc: To transform the image RGB info into gray scale.
    @Params:
        rgb: numpy array, red, green, blue signal;
    @Return:
        gray: an array of gray scale.
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_sub_dirs(dir_sp:str) -> list:
    """
    @Desc: To get all sub-diretories in a directory.
    @Params:
        dir_sp: the directory for sampling;
    @Return:
        lst_sub_dirs: list of the sub-diretories.
    """
    lst_sub_dirs = glob(py.join(dir_sp, '*/'))
    return lst_sub_dirs


def get_dir_name(dir_pth:str) -> str:
    """
    @Desc: To get names of the diretory.
    @Params:
        dir_pth: the directory path;
    @Return:
        dir_name: name of the directory
    """
    dir_name = dir_pth.split('/')[-2]
    return dir_name
