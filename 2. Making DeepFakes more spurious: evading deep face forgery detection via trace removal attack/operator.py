"""
@Author:
@Desc:
@Encoding:
"""
import torch as T
import random
from PIL import Image as I
import torchvision.transforms as transforms
from typing import Any
import numpy as np
from io import BytesIO
from marapapmann.custom import get_files_in_dir_by_key
import tqdm


class imgLoader():
    def __init__(self,
                 resize: int,
                 rd_crop: int,
                 batch_size: int,
                 dir: str,
                 ext: str,
                 to_gray: bool = False,
                 blur: bool = False,
                 prob_blur: float = 0.1,
                 jpeg: bool = False,
                 prob_jpeg: float = 0.1,
                 flip: bool = False,
                 prob_flip: float = 0.2):
        """
        To initialize the image loader.
        @Params:
            self;
            resize: Resolution for resizing;
            rd_crop: Resolution for center cropping;
            batch_size: The number of images to be parsed in 1 epoch;
            dir: The image directory;
            ext: The extension of the images;
            blur: Whether to perform random gaussian blurring;
            prob_blur: Probability of blurring;
            jpeg: Whether to perform random jpeg compression;
            prob_jpeg: Probability of jpeg compression;
            flip: Whether to perform random flipping;
            prob_flip: Probability of random flipping;
        """
        self.resize = resize
        self.rd_crop = rd_crop
        self.batch_size = batch_size
        self.dir = dir
        self.ext = ext
        self.blur = blur
        self.prob_blur = prob_blur
        self.jpeg = jpeg
        self.prob_jpeg = prob_jpeg
        self.flip = flip
        self.prob_flip = prob_flip
        self.to_gray = to_gray

    def set_data_transform(self):
        """
        To set transformation compose of the images to load.
        @params:
            self;
        @return:
        """
        compose = []
        if self.resize:
            compose.append(transforms.Resize(self.resize))
        if self.rd_crop:
            compose.append(transforms.RandomCrop(self.rd_crop))
        if self.to_gray:
            compose.append(transforms.Grayscale(num_output_channels=1))
        compose.append(transforms.ToTensor())
        self.data_transform = transforms.Compose(compose)

    def load_single_img(self, img_pth: str) -> T.Tensor:
        """
        To load a single image into a tensor.
        @Params:
            self;
            img_pth: Path of an image;
        @Return:
            imgs: A tensor of images.
        """
        img = I.open(img_pth)
        # Random jpeg compression
        if self.jpeg:
            img = self.rd_jpeg(img, self.prob_jpeg)
        # Random gaussian blur
        if self.blur:
            img = self.rd_gaussian_blur(img, self.prob_blur)
        # Random Flipping
        if self.flip:
            img = self.rd_flip(img, self.prob_flip)
        img = self.data_transform(img)
        img = img.unsqueeze(0)
        return img

    def load_imgs_in_lst(self, img_pths: list) -> T.Tensor:
        """
        To load images in a list.
        @Params:
            self;
            img_pths: Paths of images;
        @Return:
            imgs: A tensor of images.
        """
        imgs = None
        if img_pths != None:
            imgs = self.load_single_img(img_pths[0])
            if len(img_pths) > 1:
                for i in range(1, len(img_pths)):
                    cur_img = self.load_single_img(img_pths[i])
                    cur_img = self.to_tensor(cur_img)
                    cur_img = cur_img.unsqueeze(0)
                    imgs = T.cat((imgs, cur_img), 0)
        return imgs

    def get_files_in_dir(self, shuffle=True, num_file: int = 1000) -> None:
        """
        To get all files in a directory stored in a list.
        @Params:
            self;
            shuffle: Whether to shuffle the order of the files;
            num_file: The maximum number of files in a list;
        @Return:
        """
        self.files = get_files_in_dir_by_key(dir=self.dir, ext=self.ext)
        self.files.sort()
        if shuffle:
            random.shuffle(self.files)
        if num_file:
            self.files = self.files[:num_file]
        self.files_itr = self.files
        self.num_step = int(len(self.files) / self.batch_size)
        if len(self.files) % self.batch_size != 0:
            self.num_step += 1
        return

    def load_img_in_dir(self) -> None:
        """
        To load images in the diretory
        @Params:
            self;
        @Return:
        """
        if self.batch_size < len(self.files_itr):
            self.img = self.load_single_img(self.files_itr[0])
            for i in range(1, self.batch_size):
                cur_img = self.load_single_img(self.files_itr[i])
                self.img = T.cat(
                    (self.img, cur_img),
                    0)  # Concatenate the last image to the previous tensor
                del cur_img
            self.cur_files = self.files_itr[:self.
                                            batch_size]  # Save the paths of the images at the current step
        elif self.batch_size >= len(self.files_itr) & len(self.files_itr) > 0:
            self.img = self.load_single_img(self.files_itr[0])
            for i in range(1, len(self.files_itr)):
                cur_img = self.load_single_img(self.files_itr[i])
                self.img = T.cat(
                    (self.img, cur_img),
                    0)  # Concatenate the last image to the previous tensor
                del cur_img
            self.cur_files = self.files_itr  # Save the paths of the images at the current step
        else:
            pass

    def next_step(self) -> None:
        """
        To proceed to the next step and load the next set of files.
        Params:
            self;
        Return:
        """
        if len(self.files_itr) < self.batch_size:
            # print('Files enumerated!')
            pass
        else:
            self.files_itr = self.files_itr[self.batch_size:]
            self.load_img_in_dir()

    def next_epoch(self):
        self.files_itr = self.files
        random.shuffle(self.files_itr)
        self.load_img_in_dir()

    def parse_azmt_list(self, azmt_list_pth: str) -> dict:
        """
        To parse the azimuthal integral list into a dictionary.
        @Params:
            azmt_list_pth: Path to an AI list file;
        @return: dict_ai: A dictionary of AI.
        """
        # Initialization
        dict_ai = {}
        f = open(azmt_list_pth, 'r')

        # Parse the azimuthal integral csv file into a dictionary
        for i, line in enumerate(f):
            cur_line = line.rstrip().split(',')
            dict_ai.update(
                {cur_line[0]: np.array(list(map(float, cur_line[1:])))})
        return dict_ai

    def parse_candidate_list(self, cdd_lst_pth):
        """
        To parse the candidate .csv file into a dictionary.
        """
        # Initialization
        d = {}
        f = open(cdd_lst_pth, 'r')

        # Parse
        for i, line in enumerate(f):
            cur_line = line.rstrip().split(',')
            d.update({cur_line[0].rstrip('_features.pt'): cur_line[1:]})
        return d

    def rd_flip(self, img: Any, prob: float = 0.2) -> Any:
        """
        To randomly flip the given image with the given probability.
        @Params:
            self;
            img: The given image;
            prob: The given probability;
        @Return:
            img_flipped: The flipped image.
        """
        tf = transforms.Compose([transforms.RandomHorizontalFlip(prob)])
        img_flipped = tf(img)
        return img_flipped

    def rd_gaussian_blur(self, img: Any, prob: float = 0.1) -> Any:
        """
        To randomly Gaussian blur the given image with the given probability.
        @Params:
            self;
            img: The given image;
            prob: The given probability;
        @Return:
            img_blurred: The blurred image.
            img: The not blurred image.
        """
        prob_ = random.uniform(0., 1.)
        if prob > prob_:
            kernel_size = 0
            while kernel_size % 2 == 0:
                kernel_size = random.randint(1, 5)
            tf = transforms.Compose([
                transforms.GaussianBlur(kernel_size=kernel_size,
                                        sigma=(0.1, 3.))
            ])
            img_blurred = tf(img)
            return img_blurred
        else:
            return img

    def rd_jpeg(self, img: Any, prob: float = 0.1) -> Any:
        """
        To randomly JPEG compress the given image with the given probability.
        @Params:
            self;
            img: The given image;
            prob: The given probability;
        @Return:
            img_jpeg: The compressed image.
            img: The not compressed image.
        """
        prob_ = random.uniform(0., 1.)
        if prob > prob_:
            quality = random.randint(30, 100)
            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            img_jpeg = I.open(buffer)
            return img_jpeg
        else:
            return img

    def check_imgs_in_dir(self) -> list:
        """
        @Desc: Test images in directory to return a list of damaged image if any.
        @Param:
        @Return:
            lst_damaged_imgs
        """
        lst_damaged_imgs = []

        for f in tqdm.tqdm(self.files, 'Checking images...'):
            try:
                x = self.load_single_img(f)
            except Exception as e:
                print(e)
                lst_damaged_imgs.append(f)
        return lst_damaged_imgs