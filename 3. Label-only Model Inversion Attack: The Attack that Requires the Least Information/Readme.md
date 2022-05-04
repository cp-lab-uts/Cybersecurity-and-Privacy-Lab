# Label-onyl Model Inversion Attack
Authors: Dayong Ye, Tianqing Zhu, Shuai Zhou, Bo Liu, Wanlei Zhou
This is the PyTorch implementation of the paper ["Label-only Model Inversion Attack: The Attack that Requires the Least Information"](https://arxiv.org/pdf/2203.06555.pdf). We provide an example on MNIST dataset (which can also be used on Fashion-MNIST and CIFAR-10) and an example on FaceScrub dataset (which can also be used on CelebA).

## Requirements
The code is written in Python3. You can install the required packages by running:

```
$ pip3 install -r requirements.txt
```

## Data

### Datasets
Five public datasets are used to train the target classifier and the inversion model, including [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [FaceScrub](http://vintage.winklerbros.net/facescrub.html), and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

### Preprocess
MNIST, FashionMNIST and CIFAR-10 are resized to 32 × 32. 

Both FaceScrub and CelebA are transformed to greyscale images with each pixel value in [0, 1].  Resize both them to 64 × 64.


## Run

```
$ python3 train_classifier.py 
```

Train the inversion model:

```
$ python3 train_inversion.py
```
You can set the truncation size by the `--truncation` parameter and select different methods. 
- `truncation = 10 or 526 (number of classes)`: vector-based method
- `truncation = 1`: score-based method
- `truncation = 0`: one-hot method
- `truncation = -1`: our method

## Citation

```
@article{ye2022label,
  title={Label-only Model Inversion Attack: The Attack that Requires the Least Information},
  author={Ye, Dayong and Zhu, Tianqing and Zhou, Shuai and Liu, Bo and Zhou, Wanlei},
  journal={arXiv preprint arXiv:2203.06555},
  year={2022}
}
```