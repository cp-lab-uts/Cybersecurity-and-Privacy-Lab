<!--
 * @Description  : 
 * @Author       : Chi Liu
 * @Date         : 2022-04-22 15:04:06
 * @LastEditTime : 2022-04-22 15:38:01
-->
This is the PyTorch implementation of the paper ["Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack"](https://arxiv.org/abs/2203.11433)

![alt](./jpg/method_overview.jpg)

# Basic Environment 
```
pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0
torchvision=0.8.2=py38_cu110
```

# Data preparation 
The data instances in the dataset are stored and applied in a pairwise form, i.e., each instance is a pair of a real image and a fake image. The two images are concatenated horizontally. 

This can be done by ```sew_img.py```

# Training 
```training_attack.py``` contains the training code of the attack model:
```
python training_attacker.py 
        --n_ep          Epoch numbers.
        --bs            Batch size.
        --dir_exp       Experiment directory.
        --n_max_keep    Maximum number of checkpoints to keep.
        --dir_data      Data directory.
        --size          Image size.
        --n_imgs        Image numbers.
        --G_lr G_LR     Generator Learning rate.
        --D1_lr D1_LR   Discriminator1 Learning rate.
        --D2_lr D2_LR   Discriminator2 Learning rate.
        --D3_lr D3_LR   Discriminator3 Learning rate. # in the paper we ban this discriminator
        --D4_lr D4_LR   Discriminator4 Learning rate.
        --beta1 BETA1   Beta1 for Adam.
        --iter_save     Number of iterations for saving a checkpoint.
        --iter_rdc      Number of iterations for reducing learning rate.
        --paras         Number of iterations for saving a checkpoint.
        --pth_ckpt      Path to the saved checkpoint.
        --cln_space     Clean test space.
        --loss_mode     loss mode.
```
Or run ```script.sh``` for simplicity: 
```
sh script.sh
```

# Testing
Run ```testing_attack.py``` to test the attack performance. 
```
python testing_attack.py 
        --dir_exp         Experiment directory.
        --dir_data        Test Data directory.
        --cln_space       Clean test space.
        --new_attack      Start a new attack.
        --detector_ckpt   detector checkpoint path.
        --attacker_ckpt   attacker checkpoint path.
```

# Citation
Feel free to use the code. Cite our paper if you find it helpful:
```
Liu, Chi, Huajie Chen, Tianqing Zhu, Jun Zhang, and Wanlei Zhou. "Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack." arXiv preprint arXiv:2203.11433 (2022).
```