import os

# classifier
# os.system("python train_classifier.py --epochs 200 --path_out model/ --nz 10 --lr 0.0002")

# vector-based
# os.system("python train_inversion.py --epochs 200 --path_out vector_based/ --nz 10 --lr 0.01 --truncation 10 --early_stop 20")

# score-based
# os.system("python train_inversion.py --epochs 200 --path_out score_based/ --nz 10 --lr 0.005 --truncation 1 --early_stop 20")

# one-hot
# os.system("python train_inversion.py --epochs 200 --path_out one_hot/ --nz 10 --lr 0.005 --truncation 0 --early_stop 20")

# our_method
# os.system("python train_inversion.py --epochs 200 --path_out our_method/ --nz 10 --lr 0.01 --truncation -1 --early_stop 20")