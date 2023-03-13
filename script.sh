
###
 # @Description  : 
 # @Author       : Chi Liu
 # @Date         : 2022-01-13 21:50:01
 # @LastEditTime : 2022-02-03 18:33:15
### 

# cd /home/chi-liu/Projects/MTR-Net/20211220_liuchi/detection
# python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/detection/training_detector.py \
# --dir_exp exp_xception_def --detector xception --is_defense True
# python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/detection/training_detector.py \
# --dir_exp exp_resnet_def  --detector resnet --is_defense True
# python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/detection/training_detector.py \
# --dir_exp exp_xception --detector xception
# python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/detection/training_detector.py \
# --dir_exp exp_resnet  --detector resnet

cd /home/chi-liu/Projects/MTR-Net/20211220_liuchi/
# python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/training_attacker.py \
# --dir_exp exp/test_full --n_imgs 0 


echo "Activate"
python /home/chi-liu/Projects/MTR-Net/20211220_liuchi/training_attacker.py \
--dir_exp exp/test_full_wgan_lowfresim_frereq_5 --n_imgs 0 --loss_mode 'lsgan'
