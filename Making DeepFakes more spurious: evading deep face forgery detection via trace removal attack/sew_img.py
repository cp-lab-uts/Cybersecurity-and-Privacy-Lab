'''
@Description  : 
@Author       : Chi Liu
@Date         : 2022-01-05 13:58:47
@LastEditTime : 2022-01-09 17:37:13
'''
from marapapmann.custom import get_files_in_dir_by_key
import marapapmann.pylib as py
from PIL import Image as I
import tqdm
import numpy as np

if __name__ == '__main__':

    # Set global parameters
    dir_in = './data/sample_testing_selected'
    dir_out = './data/stgan/'
    ext = 'png'
    # dir_real = py.join(dir_in, 'real')
    # dir_fake = py.join(dir_in, 'fake')
    dir_sewed = py.join(dir_in, 'sewed')
    py.mkdir(dir_sewed)

    # Get files
    lst_real_pth = get_files_in_dir_by_key(dir_in, ext)
    lst_real_pth = list(set([i.split('-')[0] for i in lst_real_pth]))

    # Processing loop
    for i in tqdm.trange(len(lst_real_pth), desc='Sewing images...'):
        cur_real_pth = lst_real_pth[i] + '-real.png'
        # cur_real_bn = lst_real_pth[i].split('/')[-1]
        # print(cur_real_bn)
        # cur_fake_pth = py.join(dir_fake,
        #                        cur_real_bn.replace('real', 'haircolor'))
        cur_fake_pth = lst_real_pth[i] + '-fake-age.png'
        # print(cur_real_bn)
        cur_img_real = np.array(I.open(cur_real_pth))
        cur_img_fake = np.array(I.open(cur_fake_pth))

        cur_img_2save = np.concatenate((cur_img_real, cur_img_fake), 1)
        cur_img_2save = I.fromarray(cur_img_2save)
        cur_img_2save.save(
            py.join(dir_sewed, lst_real_pth[i].split('/')[-1] + '.png'))

    # change file name and gather together
    
