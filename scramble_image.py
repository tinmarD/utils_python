# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:35:35 2018

@author: deudon
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


image_dir = r'C:\Users\deudon\Desktop\M4\ProjetDanae\DanaSoft_clean\data\images\FAM'

out_dir = os.path.join(image_dir, 'scrambled')
    
im_list = os.listdir(image_dir)
n_images = len(im_list)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for i in range(n_images):
    print('{} : {}'.format(i, os.path.join(image_dir, im_list[i])))
    im_i = mpimg.imread(os.path.join(image_dir, im_list[i]))
    im_shape = im_i.shape
    im_i_r = im_i.reshape(im_shape[0]*im_shape[1], im_shape[2])
    random_ind = np.arange(im_i_r.shape[0])
    random.shuffle(random_ind)
    im_i_r_shuffled = im_i_r[random_ind]
    im_scrambled = im_i_r_shuffled.reshape(im_shape)
    mpimg.imsave(os.path.join(out_dir, im_list[i]), im_scrambled)
        
