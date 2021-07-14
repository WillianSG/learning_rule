# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- automatic reshape based on stimulus size
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, shutil

# dataset = os.path.join(,) # loading data

with open('12Jul2021_12-40-29_dataset_Fusi-size_2.pickle','rb') as f:(
	meta_data,
	full_dataset) = pickle.load(f)

example = full_dataset[0].reshape(20, 20)

plt.title('Pattern 0 | size 2 dataset', size = 10)

plt.imshow(example, cmap = 'Greys', interpolation = 'none')

plt.xticks([])
plt.yticks([])

plt.savefig(meta_data['timestamp'] + '_pattern_0.png')

example = full_dataset[1].reshape(20, 20)

plt.title('Pattern 1 | size 2 dataset', size = 10)

plt.imshow(example, cmap = 'Greys', interpolation = 'none')

plt.xticks([])
plt.yticks([])

plt.savefig(meta_data['timestamp'] + '_pattern_1.png')
