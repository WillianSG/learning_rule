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

with open('01Jun2021_17-32-36_dataset_Fusi-size_100.pickle','rb') as f:(
	meta_data,
	full_dataset) = pickle.load(f)

example = full_dataset[0].reshape(20, 20)

plt.title('Example pattern', size = 10)

plt.imshow(example, cmap = 'Greys', interpolation = 'none')

plt.xticks([])
plt.yticks([])

plt.savefig(meta_data['timestamp'] + '_example_pattern.png')
