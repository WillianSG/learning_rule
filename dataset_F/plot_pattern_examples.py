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

with open('01Jun2021_11-49-38_dataset_Fusi.pickle','rb') as f:(
	meta_data,
	class_0,
	class_1) = pickle.load(f)

class_0_ex = class_0[0].reshape(20, 20)
class_1_ex = class_1[0].reshape(20, 20)

plt.title('Example pattern', size = 10)

plt.imshow(class_0_ex, cmap = 'Greys', interpolation = 'none')

plt.xticks([])
plt.yticks([])

plt.savefig('example_pattern.png')
