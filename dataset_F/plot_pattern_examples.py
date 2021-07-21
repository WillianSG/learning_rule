# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- automatic reshape based on stimulus size
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, shutil
from matplotlib import gridspec

# dataset = os.path.join(,) # loading data

with open('21Jul2021_11-21-57_dataset_Fusi-size_4.pickle','rb') as f:(
	meta_data,
	full_dataset) = pickle.load(f)

# example = full_dataset[0].reshape(20, 20)

# plt.title('Pattern 0 | size 2 dataset', size = 10)

# plt.imshow(example, cmap = 'Greys', interpolation = 'none')

# plt.xticks([])
# plt.yticks([])

# plt.savefig(meta_data['timestamp'] + '_pattern_0.png')

# example = full_dataset[1].reshape(20, 20)

# plt.title('Pattern 1 | size 2 dataset', size = 10)

# plt.imshow(example, cmap = 'Greys', interpolation = 'none')

# plt.xticks([])
# plt.yticks([])

# plt.savefig(meta_data['timestamp'] + '_pattern_1.png')

# =======================================================

fig0 = plt.figure(constrained_layout = True)

widths = [8, 8]
heights = [8, 8]

spec2 = gridspec.GridSpec(
	ncols = 2, 
	nrows = 2, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

fig0.suptitle('Even IDs | dataset size ' + str(meta_data['dataset_size']), fontsize = 8)

f2_ax1 = fig0.add_subplot(spec2[0, 0])

plt.title('Pattern 0 | size 4 dataset', size = 8)

plt.xticks([])
plt.yticks([])

plt.imshow(full_dataset[0].reshape(20, 20), cmap = 'Greys', interpolation = 'none')

f2_ax1 = fig0.add_subplot(spec2[0, 1])

plt.title('Pattern 2 | size 4 dataset', size = 8)

plt.xticks([])
plt.yticks([])

plt.imshow(full_dataset[2].reshape(20, 20), cmap = 'Greys', interpolation = 'none')

fig0.suptitle('Odd IDs | dataset size ' + str(meta_data['dataset_size']), fontsize = 8)

f2_ax1 = fig0.add_subplot(spec2[1, 0])

plt.title('Pattern 1 | size 4 dataset', size = 8)

plt.xticks([])
plt.yticks([])

plt.imshow(full_dataset[1].reshape(20, 20), cmap = 'Greys', interpolation = 'none')

f2_ax1 = fig0.add_subplot(spec2[1, 1])

plt.title('Pattern 3 | size 4 dataset', size = 8)

plt.xticks([])
plt.yticks([])

plt.imshow(full_dataset[3].reshape(20, 20), cmap = 'Greys', interpolation = 'none')

plt.show()
