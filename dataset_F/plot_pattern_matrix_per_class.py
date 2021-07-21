# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- automatic reshape based on stimulus size
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle, os

with open('21Jul2021_13-09-12_dataset_Fusi-size_10.pickle','rb') as f:(
	meta_data,
	full_dataset) = pickle.load(f)

print('================== dataset metadata ==================')
for key, value in meta_data.items():
	print(key, ':', value)
print('======================================================\n')

class1_patterns_sum = np.zeros(len(full_dataset[0]))
class2_patterns_sum = np.zeros(len(full_dataset[0]))

pattern_id = 0
for pattern in full_dataset:
	if (pattern_id % 2) == 0:
		class1_patterns_sum += np.array(pattern)
	else:
		class2_patterns_sum += np.array(pattern)

	pattern_id += 1

class1_patterns_sum = class1_patterns_sum/(meta_data['dataset_size']/2)
class2_patterns_sum = class2_patterns_sum/(meta_data['dataset_size']/2)


# ========================== plotting ==========================

fig0 = plt.figure(constrained_layout = True)

widths = [8]
heights = [8, 8]

spec2 = gridspec.GridSpec(
	ncols = 1, 
	nrows = 2, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

fig0.suptitle('Avg Pattern Matrices | dataset size ' + str(meta_data['dataset_size']), fontsize = 8)

# ----------- Class 1 -----------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

reshaped_c1 = class1_patterns_sum.reshape(20, 20)

plt.title('Class 1 | even IDs', size = 10)

plt.xticks([])
plt.yticks([])

plt.imshow(reshaped_c1, cmap = 'Greys', interpolation = 'none')

# ----------- Class 2 -----------
f2_ax2 = fig0.add_subplot(spec2[1, 0])

reshaped_c2 = class2_patterns_sum.reshape(20, 20)

plt.title('Class 2| odd IDs', size = 10)

plt.imshow(reshaped_c2, cmap = 'Greys', interpolation = 'none')

plt.xticks([])
plt.yticks([])

plt.savefig(
	meta_data['timestamp'] +  '_dataset_size_' + str(meta_data['dataset_size']) + '_summed_patterns.png',
	bbox_inches = 'tight',
	dpi = 200)

fn =  meta_data['timestamp'] +  '_dataset_size_' + str(meta_data['dataset_size']) + '_summed_patterns.pickle'

with open(fn, 'wb') as f:
	pickle.dump((
		reshaped_c1,
		reshaped_c2
		), f)