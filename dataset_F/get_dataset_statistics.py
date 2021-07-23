# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- automatic reshape based on stimulus size

References:
- https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/

Problems:
- Can't add gridlines to avg matrices
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle, os, sys
import pylab as plab

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper functions
from get_ids_from_binary_pattern import *

with open('21Jul2021_13-09-12_dataset_Fusi-size_10.pickle','rb') as f:(
	meta_data,
	full_dataset) = pickle.load(f)

print('================== dataset metadata ==================')
for key, value in meta_data.items():
	print(key, ':', value)
print('======================================================\n')

class1_patterns_sum = np.zeros(len(full_dataset[0]))
class2_patterns_sum = np.zeros(len(full_dataset[0]))

class1_IDs_list = np.array([])
class2_IDs_list = np.array([])

pattern_id = 0
for pattern in full_dataset:
	if (pattern_id % 2) == 0:
		class1_IDs_list = np.concatenate(
			(class1_IDs_list, get_ids_from_binary_pattern(binarized_pattern = pattern)), 
			axis = None)

		class1_patterns_sum += np.array(pattern)
	else:
		class2_IDs_list = np.concatenate(
			(class1_IDs_list, get_ids_from_binary_pattern(binarized_pattern = pattern)), 
			axis = None)


		class2_patterns_sum += np.array(pattern)

	pattern_id += 1

# ========================== Metadata Statistics ==========================
# ----------- plotting parameters -----------
num_bins = 20

# ----------- IDs frequency -----------
class1_IDs_list = np.sort(class1_IDs_list)
class2_IDs_list = np.sort(class2_IDs_list)

unique_class1, frequency_class1 = np.unique(
	class1_IDs_list, 
	return_counts = True)

frequency_class1_unique = np.unique(frequency_class1)

freq_class1_xticks = np.round(
	1.0-(np.mean(frequency_class1_unique)/frequency_class1_unique[-1]),
	1)

unique_class2, frequency_class2 = np.unique(
	class2_IDs_list, 
	return_counts = True)

frequency_class2_unique = np.unique(frequency_class2)

freq_class2_xticks = np.round(
	1.0-(np.mean(frequency_class2_unique)/frequency_class2_unique[-1]),
	1)

# ----------- Avg matrix IDs freq normalized -----------
# class1_patterns_sum = class1_patterns_sum/max(frequency_class1)
# class2_patterns_sum = class2_patterns_sum/max(frequency_class2)

# ========================== plotting ==========================

fig0 = plt.figure(constrained_layout = True, figsize = (10, 6))

widths = [8, 8, 8]
heights = [8, 8]

spec2 = gridspec.GridSpec(
	ncols = 3, 
	nrows = 2, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

fig0.suptitle('Avg Pattern Matrices | dataset size ' + str(meta_data['dataset_size']), fontsize = 8)

# ----------- Class 1 | avg matrix -----------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

reshaped_c1 = class1_patterns_sum.reshape(20, 20)

plt.title('Class 1 | even IDs', size = 10)

plt.xticks([])
plt.yticks([])

shw_c1 = plt.imshow(reshaped_c1, cmap = 'Greys', interpolation = 'none')

cbar_c1 = plt.colorbar(shw_c1)

cbar_c1_labels = np.arange(
	0.0, 
	max(class1_patterns_sum)+1.0,
	step = 1.0)

cbar_c1.set_ticks(cbar_c1_labels)

plt.grid(True, color = 'red', linestyle = '-.', linewidth = 2)

# ----------- Class 2 | avg matrix -----------
f2_ax2 = fig0.add_subplot(spec2[1, 0])

reshaped_c2 = class2_patterns_sum.reshape(20, 20)

plt.title('Class 2| odd IDs', size = 10)

plt.xticks([])
plt.yticks([])

shw_c2 = plt.imshow(reshaped_c2, cmap = 'Greys', interpolation = 'none')
cbar_c2 = plt.colorbar(shw_c2)

cbar_c2_labels = np.arange(
	0.0, 
	max(class2_patterns_sum)+1.0,
	step = 1.0)

cbar_c2.set_ticks(cbar_c2_labels)

plt.grid(True, color = 'red', linestyle = '-', linewidth = 1)

# ----------- Class 1 | IDs histogram -----------

f2_ax3 = fig0.add_subplot(spec2[0, 1])

counts, bins = np.histogram(unique_class1, bins = num_bins)

plt.hist(
	unique_class1,
	align = 'mid',
	color = 'lightblue',
	edgecolor = 'k',
	bins = num_bins)

plt.yticks(np.arange(
	0.0, 
	max(counts)+2.0,
	step = 2.0))

plt.xlabel('Neuron IDs', size = 8)
plt.ylabel('Frequency count', size = 8)

# ----------- Class 2 | IDs histogram -----------

f2_ax4 = fig0.add_subplot(spec2[1, 1])

counts, bins = np.histogram(unique_class2, bins = num_bins)

plt.hist(
	unique_class2,
	align = 'mid',
	color = 'tomato',
	edgecolor = 'k',
	bins = num_bins)

plt.yticks(np.arange(
	0.0, 
	max(counts)+2.0,
	step = 2.0))

plt.xlabel('Neuron IDs', size = 8)
plt.ylabel('Frequency count', size = 8)

# ----------- Class 1 | IDs frequency histogram -----------

f2_ax5 = fig0.add_subplot(spec2[0, 2])

counts, bins = np.histogram(frequency_class1, bins = num_bins)

plt.hist(
	frequency_class1,
	align = 'mid',
	color = 'lightblue',
	edgecolor = 'k',
	bins = num_bins)

plt.yticks(np.arange(
	0.0, 
	max(counts)+20.0,
	step = 20.0))

plt.xticks(np.arange(
	1.0, 
	max(frequency_class1)+1.0, 
	step = 1.0))

plt.xlabel('Frequency', size = 8)
plt.ylabel('IDs count', size = 8)

# ----------- Class 2 | IDs frequency histogram -----------

f2_ax6 = fig0.add_subplot(spec2[1, 2])

counts, bins = np.histogram(frequency_class2, bins = num_bins)

plt.hist(
	frequency_class2,
	align = 'mid',
	color = 'tomato',
	edgecolor = 'k',
	bins = num_bins)

plt.yticks(np.arange(
	0.0, 
	max(counts)+20.0,
	step = 20.0))

plt.xticks(np.arange(
	1.0, 
	max(frequency_class2)+1.0, 
	step = 1.0))

plt.xlabel('Frequency', size = 8)
plt.ylabel('IDs count', size = 8)

plt.show()
