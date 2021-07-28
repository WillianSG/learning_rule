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
from mpl_toolkits.axes_grid1 import make_axes_locatable

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper functions
from get_ids_from_binary_pattern import *

with open('27Jul2021_23-41-11_dataset_Fusi-size_20.pickle','rb') as f:(
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
			(class2_IDs_list, get_ids_from_binary_pattern(binarized_pattern = pattern)), 
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

# ----------- IDs shared by both classes -----------
shared_IDs_between_classes = np.intersect1d(unique_class1, unique_class2)

total = (len(unique_class1) + len(unique_class2))/2

shared_percentage = np.round(((len(shared_IDs_between_classes)*100)/total), 1)

sharedIDs_matrix = np.zeros(len(full_dataset[0]))

# matrix of shared IDs between classes
for x in range(0, len(sharedIDs_matrix)):
	if x in shared_IDs_between_classes:
		# ---- find ID index
		idx_c1 = np.where(unique_class1 == x)[0][0]
		idx_c2 = np.where(unique_class2 == x)[0][0]

		sharedIDs_matrix[x] = frequency_class1[idx_c1] + frequency_class2[idx_c2]

# ----------- Patterns with shared/non-shared IDs per class -----------
class1_patterns_sum_sharedIDs = np.zeros(len(full_dataset[0]))
class2_patterns_sum_sharedIDs = np.zeros(len(full_dataset[0]))

class1_patterns_sum_nonSharedIDs = np.zeros(len(full_dataset[0]))
class2_patterns_sum_nonSharedIDs = np.zeros(len(full_dataset[0]))

pattern_id = 0
for pattern in full_dataset:
	if (pattern_id % 2) == 0:
		IDs_temp = get_ids_from_binary_pattern(binarized_pattern = pattern)

		if len(np.intersect1d(IDs_temp, class2_IDs_list)) == 0:
			class1_patterns_sum_nonSharedIDs += np.array(pattern)
	else:
		IDs_temp = get_ids_from_binary_pattern(binarized_pattern = pattern)

		if len(np.intersect1d(IDs_temp, class1_IDs_list)) == 0:
			class2_patterns_sum_nonSharedIDs += np.array(pattern)


	pattern_id += 1

# ========================== plotting ==========================

plot_title_size = 8

fig0 = plt.figure(constrained_layout = True, figsize = (13, 6))

widths = [8, 8, 8, 8, 8]
heights = [8, 8]

spec2 = gridspec.GridSpec(
	ncols = 5, 
	nrows = 2, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

fig0.suptitle('Dataset Metadata Summary | size: ' + str(meta_data['dataset_size']) + ', ID: ' + str(meta_data['timestamp']), fontsize = 10)

# ----------- Class 1 | avg matrix -----------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

reshaped_c1 = class1_patterns_sum.reshape(20, 20)

plt.title('Class 1 (even IDs)', size = plot_title_size)

plt.xticks([])
plt.yticks([])

shw_c1 = plt.imshow(reshaped_c1, cmap = 'Greys', interpolation = 'none')

divider = make_axes_locatable(f2_ax1)
cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start = True)
fig0.add_axes(cax)
cbar_c1 = plt.colorbar(shw_c1, cax = cax, orientation = 'horizontal')

# cbar_c1 = plt.colorbar(shw_c1)

# cbar_c1_labels = np.arange(
# 	0.0, 
# 	max(class1_patterns_sum)+1.0,
# 	step = 1.0)

cbar_c1_labels = np.unique(
	class1_patterns_sum, 
	return_counts = False)

cbar_c1.set_ticks(cbar_c1_labels)

# ----------- Class 2 | avg matrix -----------
f2_ax2 = fig0.add_subplot(spec2[1, 0])

reshaped_c2 = class2_patterns_sum.reshape(20, 20)

plt.title('Class 2 (odd IDs)', size = plot_title_size)

plt.xticks([])
plt.yticks([])

shw_c2 = plt.imshow(reshaped_c2, cmap = 'Greys', interpolation = 'none')

divider = make_axes_locatable(f2_ax2)
cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start = True)
fig0.add_axes(cax)
cbar_c2 = plt.colorbar(shw_c2, cax = cax, orientation = 'horizontal')

# cbar_c2 = plt.colorbar(shw_c2)

# cbar_c2_labels = np.arange(
# 	0.0, 
# 	max(class2_patterns_sum)+1.0,
# 	step = 1.0)

cbar_c2_labels = np.unique(
	class2_patterns_sum, 
	return_counts = False)

cbar_c2.set_ticks(cbar_c2_labels)

# ----------- Class 1 | IDs histogram -----------

f2_ax3 = fig0.add_subplot(spec2[0, 1])

counts, bins = np.histogram(unique_class1, bins = num_bins)

plt.title('Class 1 (even IDs)', size = plot_title_size)

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

plt.title('Class 2 (odd IDs)', size = plot_title_size)

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

plt.title('Class 1 (even IDs)', size = plot_title_size)

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

plt.title('Class 2 (odd IDs)', size = plot_title_size)

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

# ----------- Shared IDs percentage -----------

f2_ax6 = fig0.add_subplot(spec2[:, 3])

plt.title('Shared IDs among classes', size = plot_title_size)

patches, texts, a = f2_ax6.pie(
	[shared_percentage, np.round((100 - shared_percentage), 1)], 
	autopct = '%1.1f%%', 
	shadow = True, 
	startangle = 140,
	colors = ['mediumpurple', 'orchid'],
	explode = (0.1, 0))

labels = ['shared', 'unique']

plt.legend(patches, labels, loc = 'best')

# ----------- Class 1 + 2 | avg matrix shared IDs -----------
f2_ax7 = fig0.add_subplot(spec2[:, 4])

plt.title('Shared IDs', size = plot_title_size)

plt.xticks([])
plt.yticks([])

matrix = sharedIDs_matrix.reshape(20, 20)

shw_s = plt.imshow(matrix, cmap = 'Greys', interpolation = 'none')

divider = make_axes_locatable(f2_ax7)
cax = divider.new_vertical(size = '5%', pad = 0.1, pack_start = True)
fig0.add_axes(cax)
cbar_s = plt.colorbar(shw_s, cax = cax, orientation = 'horizontal')

cbar_s_labels = np.unique(
	sharedIDs_matrix, 
	return_counts = False)

# cbar_s_labels = np.arange(
# 	2.0, 
# 	max(sharedIDs_matrix)+1.0,
# 	step = 1.0)

cbar_s.set_ticks(cbar_s_labels)

fig0.suptitle('Dataset Metadata Summary | size: ' + str(meta_data['dataset_size']) + ', ID: ' + str(meta_data['timestamp']), fontsize = 10)

# plt.savefig(
# 	str(meta_data['timestamp']) + '_s' + str(meta_data['dataset_size']) + '.png',
# 	bbox_inches = 'tight', 
# 	dpi = 200)

plt.show()
