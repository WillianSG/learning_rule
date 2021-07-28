# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from time import localtime, strftime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

prefs.codegen.target = 'numpy'

plotting_funcs_dir = 'plotting_functions'
dataset_dir = 'dataset_F'

# Parent directory
dir_one_up = os.path.dirname(os.getcwd())
dir_two_up = os.path.abspath(os.path.join(os.getcwd() , '../..'))

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(dir_one_up)
sys.path.append(os.path.join(dir_two_up, plotting_funcs_dir))
sys.path.append(os.path.join(dir_two_up, dataset_dir))

# dict_snr = {
# 'pattern_id': pattern_id,
# 'avg_ffrq_out1': [],
# 'avg_ffrq_out2': [],
# 'std_ffrq_out1': [],
# 'std_ffrq_out2': [],
# 'snr_ffrq_out1': [],
# 'snr_ffrq_out2': []
# }

with open(
	'28Jul2021_15-34-26__LR3_1.D_bistTrue_dict_array_snr.pickle',
	'rb') as f:(
	dict_array_snr,
	M_syn,
	correct_response,
	wrong_response,
	dataset_metadata,
	plasticity_rule,
	parameter_set,
	bistability,
	stoplearning,
	populations_biasing_dict) = pickle.load(f)

print('================== dict_array_snr ==================')
for key, value in dict_array_snr[0].items():
	print(key, ':', value)
print('======================================================\n')

# each pos marks how many time the prasent was presented up to that point
x_axis = [i for i in range(1, len(dict_array_snr[0]['snr_ffrq_out1'])+1)]


# all patterns are seen the same amount of time
num_patterns = int(dataset_metadata['dataset_size']/2)

# snr of ouput neuron 1 for both classes of patterns
data_c1_on1_ffrq = [[] for i in range(0, num_patterns)]
data_c2_on1_ffrq = [[] for i in range(0, num_patterns)]

data_c1_on1_std = [[] for i in range(0, num_patterns)]
data_c2_on1_std = [[] for i in range(0, num_patterns)]

data_c1_on1_ffrq_counter = 0
data_c2_on1_ffrq_counter = 0

# snr of ouput neuron 2 for both classes of patterns
data_c1_on2_ffrq = [[] for i in range(0, num_patterns)]
data_c2_on2_ffrq = [[] for i in range(0, num_patterns)]

data_c1_on2_std = [[] for i in range(0, num_patterns)]
data_c2_on2_std = [[] for i in range(0, num_patterns)]

data_c1_on2_ffrq_counter = 0
data_c2_on2_ffrq_counter = 0

pattern_ids_c1 = []
pattern_ids_c2 = []

for x in range(0, len(dict_array_snr)):
	if (dict_array_snr[x]['pattern_id'] % 2) == 0:
		data_c1_on1_ffrq[data_c1_on1_ffrq_counter] = dict_array_snr[x]['avg_ffrq_out1']

		data_c1_on2_ffrq[data_c1_on2_ffrq_counter] = dict_array_snr[x]['avg_ffrq_out2']

		data_c1_on1_std[data_c1_on1_ffrq_counter] = dict_array_snr[x]['std_ffrq_out1']

		data_c1_on2_std[data_c1_on2_ffrq_counter] = dict_array_snr[x]['std_ffrq_out2']

		data_c1_on2_ffrq_counter += 1
		data_c1_on1_ffrq_counter += 1

		pattern_ids_c1.append(dict_array_snr[x]['pattern_id'])
	else:
		data_c2_on2_ffrq[data_c2_on2_ffrq_counter] = dict_array_snr[x]['avg_ffrq_out2']

		data_c2_on1_ffrq[data_c2_on1_ffrq_counter] = dict_array_snr[x]['avg_ffrq_out1']

		data_c2_on2_std[data_c2_on2_ffrq_counter] = dict_array_snr[x]['std_ffrq_out2']

		data_c2_on1_std[data_c2_on1_ffrq_counter] = dict_array_snr[x]['std_ffrq_out1']

		data_c2_on2_ffrq_counter += 1
		data_c2_on1_ffrq_counter += 1

		pattern_ids_c2.append(dict_array_snr[x]['pattern_id'])

# =============================================================
markers = ['1', '2', '3', '4', '+']


axis_label_size = 10
legend_font_size = 8

fig0 = plt.figure(constrained_layout = True, figsize = (8, 8))

widths = [4, 4]
heights = [4, 4]

spec2 = gridspec.GridSpec(
	ncols = 2, 
	nrows = 2, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

x = np.arange(len(data_c1_on1_ffrq[0]))

width = 0.1

# ----------------- SNR out 1 | c1 -----------------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

# counter = 0
# for y in data_c1_on1_ffrq:
# 	plt.plot(
# 		x_axis, 
# 		y, 
# 		linestyle = '-', 
# 		marker = markers[counter],
# 		label = str(pattern_ids_c1[counter]))
# 	counter += 1
# create data

plt.bar(x+0.4, data_c1_on1_ffrq[0], width)
plt.bar(x-0.4, data_c1_on1_ffrq[1], width)
plt.bar(x+0.2, data_c1_on1_ffrq[2], width)
plt.bar(x-0.2, data_c1_on1_ffrq[3], width)
plt.bar(x, data_c1_on1_ffrq[4], width)


plt.errorbar(x+0.4, data_c1_on1_ffrq[0], yerr = data_c1_on1_std[0], fmt = 'o', color = 'r')
plt.errorbar(x-0.4, data_c1_on1_ffrq[1], yerr = data_c1_on1_std[1], fmt = 'o', color = 'r')
plt.errorbar(x+0.2, data_c1_on1_ffrq[2], yerr = data_c1_on1_std[2], fmt = 'o', color = 'r')
plt.errorbar(x-0.2, data_c1_on1_ffrq[3], yerr = data_c1_on1_std[3], fmt = 'o', color = 'r')
plt.errorbar(x, data_c1_on1_ffrq[4], yerr = data_c1_on1_std[4], fmt = 'o', color = 'r')

f2_ax1.legend(prop = {'size': legend_font_size}, title = 'pattern ID')

# plt.ylabel('SNR (c1 | out 1)', size = axis_label_size)

# plt.xticks(np.arange(
# 	x_axis[0], 
# 	x_axis[-1]+1,
# 	step = 1))

# plt.xlim(x_axis[0], x_axis[-1])
# plt.ylim(0.0, np.max(data_c1_on1_ffrq))

# ax = plt.gca()
# ax.axes.xaxis.set_ticklabels([])

# ----------------- SNR out 1 | c2 -----------------
f2_ax2 = fig0.add_subplot(spec2[0, 1])

# counter = 0
# for y in data_c2_on1_ffrq:
# 	plt.plot(
# 		x_axis, 
# 		y, 
# 		linestyle = '-', 
# 		marker = markers[counter],
# 		label = str(pattern_ids_c2[counter]))
# 	counter += 1

plt.bar(x+0.4, data_c2_on1_ffrq[0], width)
plt.bar(x-0.4, data_c2_on1_ffrq[1], width)
plt.bar(x+0.2, data_c2_on1_ffrq[2], width)
plt.bar(x-0.2, data_c2_on1_ffrq[3], width)
plt.bar(x, data_c2_on1_ffrq[4], width)

plt.errorbar(x+0.4, data_c2_on1_ffrq[0], yerr = data_c2_on1_std[0], fmt = 'o', color = 'r')
plt.errorbar(x-0.4, data_c2_on1_ffrq[1], yerr = data_c2_on1_std[1], fmt = 'o', color = 'r')
plt.errorbar(x+0.2, data_c2_on1_ffrq[2], yerr = data_c2_on1_std[2], fmt = 'o', color = 'r')
plt.errorbar(x-0.2, data_c2_on1_ffrq[3], yerr = data_c2_on1_std[3], fmt = 'o', color = 'r')
plt.errorbar(x, data_c2_on1_ffrq[4], yerr = data_c2_on1_std[4], fmt = 'o', color = 'r')

f2_ax2.legend(prop = {'size': legend_font_size}, title = 'pattern ID')

# plt.ylabel('SNR (c2 | out 1)', size = axis_label_size)

# plt.xticks(np.arange(
# 	x_axis[0], 
# 	x_axis[-1]+1,
# 	step = 1))

# plt.xlim(x_axis[0], x_axis[-1])
# plt.ylim(0.0, np.max(data_c2_on1_ffrq))

# ax = plt.gca()
# ax.axes.xaxis.set_ticklabels([])

# ----------------- SNR out 2 | c2 -----------------
f2_ax3 = fig0.add_subplot(spec2[1, 0])

# counter = 0
# for y in data_c2_on2_ffrq:
# 	plt.plot(
# 		x_axis, 
# 		y, 
# 		linestyle = '-', 
# 		marker = markers[counter],
# 		label = str(pattern_ids_c2[counter]))
# 	counter += 1

plt.bar(x+0.4, data_c2_on2_ffrq[0], width)
plt.bar(x-0.4, data_c2_on2_ffrq[1], width)
plt.bar(x+0.2, data_c2_on2_ffrq[2], width)
plt.bar(x-0.2, data_c2_on2_ffrq[3], width)
plt.bar(x, data_c2_on2_ffrq[4], width)

plt.errorbar(x+0.4, data_c2_on2_ffrq[0], yerr = data_c2_on2_std[0], fmt = 'o', color = 'r')
plt.errorbar(x-0.4, data_c2_on2_ffrq[1], yerr = data_c2_on2_std[1], fmt = 'o', color = 'r')
plt.errorbar(x+0.2, data_c2_on2_ffrq[2], yerr = data_c2_on2_std[2], fmt = 'o', color = 'r')
plt.errorbar(x-0.2, data_c2_on2_ffrq[3], yerr = data_c2_on2_std[3], fmt = 'o', color = 'r')
plt.errorbar(x, data_c2_on2_ffrq[4], yerr = data_c2_on2_std[4], fmt = 'o', color = 'r')

f2_ax3.legend(prop = {'size': legend_font_size}, title = 'pattern ID')

# plt.ylabel('SNR (c2 | out 2)', size = axis_label_size)
# plt.xlabel('# presentation', size = axis_label_size)

# plt.xticks(np.arange(
# 	x_axis[0], 
# 	x_axis[-1]+1,
# 	step = 1))

# plt.xlim(x_axis[0], x_axis[-1])
# plt.ylim(0.0, np.max(data_c2_on2_ffrq))

# ----------------- SNR out 2 | c1 -----------------
f2_ax4 = fig0.add_subplot(spec2[1, 1])

# counter = 0
# for y in data_c1_on2_ffrq:
# 	plt.plot(
# 		x_axis, 
# 		y, 
# 		linestyle = '-', 
# 		marker = markers[counter],
# 		label = str(pattern_ids_c1[counter]))
# 	counter += 1

plt.bar(x+0.4, data_c1_on2_ffrq[0], width)
plt.bar(x-0.4, data_c1_on2_ffrq[1], width)
plt.bar(x+0.2, data_c1_on2_ffrq[2], width)
plt.bar(x-0.2, data_c1_on2_ffrq[3], width)
plt.bar(x, data_c1_on2_ffrq[4], width)

plt.errorbar(x+0.4, data_c1_on2_ffrq[0], yerr = data_c1_on2_std[0], fmt = 'o', color = 'r')
plt.errorbar(x-0.4, data_c1_on2_ffrq[1], yerr = data_c1_on2_std[1], fmt = 'o', color = 'r')
plt.errorbar(x+0.2, data_c1_on2_ffrq[2], yerr = data_c1_on2_std[2], fmt = 'o', color = 'r')
plt.errorbar(x-0.2, data_c1_on2_ffrq[3], yerr = data_c1_on2_std[3], fmt = 'o', color = 'r')
plt.errorbar(x, data_c1_on2_ffrq[4], yerr = data_c1_on2_std[4], fmt = 'o', color = 'r')

f2_ax4.legend(prop = {'size': legend_font_size}, title = 'pattern ID')

# plt.ylabel('SNR (c1 | out 2)', size = axis_label_size)
# plt.xlabel('# presentation', size = axis_label_size)

# plt.xticks(np.arange(
# 	x_axis[0], 
# 	x_axis[-1]+1,
# 	step = 1))

# plt.xlim(x_axis[0], x_axis[-1])
# plt.ylim(0.0, np.max(data_c1_on2_ffrq))

plt.show()


