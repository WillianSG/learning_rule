from pyentropy import DiscreteSystem
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path as path
import os, sys, pickle, shutil
import scipy.stats as st

# helper_function = path.abspath(path.join(__file__ ,"../.."))

# Parent directory
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from histograms_firing_rate_t_window import *

def add_noise(spkts_array, low, high, max_time, noise_amount = 0.1, shift = 'random'):
	altered_array = [0]*len(spkts_array)

	if low == 0 and high == 0:
		return np.array(spkts_array)
	else:
		for x in range(0, len(altered_array)):
			rand_perc = np.random.uniform(low = low, high = high)
			decision = np.random.uniform(low = 0.0, high = 1.0)

			decision2 = np.round(np.random.uniform(low = 0.0, high = 1.0), 3)

			target = spkts_array[x]
			perc = np.round((target*rand_perc)/100, 3)

			if shift == 'random':
				if decision2 <= noise_amount:
					if decision > 0.5:
						altered_array[x] = np.round((spkts_array[x] + perc), 3)
					else:
						altered_array[x] = np.round((spkts_array[x] - perc), 3)

					if (altered_array[x] > max_time) or (altered_array[x] < 0):
						altered_array[x] = spkts_array[x]
				else:
					altered_array[x] = spkts_array[x]
			elif shift == 'positive':
				if decision2 <= noise_amount:
					altered_array[x] = np.round((spkts_array[x] + perc), 3)
				else:
					altered_array[x] = spkts_array[x]

		final_array = np.unique(np.array(sorted(altered_array)))

		return final_array

# =========== generating spk times ===========

sim_time_s = float(sys.argv[1])

binned_spks_t_windos = int(sys.argv[2]) # ms

start_scope()

input_neurons = NeuronGroup(
	N = 2,
	model = 'rates : Hz', 
	threshold = 'rand()<rates*dt', 
	name = 'Input_to_I')

input_neurons.rates[range(0, 2)] = 150*Hz

spikemon = SpikeMonitor(input_neurons)

run(sim_time_s*second)

neuron1_spkts = []
neuron2_spkts = []

for x in range(0, len(spikemon.i)):
	if spikemon.i[x] == 0:
		neuron1_spkts.append(np.round(spikemon.t[x]/second, 3))
	else:
		neuron2_spkts.append(np.round(spikemon.t[x]/second, 3))

neuron1_spkts = np.array(neuron1_spkts)
neuron2_spkts = np.array(neuron2_spkts)

out_neuron_spkts = add_noise(
	spkts_array = neuron1_spkts,
	low = float(sys.argv[3]), 
	high = float(sys.argv[4]), 
	max_time = spikemon.t[x]/second,
	noise_amount = float(sys.argv[5]),
	shift = 'positive')

# =========== generating histograms ===========

hist_neuron1, bins_neuron1 = histograms_firing_rate_t_window(
	t_points = neuron1_spkts,
	sim_t = sim_time_s,
	t_window = binned_spks_t_windos)

binary_binned_spk_count_n1 = []
for x in range(0, len(hist_neuron1)):
	if hist_neuron1[x] > 0:
		binary_binned_spk_count_n1.append(1)
	else:
		binary_binned_spk_count_n1.append(0)

hist_neuron2, bins_neuron2 = histograms_firing_rate_t_window(
	t_points = neuron2_spkts,
	sim_t = sim_time_s,
	t_window = binned_spks_t_windos)

binary_binned_spk_count_n2 = []
for x in range(0, len(hist_neuron2)):
	if hist_neuron2[x] > 0:
		binary_binned_spk_count_n2.append(1)
	else:
		binary_binned_spk_count_n2.append(0)

hist_out_neuron, bins_out_neuron = histograms_firing_rate_t_window(
	t_points = out_neuron_spkts,
	sim_t = sim_time_s,
	t_window = binned_spks_t_windos)

binary_binned_spk_count_no = []
for x in range(0, len(hist_out_neuron)):
	if hist_out_neuron[x] > 0:
		binary_binned_spk_count_no.append(1)
	else:
		binary_binned_spk_count_no.append(0)

# =========== setup system and calculate entropies ===========
# generate random input
ds_n1_no = DiscreteSystem(
	binary_binned_spk_count_n1,
	(1, 2), 
	binary_binned_spk_count_no,
	(1, 2))

ds_n1_no.calculate_entropies(
	method = 'plugin', 
	calc = ['HX', 'HXY'])

ds_n2_no = DiscreteSystem(
	binary_binned_spk_count_n2,
	(1, 2), 
	binary_binned_spk_count_no,
	(1, 2))

ds_n2_no.calculate_entropies(
	method = 'plugin', 
	calc = ['HX', 'HXY'])

# =========== plotting ===========

axis_label_size = 6

fig0 = plt.figure(constrained_layout = True, figsize = (10, 4))

widths = [8, 2]
heights = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]

spec2 = gridspec.GridSpec(
	ncols = 2, 
	nrows = 6, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

# ----------------- input neuron 1 | spks -----------------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

y = [0]*len(neuron1_spkts)

plt.plot(neuron1_spkts, y, '|')

plt.ylabel('Neuron 1', size = axis_label_size)

plt.yticks([])

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

# ----------------- input neuron 1 | histogram -----------------
f2_ax2 = fig0.add_subplot(spec2[1, 0])

plt.hist(neuron1_spkts, bins = bins_neuron1, label = 'spks')

mn, mx = plt.xlim()
plt.xlim(mn, mx)

kde_xs = np.linspace(mn, mx, 300)
kde = st.gaussian_kde(neuron1_spkts)

plt.plot(
	kde_xs, 
	kde.pdf(kde_xs), 
	label = 'PDF', 
	color = 'darkred', 
	linewidth = 2,
	linestyle = '--')

f2_ax2.legend(prop = {'size': 8})

plt.ylabel('Spike Count', size = 6)

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

#----------------- output neuron -----------------
f2_ax3 = fig0.add_subplot(spec2[2, 0])

y = [0]*len(out_neuron_spkts)

plt.plot(out_neuron_spkts, y, 'k|')

plt.ylabel('Target', size = axis_label_size)

plt.yticks([])

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

# ----------------- output neuron | histogram -----------------
f2_ax4 = fig0.add_subplot(spec2[3, 0])

plt.hist(out_neuron_spkts, bins = bins_out_neuron, label = 'spks', color = 'k')

mn, mx = plt.xlim()
plt.xlim(mn, mx)

kde_xs = np.linspace(mn, mx, 300)
kde = st.gaussian_kde(out_neuron_spkts)

plt.plot(
	kde_xs, 
	kde.pdf(kde_xs), 
	label = 'PDF', 
	color = 'darkred', 
	linewidth = 2,
	linestyle = '--')

f2_ax4.legend(prop = {'size': 8})

plt.ylabel('Spike Count', size = 6)

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

# ----------------- input neuron 2 -----------------
f2_ax5 = fig0.add_subplot(spec2[4, 0])

y = [0]*len(neuron2_spkts)

plt.plot(neuron2_spkts, y, 'r|')

plt.ylabel('Neuron 2', size = axis_label_size)

plt.yticks([])
plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

# ----------------- input neuron 2 | histogram -----------------
f2_ax6 = fig0.add_subplot(spec2[5, 0])

plt.hist(neuron2_spkts, bins = bins_neuron2, label = 'spks', color = 'r')

mn, mx = plt.xlim()
plt.xlim(mn, mx)

kde_xs = np.linspace(mn, mx, 300)
kde = st.gaussian_kde(neuron2_spkts)

plt.plot(
	kde_xs, 
	kde.pdf(kde_xs), 
	label = 'PDF', 
	color = 'darkred', 
	linewidth = 2,
	linestyle = '--')

f2_ax6.legend(prop = {'size': 8})

plt.ylabel('Spike Count', size = 6)

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

plt.xlabel('Time (s)', size = axis_label_size)

# ----------------- Mutual Information / n1-no/n2->no -----------------
f2_ax7 = fig0.add_subplot(spec2[:, 1])

labels_array = ['n1', 'n2']

mi_array = [ds_n1_no.I(), ds_n2_no.I()]

barlist = plt.bar(labels_array, mi_array)

barlist[1].set_color('r')

plt.ylabel('MI (bits)', size = axis_label_size)

f2_ax7.set_ylim([0.0, 1.0])

plt.yticks(np.arange(
	0.0, 
	1.2,
	step = 0.2))

plt.show()