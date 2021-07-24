from pyentropy import DiscreteSystem
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

def add_noise(spkts_array, low, high, max_time, shift = 'random'):
	altered_array = [0]*len(spkts_array)

	for x in range(0, len(altered_array)):
		rand_perc = np.random.uniform(low = low, high = high)
		decision = np.random.uniform(low = 0.0, high = 1.0)

		target = spkts_array[x]
		perc = np.round((target*rand_perc)/100, 3)

		if shift == 'random':
			if decision > 0.5:
				altered_array[x] = np.round((spkts_array[x] + perc), 3)
			else:
				altered_array[x] = np.round((spkts_array[x] - perc), 3)

			if (altered_array[x] > max_time) or (altered_array[x] < 0):
				altered_array[x] = spkts_array[x]
		elif shift == 'positive':
			altered_array[x] = np.round((spkts_array[x] + perc), 3)

	final_array = np.unique(np.array(sorted(altered_array)))

	return final_array

# =========== generating spk times ===========

sim_time_s = 1.0

start_scope()

input_neurons = NeuronGroup(
	N = 2,
	model = 'rates : Hz', 
	threshold = 'rand()<rates*dt', 
	name = 'Input_to_I')

input_neurons.rates[range(0, 2)] = 100*Hz

spikemon = SpikeMonitor(input_neurons)

run(sim_time_s*second)

# plt.plot(spikemon.t/second, spikemon.i, '|')
# plt.xlabel('Time (s)')
# plt.ylabel('Neuron index')

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
	low = 1, 
	high = 1, 
	max_time = spikemon.t[x]/second,
	shift = 'positive')

# =========== generating spk times ===========


# =========== plotting ===========

axis_label_size = 6

fig0 = plt.figure(constrained_layout = True, figsize = (8, 3))

widths = [8]
heights = [0.1, 0.1, 0.1]

spec2 = gridspec.GridSpec(
	ncols = 1, 
	nrows = 3, 
	width_ratios = widths,
	height_ratios = heights,
	figure = fig0)

# ----------------- input neuron 0 -----------------
f2_ax1 = fig0.add_subplot(spec2[0, 0])

y = [0]*len(neuron1_spkts)

plt.plot(neuron1_spkts, y, '|')

plt.ylabel('Neuron 0', size = axis_label_size)

plt.yticks([])

plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

# ----------------- input neuron 1 -----------------
f2_ax2 = fig0.add_subplot(spec2[1, 0])

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

# ----------------- input neuron 1 -----------------
f2_ax3 = fig0.add_subplot(spec2[2, 0])

y = [0]*len(neuron2_spkts)

plt.plot(neuron2_spkts, y, 'r|')

plt.ylabel('Neuron 1', size = axis_label_size)

plt.yticks([])
plt.xticks(np.arange(
	0.0, 
	sim_time_s+0.2,
	step = 0.2))
plt.xlim(0.0, sim_time_s)

plt.xlabel('Time (s)', size = axis_label_size)




# =========== setup system and calculate entropies ===========
# # generate random input
# x = np.random.randint(0, 2, 10000)

# percent = 2

# # corrupt half of output
# y = x.copy()
# indx = np.random.permutation(len(x))[:int(len(x)/percent)]
# y[indx] = np.random.randint(0, 2, int(len(x)/percent))

# s = DiscreteSystem(x,(1, 2), y,(1, 2))

# s.calculate_entropies(
# 	method = 'plugin', 
# 	calc = ['HX', 'HXY'])

# print('\n\n\nMI: ', s.I())

plt.show()