# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- sys.argv[1] = simulation time (float)
- sys.argv[2] = number of simulation repetitions
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
import numpy as np
from time import localtime, strftime

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'
dataset_dir = 'dataset_F'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))
sys.path.append(os.path.join(parent_dir, dataset_dir))

# Helper modules
from feedforward_snn import FeedforwardNetwork
from visualise_connectivity import visualise_connectivity
from plot_feedforwad_net import *
from feedforward_plot_activity import *

def main():
	sim_repetitions = int(sys.argv[2])
	# ----------- Network Initialization -----------

	network = FeedforwardNetwork()

	# Select single (test) stimulus
	network.stimulus_id = 'all'

	# Simulation
	network.exp_type = 'feedforward_network'
	network.exp_date = strftime("%d%b%Y_%H-%M-%S_", localtime())

	# Execution Parameters
	network.dt_resolution = 0.001*second 	# Delta t of clock intervals
	network.mon_dt = 0.001*second


	network.t_run = float(sys.argv[1])*second
	network.int_meth_neur = 'linear'
	network.int_meth_syn = 'euler'

	# Learning Rule
	network.plasticity_rule = 'LR3'
	network.parameter_set = '1.0'
	network.bistability = True

	# Neurons
	network.neuron_type = 'LIF'
	network.N_c = 1

	# Synaptic weights (max.)
	network.w_max = 10*mV				# Input to Output - 5*mV

	network.teacher_to_Eout_w = 100*mV 	# Teacher to Output - 30*mV
	network.I_to_Eout_w = 20*mV			# Inhibitory to Output - 20*mV

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	network.spont_to_input_w = 100*mV 	# Spontaneous to Input - 100*mV

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 80*Hz 	# Input pop. - 75*Hz
	network.stim_freq_teach = 0*Hz 	# Teacher pop. - 400*Hz/180*Hz
	network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 2*Hz
	network.stim_freq_i = 0*Hz		# Inhib. pop. - 100*Hz

	# Initializing network objects
	network.network_id = network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability)

	network.initialize_network_modules()

	network.set_weights()

	# ----------- Network Connectivity -----------	
	# visualise_connectivity(network.Input_1st_layer)
	# visualise_connectivity(network.Input_I)
	# visualise_connectivity(network.Input_to_Output)
	# visualise_connectivity(network.I_Eout)

	# ----------- Results Directories -----------

	# Results Directories
	# if make_dir == '1':
	results_dir = os.path.join(parent_dir, 'network_results')
	if not(os.path.isdir(results_dir)):
		os.mkdir(results_dir)

	sim_results = os.path.join(results_dir,	network.exp_type)
	if not(os.path.isdir(sim_results)):
		os.mkdir(sim_results)

	sim_resul_final_path = os.path.join(sim_results, network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability))
	if not(os.path.isdir(sim_resul_final_path)):
		os.mkdir(sim_resul_final_path)

	network.simulation_path = sim_resul_final_path

	# Storing network initial state
	network.net.store(name = network.network_id + '_initial_state', filename = os.path.join(network.simulation_path, network.network_id + '_initial_state'))

	# ----------- Loading dataset -----------

	sim_data = '/home/p302242/PhD_codes/learning_rule/dataset_F/01Jun2021_17-32-36_dataset_Fusi-size_100.pickle'

	with open(sim_data,'rb') as f:(
		meta_data,
		full_dataset) = pickle.load(f)

	# ----------- Training -----------

	network.Input_to_Output.plastic = False

	network.set_stimulus_dataset(full_dataset[0]) # 1st pattern

	# arrays for avgs
	rho_all = []
	xpost_all = []
	xpre_all = []

	for exposure_n in range(0, sim_repetitions):
		network.net.restore(name = network.network_id + '_initial_state', filename = os.path.join(network.simulation_path, network.network_id + '_initial_state'))

		network.randomize_synaptic_weights() # ATTENTION - make sure r random

		# virtual input drives only active neurons
		network.update_input_connectivity()

		network.run_net()

		# # ----------- Storing simulation data -----------
		rho_all.append(network.Input_to_Output_stamon.rho)
		xpost_all.append(network.Input_to_Output_stamon.xpost)
		xpre_all.append(network.Input_to_Output_stamon.xpre)


	# ----------- Xpre/Xpost avgs -----------

	xpre_active_avg = np.zeros(len(xpre_all[0][0]))
	xpre_inactive_avg = np.zeros(len(xpre_all[0][0]))
	
	xpost_avg = np.zeros(len(xpost_all[0][0]))

	count1 = 0
	count2 = 0

	for a in range(0, sim_repetitions):

		xpost_avg += xpost_all[a][0]

		for b in range(0, network.N_e):
			if b in network.stimulus_ids_Ninp:
				xpre_active_avg += xpre_all[a][b]
			else:
				xpre_inactive_avg += xpre_all[a][b]

	# avg Ca of activated neurons in the pattern
	xpre_active_avg = (xpre_active_avg/len(network.stimulus_ids_Ninp))/sim_repetitions

	# avg Ca of inactive (spontaneous) neurons in the pattern
	xpre_inactive_avg = (xpre_inactive_avg/(network.N_e-len(network.stimulus_ids_Ninp)))/sim_repetitions

	# avg Ca of (single) output neuron
	xpost_avg = xpost_avg/sim_repetitions

	# simulation time array
	sim_t_array = network.Input_to_Output_stamon.t

	# ----------- Plotting -----------

	fig0 = plt.figure(constrained_layout = True)
	spec2 = gridspec.GridSpec(ncols = 2, nrows = 1, figure = fig0)

	fig0.suptitle('rule ' + network.plasticity_rule + '/param. ' + network.parameter_set, fontsize = 8)

	# avg Ca2+ pre -----------
	f2_ax1 = fig0.add_subplot(spec2[0, 0])

	plt.plot(sim_t_array, xpre_active_avg, color = 'tomato', linestyle = 'solid', label = r'$Ca^{2+}_{act.}$')

	plt.plot(sim_t_array, xpre_inactive_avg, color = 'grey', linestyle = 'solid', label = r'$Ca^{2+}_{inact.}$')

	plt.hlines(network.thr_pre, 0, sim_t_array[-1], color = 'k', linestyle = '--', label = '$\\theta_{pre}$')

	f2_ax1.legend(prop = {'size': 8})

	f2_ax1.set_ylim([0.0, 1.0])

	plt.ylabel(r'$Ca^{2+}$' + ' (a.u.)', size = 6)
	plt.xlabel('time (s)', size = 6)

	# avg Ca2+ post -----------
	f2_ax2 = fig0.add_subplot(spec2[0, 1])

	plt.plot(sim_t_array, xpost_avg, color = 'lightblue', linestyle = 'solid', label = r'$Ca^{2+}_{output}$')

	plt.hlines(network.thr_post, 0, sim_t_array[-1], color = 'k', linestyle = '--', label = '$\\theta_{post}$')

	f2_ax2.legend(prop = {'size': 8})

	f2_ax2.set_ylim([0.0, 1.0])

	plt.ylabel(r'$Ca^{2+}$' + ' (a.u.)', size = 6)
	plt.xlabel('time (s)', size = 6)

	plt.show()

if __name__ == "__main__":
	main()

	print("\n> feedforward_net.py - END\n")