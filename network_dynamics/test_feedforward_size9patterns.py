# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
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
from separate_ids_tpoints_active_spont import *

def main():
	# ----------- Network Initialization -----------

	network = FeedforwardNetwork()

	network.N_e = 9
	network.N_e_outp = 2

	# Select single (test) stimulus
	network.stimulus_id = 'all'

	# Simulation
	network.exp_type = 'feedforward_network'
	network.exp_date = strftime("%d%b%Y_%H-%M-%S_", localtime())

	# Execution Parameters
	network.dt_resolution = 0.001*second 	# Delta t of clock intervals
	network.mon_dt = 0.001*second


	network.t_run = 5.0*second
	network.int_meth_neur = 'linear'
	network.int_meth_syn = 'euler'

	# Learning Rule
	network.plasticity_rule = 'LR3'
	network.parameter_set = '1.0'
	network.bistability = False

	# Neurons
	network.neuron_type = 'LIF'
	network.N_c = 1

	# Synaptic weights (max.)
	network.teacher_to_Eout_w = 40*mV 	# Teacher to Output - 40*mV
	network.I_to_Eout_w = 0*mV			# Inhibitory to Output - 20*mV

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	network.spont_to_input_w = 100*mV 	# Spontaneous to Input - 100*mV

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 75*Hz 	# Input pop. - 75*Hz
	network.stim_freq_teach = 200*Hz 	# Teacher pop. - 200*Hz/20*Hz
	network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 2*Hz
	network.stim_freq_i = 0*Hz		# Inhib. pop. - 100*Hz

	# Initializing network objects
	network.network_id = network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability)

	network.initialize_network_modules()
	
	network.set_weights()

	network.Input_to_Output.plastic = True

	print('\n================== network metadata ==================')
	print('active input (Hz/w) : ', network.stim_freq_Ninp, '/', network.w_max)
	print('spont. input (Hz/w) : ', network.stim_freq_spont, '/', network.w_max)
	print('teacher (Hz/w)      : ', network.stim_freq_teach, '/', network.teacher_to_Eout_w)
	print('inhibition (Hz/w)   : ', network.stim_freq_i, '/', network.I_to_Eout_w)
	print('\nmax. plastic weight : ', network.w_max)
	print('======================================================\n')

	for i in range(0, 2):
		if i == 0:
			network.stimulus_id = '+'
			network.update_teachers_rates(target_out = 0)
		else:
			network.stimulus_id = 'x'
			network.update_teachers_rates(target_out = 1)

		network.set_stimulus_Ninp()

		# update who's active/spontaneous in the input layer
		network.update_input_connectivity()
		
		network.run_net()

		network.export_syn_matrix()

if __name__ == "__main__":
	main()

	print("\n> test_feedforward_size9patterns.py - END\n")