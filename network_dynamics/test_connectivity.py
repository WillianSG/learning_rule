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

	# Select single (test) stimulus
	network.stimulus_id = 'all'

	# Simulation
	network.exp_type = 'feedforward_network'
	network.exp_date = strftime("%d%b%Y_%H-%M-%S_", localtime())

	# Execution Parameters
	network.dt_resolution = 0.001*second 	# Delta t of clock intervals
	network.mon_dt = 0.001*second


	network.t_run = 0.05*second
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
	network.w_max = 10*mV				# Input to Output - 5*mV

	network.teacher_to_Eout_w = 100*mV 	# Teacher to Output - 30*mV
	network.I_to_Eout_w = 20*mV			# Inhibitory to Output - 20*mV

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	network.spont_to_input_w = 100*mV 	# Spontaneous to Input - 100*mV

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 75*Hz 	# Input pop. - 75*Hz
	network.stim_freq_teach = 0*Hz 	# Teacher pop. - 40*Hz/20*Hz
	network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 2*Hz
	network.stim_freq_i = 0*Hz		# Inhib. pop. - 100*Hz

	# Initializing network objects
	network.network_id = network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability)

	network.initialize_network_modules()
	network.set_weights()

	# visualise_connectivity(network.Input_1st_layer)

	network.stimulus_id = '.'

	network.set_stimulus_Ninp()
	network.run_net()

	# E_inp
	s_tpoints_E_inp = network.E_inp_spkmon.t[:]
	n_inds_E_inp = network.E_inp_spkmon.i[:]

	# E_outp
	s_tpoints_E_outp = network.E_outp_spkmon.t[:]
	n_inds_E_outp = network.E_outp_spkmon.i[:]

	print(s_tpoints_E_inp)
	print(n_inds_E_inp, '\n')

	[active_tpoints, 
	active_ids, 
	spontaneous_tpoints, 
	spontaneous_ids] = separate_ids_tpoints_active_spont(
		input_tpoints = s_tpoints_E_inp, 
		input_ids = n_inds_E_inp, 
		active_input_ids = network.stimulus_ids_Ninp)

	print('active \n', active_tpoints, '\n', active_ids, '\n')
	print('inactive \n', spontaneous_tpoints, '\n', spontaneous_ids, '\n')

	

if __name__ == "__main__":
	main()

	print("\n> feedforward_net.py - END\n")