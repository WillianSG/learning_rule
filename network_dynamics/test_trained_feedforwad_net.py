# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- sys.argv[1] = simulation time (float)
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
from make_ids_training_list import *

def main():
	print('\n> testing trained network....\n')

	# 1 - getting trained network simulation metadata
	sim_data = '/home/p302242/PhD_codes/learning_rule/network_results/feedforward_network/09Jul2021_17-20-33__LR3_1.0_bistTrue/09Jul2021_17-20-33__LR3_1.0_bistTrue_feedforward_network_simulation_metadata.pickle'

	with open(sim_data,'rb') as f:(
		simulation_path,
		network_id,
		t_run,
		dt_resolution,
		mon_dt,
		int_meth_neur,
		int_meth_syn,
		plasticity_rule,
		parameter_set,
		bistability,
		num_epochs,
		meta_data) = pickle.load(f) # 'meta_data' is the dataset metadata

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
	# network.teacher_to_Eout_w = 50*mV 	# Teacher to Output - 50*mV
	# network.I_to_Eout_w = 40*mV			# Inhibitory to Output - 40*mV

	# network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	# network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	# network.spont_to_input_w = 100*mV 	# Spontaneous to Input - 100*mV

	# # Neuron populations mean frequency
	# network.stim_freq_Ninp = 65*Hz 	# Input pop. - 65*Hz
	# network.stim_freq_teach = 20*Hz 	# Teacher pop. - 300*Hz/20*Hz
	# network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 20*Hz
	# network.stim_freq_i = 20*Hz		# Inhib. pop. - 20*Hz

	# Initializing network objects
	network.network_id = network_id

	network.initialize_network_modules()

	network.set_weights()

	network.net.restore(name = network_id + '_trained', filename = os.path.join(simulation_path, network_id + '_trained'))

	print('restored')

	for x in range(0, 2):
		network.net.restore(name = network_id + '_trained', filename = os.path.join(simulation_path, network_id + '_trained'))

		network.silince_for_testing()
		network.Input_to_Output.plastic = False

		print('pattern_id:  ', x+1)
	
		network.set_stimulus_dataset(full_dataset[x])

		network.run_net(report = None)

		# ----------- Storing simulation data -----------
		# Input_to_E_inp
		s_tpoints_Input_to_Einp = network.Input_to_E_inp_spkmon.t[:]
		n_inds_Input_to_Einp = network.Input_to_E_inp_spkmon.i[:]

		# E_inp
		s_tpoints_E_inp = network.E_inp_spkmon.t[:]
		n_inds_E_inp = network.E_inp_spkmon.i[:]

		# E_outp
		s_tpoints_E_outp = network.E_outp_spkmon.t[:]
		n_inds_E_outp = network.E_outp_spkmon.i[:]

		# Input_to_I
		s_tpoints_Input_to_I = network.Input_to_I_spkmon.t[:]
		n_inds_Input_to_I = network.Input_to_I_spkmon.i[:]

		# I
		s_tpoints_I = network.I_spkmon.t[:]
		n_inds_I = network.I_spkmon.i[:]

		# teacher
		s_tpoints_teach = network.teacher_spkmon.t[:]
		n_inds_teach = network.teacher_spkmon.i[:]

		# spontaneous
		s_tpoints_spont = network.spont_spkmon.t[:]
		n_inds_spont = network.spont_spkmon.i[:]

		sim_id = network.network_id
		path_sim = network.simulation_path


		t_run += network.t_run
		t_start = t_run - network.t_run

		exp_type = network.exp_type

		plasticity_rule = network.plasticity_rule
		parameter_set = network.parameter_set
		bistability = network.bistability

		stim_type = network.stimulus_id
		stim_size = network.stim_size
		stim_freq = network.stim_freq_Ninp*Hz
		stim_freq_i = network.coding_lvl*Hz

		len_stim_inds_original_E = network.stim_size

		n_Eoutp = network.N_e_outp
		n_Einp = network.N_e
		n_I = network.N_e_outp
		n_pool = network.N_c

		fn = os.path.join(network.simulation_path, network.network_id + '_' + network.exp_type + '_test_expos' + str(x+1) + '.pickle')

		with open(fn, 'wb') as f:
			pickle.dump((
				path_sim,
				sim_id,
				t_run,
				exp_type,
				stim_type,
				stim_size,
				stim_freq,
				stim_freq_i,
				n_Eoutp,
				n_Einp,
				n_I,
				n_pool,
				len_stim_inds_original_E,
				s_tpoints_Input_to_Einp,
				n_inds_Input_to_Einp,
				s_tpoints_E_inp,
				n_inds_E_inp,
				s_tpoints_E_outp,
				n_inds_E_outp,
				s_tpoints_Input_to_I,
				n_inds_Input_to_I,
				s_tpoints_I,
				n_inds_I,
				s_tpoints_teach,
				n_inds_teach
				), f)

if __name__ == "__main__":
	main()

	print("\n> test_trained_feedforward_net.py - END\n")