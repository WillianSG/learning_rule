# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- sys.argv[1] = simulation time (float)
- sys.argv[2] = 1 : save sim data
- parameters set considering active input of 45 neurons
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

def main():
	make_dir = str(sys.argv[2])
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
	network.w_max = 5*mV				# Input to Output - 5*mV

	network.teacher_to_Eout_w = 30*mV 	# Teacher to Output - 30*mV
	network.I_to_Eout_w = 20*mV			# Inhibitory to Output - 20*mV

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	network.spont_to_input_w = 25*mV 	# Spontaneous to Input - 25*mV

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 200*Hz 	# Input pop. - 200*Hz
	network.stim_freq_teach = 180*Hz 	# Teacher pop. - 400*Hz/180*Hz
	network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 20*Hz
	network.stim_freq_i = 100*Hz		# Inhib. pop. - 100*Hz

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
	if make_dir == '1':
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
	if make_dir == '1':
		network.net.store(name = network.network_id + '_initial_state', filename = os.path.join(network.simulation_path, network.network_id + '_initial_state'))

	t_run = 0*second
	t_start = 0*second

	# ----------- Loading dataset -----------

	sim_data = '/home/p302242/PhD_codes/learning_rule/dataset_F/01Jun2021_17-32-36_dataset_Fusi-size_100.pickle'

	with open(sim_data,'rb') as f:(
		meta_data,
		full_dataset) = pickle.load(f)

	print('\n> dataset (meta): ', meta_data)

	# ----------- Simulation -----------

	for exposure_n in range(0, 1):
		print('\nrun ', exposure_n)
		
		network.set_stimulus_dataset(full_dataset[0])

		network.Input_to_Output.plastic = False

		network.update_params_datasetclass(pattern_id = 1)

		network.run_net()

		# ----------- Storing simulation data -----------
		# Input_to_E_inp
		s_tpoints_Input_to_Einp = network.Input_to_E_inp_spkmon.t[:]
		n_inds_Input_to_Einp = network.Input_to_E_inp_spkmon.i[:]
		# print('\n# spks from input to 1st layer: ', len(s_tpoints_Input_to_Einp))
		# print(n_inds_Input_to_Einp)

		# E_inp
		s_tpoints_E_inp = network.E_inp_spkmon.t[:]
		n_inds_E_inp = network.E_inp_spkmon.i[:]
		# print('# spks 1st layer: ', len(s_tpoints_E_inp))

		# E_outp
		s_tpoints_E_outp = network.E_outp_spkmon.t[:]
		n_inds_E_outp = network.E_outp_spkmon.i[:]
		# print('# spks out layer: ', len(s_tpoints_E_outp))

		# Input_to_I
		s_tpoints_Input_to_I = network.Input_to_I_spkmon.t[:]
		n_inds_Input_to_I = network.Input_to_I_spkmon.i[:]
		# print('input to I pop: ', len(s_tpoints_Input_to_I))

		# I
		s_tpoints_I = network.I_spkmon.t[:]
		n_inds_I = network.I_spkmon.i[:]
		# print('# spks I pop: ', len(s_tpoints_I))

		# teacher
		s_tpoints_teach = network.teacher_spkmon.t[:]
		n_inds_teach = network.teacher_spkmon.i[:]
		# print('# spks teacher pop: ', len(s_tpoints_teach))

		# spontaneous
		s_tpoints_spont = network.spont_spkmon.t[:]
		n_inds_spont = network.spont_spkmon.i[:]
		# print('spont pop: ', len(s_tpoints_spont))

		sim_id = network.network_id
		path_sim = network.simulation_path


		t_run += network.t_run
		t_start += t_run - network.t_run


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

		if make_dir == '1':
			fn = os.path.join(network.simulation_path, network.network_id + '_' + network.exp_type + '_expos' + str(exposure_n) + '.pickle')

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

			plot_feedforwad_net(
				network_state_path = network.simulation_path,
				pickled_data = network.network_id + '_' + network.exp_type + '_expos' + str(exposure_n) + '.pickle',
				exposure_n = exposure_n,
				t_start = t_start)

			feedforward_plot_activity(
				sim_id = network.network_id, 
				path_sim = network.simulation_path, 
				t_run = t_run, 
				rho_matrix = network.Input_to_Output_stamon.rho, 
				time_arr = network.Input_to_Output_stamon.t[:],
				w_matrix = network.Input_to_Output_stamon.w, 
				time_arr_w = network.Input_to_Output_stamon.t[:],
				eout_mon = network.Eout_stamon.Vm,
				eout_time_arr = network.Eout_stamon.t[:],
				stim_ids = network.stimulus_ids_Ninp,
				exposure_n = exposure_n,
				t_start = t_start)

if __name__ == "__main__":
	main()

	print("\n> feedforward_net.py - END\n")