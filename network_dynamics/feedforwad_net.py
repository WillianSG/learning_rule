# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- sys.argv[1] = simulation time (float)
- sys.argv[2] = 1 : save sim data
- sys.argv[3] = total training epochs
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
from histograms_firing_rate import *

def main():
	# ----------- Simulation parameters -----------
	make_dir = str(sys.argv[2])

	num_epochs = int(sys.argv[3])

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
	network.teacher_to_Eout_w = 50*mV 	# Teacher to Output - 50*mV
	network.I_to_Eout_w = 40*mV			# Inhibitory to Output - 40*mV

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input - 100*mV
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory - 100*mV

	network.spont_to_input_w = 100*mV 	# Spontaneous to Input - 100*mV

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 65*Hz 	# Input pop. - 65*Hz
	network.stim_freq_teach = 20*Hz 	# Teacher pop. - 300*Hz/20*Hz
	network.stim_freq_spont = 20*Hz 	# Spontaneous pop. - 20*Hz
	network.stim_freq_i = 20*Hz		# Inhib. pop. - 20*Hz

	# Initializing network objects
	network.network_id = network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability)

	network.initialize_network_modules()

	network.set_weights()

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

	total_sim_t = 0*second
	t_start = 0*second

	# ----------- Loading dataset -----------

	# sim_data = '/home/p302242/PhD_codes/learning_rule/dataset_F/12Jul2021_12-40-29_dataset_Fusi-size_2.pickle'
	sim_data = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\dataset_F\\12Jul2021_12-40-29_dataset_Fusi-size_2.pickle'

	with open(sim_data,'rb') as f:(
		meta_data,
		full_dataset) = pickle.load(f)

	print('\n================== dataset metadata ==================')
	for key, value in meta_data.items():
		print(key, ':', value)
	print('======================================================\n')


	# ----------- Training -----------

	print('\n> training network....')

	network.Input_to_Output.plastic = True

	for epoch in range(1, num_epochs+1):
		
		epoch_ids_list = make_ids_traning_list(
			dataset_size = meta_data['dataset_size'],
			epoch = epoch)

		print(' \nepoch #', epoch, ' (', len(epoch_ids_list), ' presentations)')

		for pattern_id in epoch_ids_list:			
			# 1 - select next pattern to be presented
			network.set_stimulus_dataset(full_dataset[pattern_id])

			print(' -> pattern ', pattern_id, ' (', total_sim_t, ')')

			# 2- update teacher signal based on pattern class
			network.update_params_datasetclass(pattern_id = pattern_id+1)

			# 3 - update who's active/spontaneous in the input layer
			network.update_input_connectivity()

			# 4 - simulate
			network.run_net(report = None)

			total_sim_t += network.t_run

			# 5 - (optional) plot simulation data
			if make_dir == '1':
				# # ----------- Storing simulation data -----------
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

				fn = os.path.join(network.simulation_path, network.network_id + '_' + network.exp_type + '_expos' + str(pattern_id) + '.pickle')

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
					pickled_data = network.network_id + '_' + network.exp_type + '_expos' + str(pattern_id) + '.pickle',
					exposure_n = pattern_id,
					t_start = t_start)

				feedforward_plot_activity(
					sim_id = network.network_id, 
					path_sim = network.simulation_path, 
					t_run = total_sim_t, 
					rho_matrix = network.Input_to_Output_stamon.rho, 
					time_arr = network.Input_to_Output_stamon.t[:],
					w_matrix = network.Input_to_Output_stamon.w, 
					time_arr_w = network.Input_to_Output_stamon.t[:],
					eout_mon = network.Eout_stamon.Vm,
					eout_time_arr = network.Eout_stamon.t[:],
					stim_ids = network.stimulus_ids_Ninp,
					exposure_n = pattern_id,
					t_start = t_start)

	# ----------- Finalizing Training (saving network state) -----------

	# 6 - binarize weights based on synaptic internal state variable
	network.w_trained_binarize()

	# 6.1 - turning plasticity OFF for testing
	network.Input_to_Output.plastic = False

	# 7 - save trained network state
	network.net.store(name = network.network_id + '_trained', filename = os.path.join(network.simulation_path, network.network_id + '_trained'))

	# ----------- Testing trained network -----------

	print('\n> testing trained network...\n')

	presentation_time = float(sys.argv[1])*second

	mean_activity_c1 = []
	mean_activity_c2 = []

	# 0 - test results destination directory
	test_data_path_c1 = os.path.join(sim_results, network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability), 'class_1')
	
	if not(os.path.isdir(test_data_path_c1)):
		os.mkdir(test_data_path_c1)

	test_data_path_c2 = os.path.join(sim_results, network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability), 'class_2')
	
	if not(os.path.isdir(test_data_path_c2)):
		os.mkdir(test_data_path_c2)

	# 1 - restoring trained network state
	network.net.restore(name = network.network_id + '_trained', filename = os.path.join(network.simulation_path, network.network_id + '_trained'))

	# 2 - silencing auxiliary populations
	network.silince_for_testing()

	# 3 - testing learned patterns
	for pattern_id in range(0, meta_data['dataset_size']):
		print(' -> pattern ', pattern_id+1, ' (', total_sim_t, ')')
		
		# setting stimulus to be presented
		network.set_stimulus_dataset(full_dataset[pattern_id])

		# update who's active/spontaneous in the input layer
		network.update_input_connectivity()

		# simulating
		network.run_net(report = None)

		total_sim_t += network.t_run

		# ----------- Output monitor data -----------
		# spike times
		temp_output_spks = network.E_outp_spkmon.t[:]

		# where in spkmon current stimulus response starts
		t_start = total_sim_t - presentation_time

		# retrieving response to stimulus
		output_spks = temp_output_spks[temp_output_spks >= t_start]

		# firing frequency histogram data
		[t_hist_edges,
		t_hist_freq, 
		t_hist_bin_widths] = histograms_firing_rate(
			t_points = output_spks, 
			pop_size = 1)

		# simulation time array
		temp_sim_t_array = network.Input_to_Output_stamon.t
		sim_t_array = temp_sim_t_array[temp_sim_t_array >= t_start]

		# ----------- Plotting output activity -----------
		if ((pattern_id+1) % 2) == 0:
			mean_activity_c1.append(np.round(np.mean(t_hist_freq), 1))

			img_name = os.path.join(test_data_path_c1, 'pattern_' + str(pattern_id) + '_c1.png')

			plt.bar(
				x = t_hist_edges,
				height = t_hist_freq,
				width = t_hist_bin_widths,
				color = 'lightblue',
				edgecolor = 'k',
				linewidth = 0.5)

			plt.title('Output response: pattern ' + str(pattern_id) + ' | class 1')
		else:
			mean_activity_c2.append(np.round(np.mean(t_hist_freq), 1))

			img_name = os.path.join(test_data_path_c2, 'pattern_' + str(pattern_id) + '_c2.png')

			plt.bar(
				x = t_hist_edges,
				height = t_hist_freq,
				width = t_hist_bin_widths,
				color = 'tomato',
				edgecolor = 'k',
				linewidth = 0.5)

			plt.title('Output response: pattern ' + str(pattern_id) + ' | class 2')

		plt.xlim([t_start, total_sim_t])

		plt.ylabel('freq (Hz)', size = 6)
		plt.xlabel('time (s)', size = 6)

		mean_activity = np.round(np.mean(t_hist_freq), 1)

		plt.hlines(mean_activity, t_start, total_sim_t, color = 'k', linestyle = '--', label = 'avg frequency', linewidth = 0.5)

		plt.legend(prop = {'size': 6})

		plt.savefig(img_name)
		plt.close()

	# saving simulation data
	fn = os.path.join(network.simulation_path, network.network_id + '_' + network.exp_type + '_simulation_metadata_n_results.pickle')

	with open(fn, 'wb') as f:
		pickle.dump((
			network.simulation_path,
			network.network_id,
			total_sim_t,
			network.dt_resolution,
			network.mon_dt,
			network.int_meth_neur,
			network.int_meth_syn,
			network.plasticity_rule,
			network.parameter_set,
			network.bistability,
			num_epochs,
			meta_data, # 'meta_data' is the dataset metadata
			mean_activity_c1,
			mean_activity_c2), f)

if __name__ == "__main__":
	main()

	print("\nfeedforward_net.py - END\n")