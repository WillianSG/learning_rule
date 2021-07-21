# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- sys.argv[1] = simulation time (float)
- sys.argv[2] = total training epochs
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
	num_epochs = int(sys.argv[2])

	# ----------- Network Initialization -----------
	
	network = FeedforwardNetwork()

	# populations sizes
	network.N_e = 400
	network.N_e_outp = 2

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
	network.parameter_set = '1.C'
	network.bistability = True

	# Neurons
	network.neuron_type = 'LIF'
	network.N_c = 1

	# Synaptic weights (max.)
	network.teacher_to_Eout_w = 40*mV 	# Teacher to Output
	network.I_to_Eout_w = 40*mV			# Inhibitory to Output

	network.Input_to_Einp_w = 100*mV 	# 'virtual input' to Input
	network.Input_to_I_w = 100*mV 		# 'virtual inh.' to Inhibitory
	network.spont_to_input_w = 100*mV 	# Spontaneous to Input

	# Neuron populations mean frequency
	network.stim_freq_Ninp = 130*Hz 	# Input pop.
	network.stim_freq_teach = 300*Hz 	# Teacher pop.
	network.stim_freq_spont = 1*Hz 		# Spontaneous pop.
	network.stim_freq_i = 250*Hz		# Inhib. pop.

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

	sim_resul_final_path = os.path.join(sim_results, network.exp_date + '_' + network.plasticity_rule + '_' + network.parameter_set + '_bist' + str(network.bistability)) + '_epochs' + str(num_epochs) + '_secs' + str(network.t_run)

	if not(os.path.isdir(sim_resul_final_path)):
		os.mkdir(sim_resul_final_path)

	network.simulation_path = sim_resul_final_path

	# Storing network initial state
	network.net.store(name = network.network_id + '_initial_state', filename = os.path.join(network.simulation_path, network.network_id + '_initial_state'))

	total_sim_t = 0*second
	t_start = 0*second

	# ----------- Loading dataset -----------

	sim_data = '/home/p302242/PhD_codes/learning_rule/dataset_F/21Jul2021_13-09-12_dataset_Fusi-size_10.pickle'
	# sim_data = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\dataset_F\\12Jul2021_18-25-21_dataset_Fusi-size_10.pickle'

	with open(sim_data,'rb') as f:(
		meta_data,
		full_dataset) = pickle.load(f)

	# ----------- Loading patterns matrix summary -----------

	patterns_avg_filename = meta_data['timestamp'] +  '_dataset_size_' + str(meta_data['dataset_size']) + '_summed_patterns.pickle'

	# dataset_patterns_avg = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\dataset_F\\' + patterns_avg_filename
	dataset_patterns_avg = '/home/p302242/PhD_codes/learning_rule/dataset_F/' + patterns_avg_filename

	with open(dataset_patterns_avg,'rb') as f:(
		reshaped_c1,
		reshaped_c2) = pickle.load(f)

	# ----------- Simulation summary -----------

	print('\n================== network metadata ==================')
	print('active input (Hz/w) : ', network.stim_freq_Ninp, '/', network.w_max)
	print('spont. input (Hz/w) : ', network.stim_freq_spont, '/', network.w_max)
	print('teacher (Hz/w)      : ', network.stim_freq_teach, '/', network.teacher_to_Eout_w)
	print('inhibition (Hz/w)   : ', network.stim_freq_i, '/', network.I_to_Eout_w)
	print('\nmax. plastic weight : ', network.w_max)
	print('\nnum. input neurons  : ', network.N_e)
	print('num. output neurons : ', network.N_e_outp)
	print('\nt run               : ', network.t_run)
	print('======================================================\n')

	print('================== dataset metadata ==================')
	for key, value in meta_data.items():
		print(key, ':', value)
	print('======================================================\n')

	network.export_syn_matrix(name = 'initial')

	# ----------- Training -----------

	print('\n====================== training ======================')

	network.Input_to_Output.plastic = True

	opt_counter = 0
	for epoch in range(1, num_epochs+1):
		
		epoch_ids_list = make_ids_traning_list(
			dataset_size = meta_data['dataset_size'],
			epoch = epoch)

		print(' \nepoch #', epoch, ' (', len(epoch_ids_list), ' presentations)')

		for pattern_id in epoch_ids_list:
			# 1 - select next pattern to be presented
			network.set_stimulus_dataset(full_dataset[pattern_id])

			if (pattern_id % 2) == 0:
				network.update_teachers_rates(target_out = 0)
				target_out = 0
			else:
				network.update_teachers_rates(target_out = 1)
				target_out = 1

			print(' -> pattern ', pattern_id, ' (', total_sim_t, ')', ' | target output: ', target_out)

			# 2 - update who's active/spontaneous in the input layer
			network.update_input_connectivity()

			# 3 - simulate
			network.run_net(report = None)

			total_sim_t += network.t_run

			opt_counter += 1

			# network.export_syn_matrix(name = 'training', opt = '_' + str(opt_counter) + '_', class1 = reshaped_c1, class2 = reshaped_c2)

	# ----------- Finalizing Training (saving network state) -----------

	# 6 - binarize weights based on synaptic internal state variable
	network.binarize_syn_matrix()

	network.export_syn_matrix(name = 'trained_withClasses_', class1 = reshaped_c1, class2 = reshaped_c2)

	network.export_syn_matrix(name = 'trained')

	# 6.1 - turning plasticity OFF for testing
	network.Input_to_Output.plastic = False

	# 7 - save trained network state
	network.net.store(name = network.network_id + '_trained', filename = os.path.join(network.simulation_path, network.network_id + '_trained'))

	print('======================================================\n')

	# ----------- Testing trained network -----------
	print('\n====================== testing =======================')

	# 0 - index represents pattern id, value represents output active neuron
	out_winning_response_per_pattern = []

	presentation_time = float(sys.argv[1])*second
	# presentation_time = 2*second
	# network.t_run = 2*second

	correct_response = 0
	wrong_response = 0

	# 1 - restoring trained network state
	network.net.restore(name = network.network_id + '_trained', filename = os.path.join(network.simulation_path, network.network_id + '_trained'))

	# 2 - silencing auxiliary populations
	network.silince_for_testing()

	# 3 - testing learned patterns
	for pattern_id in range(0, meta_data['dataset_size']):
		print(' -> pattern ', pattern_id, ' (', total_sim_t, ')')
		
		# setting stimulus to be presented
		network.set_stimulus_dataset(full_dataset[pattern_id])

		# update who's active/spontaneous in the input layer
		network.update_input_connectivity()

		# simulating
		network.run_net(report = None)

		total_sim_t += network.t_run

		# ----------- saving output neuron responding to pattern -----------

		# =============================================================

		# 1 - Where in spkmon current stimulus response starts
		t_start = total_sim_t - presentation_time

		# 2 - Gets spikes times for each of the output neurons
		"""
		output_spks_t[0] = spikes times of output neuron 0
		output_spks_t[1] = spikes times of output neuron 1
		...
		"""
		output_spks_t = network.get_out_neurons_spks_t(start = t_start)

		mean_activity_output_neurons = []

		# 3 - Calc. outputs firing freqs.
		for out_n in range(0, network.N_e_outp):
			[t_hist_edges,
			t_hist_freq, 
			t_hist_bin_widths] = histograms_firing_rate(
				t_points = output_spks_t[out_n], 
				pop_size = 1)

			mean_freq = np.mean(t_hist_freq)

			# 3.1 - saving mean activity
			mean_activity_output_neurons.append(mean_freq)

		# 4 - Who's firing the most
		max_freq = max(mean_activity_output_neurons)
		max_neuro_id = mean_activity_output_neurons.index(max_freq)

		# 5 - Saving winning neuron
		out_winning_response_per_pattern.append(max_neuro_id)

		if ((pattern_id % 2) == 0) and (max_neuro_id == 0):
				correct_response += 1
		elif ((pattern_id % 2) != 0) and (max_neuro_id == 1):
			correct_response += 1
		else:
			wrong_response += 1

		# =============================================================

	print('======================================================\n')

	print('\n\ncorrect responses: ', correct_response)
	print('wrong responses: ', wrong_response)

if __name__ == "__main__":
	main()

	print("\nfeedforward_net.py - END\n")