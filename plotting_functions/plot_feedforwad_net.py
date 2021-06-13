# -*- coding: utf-8 -*-
"""
@author: wgirao

Inputs:

Outputs:

Comments:
"""

from brian2 import ms, mV, second
import os, sys, pickle
import numpy as np

from feedforward_plot_spktrains_histograms import *

def plot_feedforwad_net(network_state_path = '', 
	pickled_data = '', 
	exposure_n = 0,
	t_start = 0*second):

	sim_data = os.path.join(network_state_path, pickled_data) # loading data

	with open(sim_data,'rb') as f:(
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
		n_inds_teach) = pickle.load(f)

	print('\n -> plotting spiketrains and histograms\n')

	feedfoward_plot_spktrains_histograms(
		sim_id = sim_id,
		path_sim = path_sim,
		stim_size = stim_size,
		N = [n_Einp, n_Eoutp, n_I, n_pool],
		s_tpoints = [s_tpoints_Input_to_Einp, s_tpoints_E_inp, s_tpoints_E_outp, s_tpoints_Input_to_I, s_tpoints_I, s_tpoints_teach],
		n_inds = [n_inds_Input_to_Einp, n_inds_E_inp, n_inds_E_outp, n_inds_Input_to_I, n_inds_I, n_inds_teach],
		bin_width_desired = 50*ms,
		t_run = t_run,
		exp_type = exp_type,
		exposure_n = exposure_n,
		t_start = t_start)
















