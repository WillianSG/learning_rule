# -*- coding: utf-8 -*-
"""
@author: wgirao
Inputs:
Outputs:
Comments:
- https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
"""

import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from brian2 import mV, Hz, second, ms, mean, std
from firing_rate_histograms import firing_rate_histograms

def feedfoward_plot_spktrains_histograms(
	sim_id, 
	path_sim, 
	stim_size, 
	N, 
	s_tpoints, 
	n_inds, 
	bin_width_desired, 
	t_run, 
	exp_type,
	exposure_n = 0,
	t_start = 0*second):

	# General plotting settings

	lwdth = 3
	s1 = 60
	s2 = 105
	s3 = 120
	mpl.rcParams['axes.linewidth'] = lwdth

	plt.close('all')

	# Calculating firing rate histograms for plotting
	# 'time_resolved' histograms

	# Input to 1st layer
	[input_to_Einp_t_hist_count, 
	input_to_Einp_t_hist_edgs,
	input_to_Einp_t_hist_bin_widths,
	input_to_Einp_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[0],
		inds = n_inds[0],
		bin_width = bin_width_desired,
		N_pop = stim_size,
		flag_hist = 'time_resolved')

	# Einp
	[Ninp_t_hist_count, 
	Ninp_t_hist_edgs, 
	Ninp_t_hist_bin_widths,
	Ninp_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[1],
		inds = n_inds[1],
		bin_width = bin_width_desired,
		N_pop = N[0],
		flag_hist = 'time_resolved')

	# Eoutp
	[Eoutp_t_hist_count,
	Eoutp_t_hist_edgs,
	Eoutp_t_hist_bin_widths,
	Eoutp_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[2],
		inds = n_inds[2],
		bin_width = bin_width_desired,
		N_pop = N[1],
		flag_hist = 'time_resolved')

	# Input to I
	[input_to_I_t_hist_count,
	input_to_I_t_hist_edgs,
	input_to_I_t_hist_bin_widths,
	input_to_I_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[3],
		inds = n_inds[3],
		bin_width = bin_width_desired,
		N_pop = N[2],
		flag_hist = 'time_resolved')

	# I
	[I_t_hist_count,
	I_t_hist_edgs,
	I_t_hist_bin_widths,
	I_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[4],
		inds = n_inds[4],
		bin_width = bin_width_desired,
		N_pop = N[2],
		flag_hist = 'time_resolved')

	# teacher
	[teach_t_hist_count,
	teach_t_hist_edgs,
	teach_t_hist_bin_widths,
	teach_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[5],
		inds = n_inds[5],
		bin_width = bin_width_desired,
		N_pop = 1,
		flag_hist = 'time_resolved')

	# ---------------------------------------

	# Plotting spiking activity and histograms

	fig = plt.figure(figsize = (65, 45)) # w, h

	gs = gridspec.GridSpec(ncols = 1, nrows = 7, 
		height_ratios = [1, 2, 1, 2, 1, 1, 1])

	# gs.update(wspace = 0.07, hspace = 0.3, left = None, right = None)

	# Input to 1st layer ---------------------------------------
	ax0 = fig.add_subplot(gs[0, 0]) # row, col

	plt.bar(input_to_Einp_t_hist_edgs[:-1], input_to_Einp_t_hist_fr, input_to_Einp_t_hist_bin_widths, edgecolor = 'k', color = 'white', linewidth = lwdth)

	plt.ylabel('$\\nu_{Stimulus}$\n(Hz)', size = s1, labelpad = 35, 
		horizontalalignment = 'center')
	plt.xlim(t_start/second, t_run/second)
	plt.ylim(0, max(input_to_Einp_t_hist_fr)*1.1)

	yticks = np.linspace(0, max(input_to_Einp_t_hist_fr)*1.1, 3)

	ax0.set_yticks(yticks)
	ax0.set_yticklabels(np.around(yticks))

	ax0.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# Einp ---------------------------------------
	ax1 = fig.add_subplot(gs[1, 0])

	plt.scatter(s_tpoints[1], n_inds[1], color = 'b', s = s1)
	plt.ylabel('$E_{in}$', size = s1, labelpad = 105, horizontalalignment = 'center')

	ax1.set_yticks(np.arange(0, N[0]+1, N[0]/2))
	ax1.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)

	plt.ylim(0, N[0])

	plt.yticks(size = s1)
	plt.xticks(size = s1)

	plt.xlim(t_start/second, t_run/second)

	# Einp: histogram
	ax2 = fig.add_subplot(gs[2, 0])

	plt.bar(Ninp_t_hist_edgs[:-1], Ninp_t_hist_fr, Ninp_t_hist_bin_widths, edgecolor = 'b', color = 'white', linewidth = lwdth)
	plt.ylabel('$\\nu_{Ein}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(t_start/second, t_run/second)
	plt.ylim(0, max(Ninp_t_hist_fr)*1.1)

	yticks = np.linspace(0, max(Ninp_t_hist_fr)*1.1, 3)

	ax2.set_yticks(yticks)
	ax2.set_yticklabels(np.around(yticks))
	ax2.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# Eoutp ---------------------------------------
	ax3 = fig.add_subplot(gs[3, 0])
	
	plt.scatter(s_tpoints[2], n_inds[2], color = 'b', s = s1)
	plt.ylabel('$E_{out}$', size = s1, labelpad = 35, horizontalalignment = 'center')

	ax3.set_yticks(np.arange(0, N[1]+1, N[1]))
	ax3.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.ylim(-1, N[1])
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(t_start/second, t_run/second)

	# Eoutp: histogram
	ax4 = fig.add_subplot(gs[4, 0]) 

	plt.bar(Eoutp_t_hist_edgs[:-1], Eoutp_t_hist_fr, Eoutp_t_hist_bin_widths, 
		edgecolor = 'b', color = 'white', linewidth = lwdth)
	plt.ylabel('$\\nu_{E_{out}}$\n(Hz)', size = s1, labelpad = 35,
		horizontalalignment = 'center')
	plt.xlim(t_start/second, t_run/second)

	yticks = np.linspace(0, max(Eoutp_t_hist_fr)*1.1, 3)

	ax4.set_yticks(yticks)
	ax4.set_yticklabels(np.around(yticks))
	ax4.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# I ---------------------------------------
	ax5 = fig.add_subplot(gs[5, 0])

	plt.bar(I_t_hist_edgs[:-1], I_t_hist_fr,
		I_t_hist_bin_widths, 
		edgecolor = 'tomato', 
		color = 'white', 
		linewidth = lwdth)
	plt.ylabel('$\\nu_{I}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(t_start/second, t_run/second)

	yticks = np.linspace(0, max(I_t_hist_fr)*1.1, 3)

	ax5.set_yticks(yticks)
	ax5.set_yticklabels(np.around(yticks))
	ax5.set_xticklabels([])

	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# teach ---------------------------------------
	ax6 = fig.add_subplot(gs[6, 0])

	plt.bar(teach_t_hist_edgs[:-1], teach_t_hist_fr,
		teach_t_hist_bin_widths, 
		edgecolor = 'green', 
		color = 'white', 
		linewidth = lwdth)
	plt.ylabel('$\\nu_{teach}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(t_start/second, t_run/second)
	plt.ylim(0, max(teach_t_hist_fr)*1.1)

	yticks = np.linspace(0, max(teach_t_hist_fr)*1.1, 3)

	ax6.set_yticks(yticks)
	ax6.set_yticklabels(np.around(yticks))

	plt.yticks(size = s1)
	plt.xticks(size = s1)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 15)

	plt.xlabel('Time (s)', size = s1)

	plt.savefig(os.path.join(path_sim, sim_id + '_population_spiking_expos' + str(exposure_n) + '.png'), bbox_inches = 'tight')

	plt.close(fig)