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

def feedforward_plot_activity(
	sim_id, 
	path_sim,
	t_run,
	rho_matrix,
	time_arr,
	w_matrix,
	time_arr_w,
	eout_mon,
	eout_time_arr,
	stim_ids,
	exposure_n = 0,
	t_start = 0*second):

	# General plotting settings

	lwdth = 1
	s1 = 60
	s2 = 105
	mpl.rcParams['axes.linewidth'] = lwdth

	plt.close('all')

	fig = plt.figure() # w, h

	# spec2 = gridspec.GridSpec(ncols=2, nrows=2)
	spec2 = gridspec.GridSpec(ncols=2, nrows=1)

	# Rho ---
	f2_ax1 = fig.add_subplot(spec2[0, 0])

	count = 0
	for row in rho_matrix:
		if (count in stim_ids):
			plt.plot(time_arr, row, label='neur. # ' + str(count),linestyle = '-', linewidth = 2, color = 'tomato')
		else:
			plt.plot(time_arr, row, linestyle = '--', linewidth = 1, color = 'gray')
		count += 1

	plt.ylabel('rho (a.u.)')

	plt.xlabel('time (sec)')

	plt.xlim(t_start/second, t_run/second)
	plt.ylim(0, 1)

	# Post membrane potential ---
	f2_ax2 = fig.add_subplot(spec2[0, 1])

	plt.plot(eout_time_arr, eout_mon[0]/mV)

	plt.ylabel('$out_{mem}$ (mV)')

	plt.xlabel('time (sec)')

	plt.xlim(t_start/second, t_run/second)

	# Weight ---
	# f2_ax3 = fig.add_subplot(spec2[1, 0])

	# for row in w_matrix:
	# 	plt.plot(time_arr_w, row/mV)

	# plt.ylabel('w (mV)')

	# plt.xlabel('time (sec)')

	# plt.xlim(0, t_run/second)
	# plt.ylim(0, 1)

	# ---
	# f2_ax4 = fig.add_subplot(spec2[1, 1])

	fig.tight_layout()

	plt.savefig(os.path.join(path_sim, sim_id + '_activity_expos' + str(exposure_n) + '.png'), bbox_inches = 'tight')

	plt.close(fig)




	# ---------------------------------------

	# Plotting spiking activity and histograms

	# fig = plt.figure(figsize = (65, 45)) # w, h

	# gs = gridspec.GridSpec(ncols = 1, nrows = 6, 
	# 	height_ratios = [1, 2, 1, 2, 1, 1])

	# # gs.update(wspace = 0.07, hspace = 0.3, left = None, right = None)

	# # Input to 1st layer ---------------------------------------
	# ax0 = fig.add_subplot(gs[0, 0]) # row, col

	# plt.bar(input_to_Einp_t_hist_edgs[:-1], input_to_Einp_t_hist_fr, input_to_Einp_t_hist_bin_widths, edgecolor = 'k', color = 'white', linewidth = lwdth)

	# plt.ylabel('$\\nu_{Stimulus}$\n(Hz)', size = s1, labelpad = 35, 
	# 	horizontalalignment = 'center')
	# plt.xlim(0, t_run/second)
	# plt.ylim(0, max(input_to_Einp_t_hist_fr)*1.1)

	# yticks = np.linspace(0, max(input_to_Einp_t_hist_fr)*1.1, 3)

	# ax0.set_yticks(yticks)
	# ax0.set_yticklabels(np.around(yticks))

	# ax0.set_xticklabels([])

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	# plt.yticks(size = s1)
	# plt.xticks(size = s1)

	# # Einp ---------------------------------------
	# ax1 = fig.add_subplot(gs[1, 0])
	
	# plt.plot(s_tpoints[1], n_inds[1], '.', color = 'skyblue')
	# plt.ylabel('$E_{in}$', size = s1, labelpad = 105, horizontalalignment = 'center')

	# ax1.set_yticks(np.arange(0, N[0]+1, N[0]/2))
	# ax1.set_xticklabels([])

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)

	# plt.ylim(0, N[0])

	# plt.yticks(size = s1)
	# plt.xticks(size = s1)

	# plt.xlim(0, t_run/second)

	# # Einp: histogram
	# ax2 = fig.add_subplot(gs[2, 0])

	# plt.bar(Ninp_t_hist_edgs[:-1], Ninp_t_hist_fr, Ninp_t_hist_bin_widths, edgecolor = 'skyblue', color = 'white', linewidth = lwdth)
	# plt.ylabel('$\\nu_{Ein}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	# plt.xlim(0, t_run/second)
	# plt.ylim(0, max(Ninp_t_hist_fr)*1.1)

	# yticks = np.linspace(0, max(Ninp_t_hist_fr)*1.1, 3)

	# ax2.set_yticks(yticks)
	# ax2.set_yticklabels(np.around(yticks))
	# ax2.set_xticklabels([])

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
	# 	pad = 10)
	# plt.yticks(size = s1)
	# plt.xticks(size = s1)

	# # Eoutp ---------------------------------------
	# ax3 = fig.add_subplot(gs[3, 0])

	# plt.plot(s_tpoints[2], n_inds[2], '.', color = 'mediumblue')
	# plt.ylabel('$E_{out}$', size = s1, labelpad = 35, horizontalalignment = 'center')

	# ax3.set_yticks(np.arange(0, N[1]+1, N[1]))
	# ax3.set_xticklabels([])

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	# plt.ylim(0, N[1])
	# plt.yticks(size = s1)
	# plt.xticks(size = s1)
	# plt.xlim(0, t_run/second)

	# # Eoutp: histogram
	# ax4 = fig.add_subplot(gs[4, 0]) 

	# plt.bar(Eoutp_t_hist_edgs[:-1], Eoutp_t_hist_fr, Eoutp_t_hist_bin_widths, 
	# 	edgecolor = 'mediumblue', color = 'white', linewidth = lwdth)
	# plt.ylabel('$\\nu_{E_{out}}$\n(Hz)', size = s1, labelpad = 35,
	# 	horizontalalignment = 'center')
	# plt.xlim(0, t_run/second)

	# yticks = np.linspace(0, max(Eoutp_t_hist_fr)*1.1, 3)

	# ax4.set_yticks(yticks)
	# ax4.set_yticklabels(np.around(yticks))
	# ax4.set_xticklabels([])

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
	# 	pad = 10)
	# plt.yticks(size = s1)
	# plt.xticks(size = s1)

	# # I ---------------------------------------
	# ax5 = fig.add_subplot(gs[5, 0])

	# plt.bar(I_t_hist_edgs[:-1], I_t_hist_fr,
	# 	I_t_hist_bin_widths, 
	# 	edgecolor = 'tomato', 
	# 	color = 'white', 
	# 	linewidth = lwdth)
	# plt.ylabel('$\\nu_{I}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	# plt.xlim(0, t_run/second)
	# plt.ylim(0, max(I_t_hist_fr)*1.1)

	# yticks = np.linspace(0, max(I_t_hist_fr)*1.1, 3)

	# ax5.set_yticks(yticks)
	# ax5.set_yticklabels(np.around(yticks))
	# ax5.set_xticklabels([])

	# # ax5.set_yticks(np.arange(0, 4500, 2000))

	# plt.yticks(size = s1)
	# plt.xticks(size = s1)

	# plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
	# 	pad = 15)

	# plt.xlabel('Time (s)', size = s1)

	# plt.savefig(os.path.join(path_sim, sim_id + '_population_spiking.png'), bbox_inches = 'tight')

	# plt.close(fig)