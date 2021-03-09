# -*- coding: utf-8 -*-
"""
@author: Lehfeldt with some adaptations by asonntag and wgirao 

Input:

Output:
- counts:
- mean_attractor_frequencies:
- std_attractor_frequencies:
- attractor_frequencies_classified:

Comments:
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, mean, std

def varying_params_plot_performance_analysis(
	path_sim, 
	sim_id, 
	exp_type, 
	num_networks, 
	sim_folders_list, 
	simulation_flags, 
	varying_params, 
	delay_activities):

	lwdth = 2
	s0 = 20
	s1 = 30
	s2 = 60
	mpl.rcParams['axes.linewidth'] = lwdth

	# 1 - Counts of sucessful delay activity

	count_da = np.zeros(len(varying_params))
	count_fading_da = np.zeros(len(varying_params))
	count_no_da = np.zeros(len(varying_params))

	no_da_hline = 0
	fading_da_hline = 0
	da_hline = 0

	for i in np.arange(0, len(delay_activities), 1):
		dataset_temp = delay_activities[sim_folders_list[i]]
		flag_pos = simulation_flags[i][0]

		# No delay activity
		if dataset_temp.count(True) == 0:
			count_no_da[flag_pos] += 1

		# Fading delay activity    
		if dataset_temp.count(True) == 1:
			count_fading_da[flag_pos] += 1

		# Delay activity
		if dataset_temp.count(True) == 2:
			count_da[flag_pos] += 1

		param = sim_folders_list[i].split('_')[-1]

		if param == 'none':
			no_da_hline += count_no_da[flag_pos]
			fading_da_hline += count_fading_da[flag_pos]
			da_hline += count_da[flag_pos]

	count_no_da = count_no_da/num_networks
	count_fading_da = count_fading_da/num_networks
	count_da = count_da/num_networks

	no_da_hline = no_da_hline/num_networks
	fading_da_hline = fading_da_hline/num_networks
	da_hline = da_hline/num_networks


	all_counts_concatenated = np.concatenate((count_da, count_fading_da, count_no_da))

	counts = {}

	counts['No delay activity'] = count_no_da
	counts['Fading delay activity'] = count_fading_da
	counts['Delay activity'] = count_da
	counts['Parameters'] = varying_params

	# 2.1 Plotting counts as bar charts

	os.chdir(path_sim)
	plt.close('all')

	bar_width = 0.065 # 0.2 settings for find_wmax plots

	# Set position of bar on X axis
	r1 = np.arange(len(varying_params))
	r2 = [x + bar_width for x in r1]
	r3 = [x + bar_width for x in r2]

	fig = plt.figure(figsize = (25, 17.5))

	ax0 = fig.add_subplot(1, 1, 1)

	p1 = ax0.bar(r1, count_no_da, bar_width, color = 'lightcoral', edgecolor = 'black', hatch = '', linewidth = lwdth)

	p2 = ax0.bar(r2, count_fading_da, bar_width, color = 'lightblue', edgecolor = 'black', linewidth = lwdth)

	p3 = ax0.bar(r3, count_da, bar_width, color = 'deepskyblue', edgecolor = 'black',  linewidth = lwdth)

	ax0.legend(
		(p1[0],p2[0],p3[0]), 
		('No DA', 'Fading DA', 'DA'), 
		prop = {'size': 30}, 
		bbox_to_anchor = (1, 1), 
		ncol = 3)

	plt.axhline(y = no_da_hline, color = 'lightcoral', linestyle = '--')
	plt.axhline(y = fading_da_hline, color = 'lightblue', linestyle = '--')
	plt.axhline(y = da_hline, color = 'deepskyblue', linestyle = '--')

	plt.xlabel('Parameters', size = s1, labelpad = 15)
	plt.ylabel('Performance', size = s1, labelpad = 15)

	plt.title('Parameter variability', size = s1, pad = 15) 

	formated_x_labels = []

	for x in varying_params:
		if x == 'none':
			formated_x_labels.append(r'$\star$')
		elif x == 'wmax':
			formated_x_labels.append(r'$\mathit{w_{max}}$')
		elif x == 'c':
			formated_x_labels.append(r'$\mathit{c}$')
		elif x == 'tau_pre':
			formated_x_labels.append(r'$\mathit{\tau_{pre}}$')
		elif x == 'tau_post':
			formated_x_labels.append(r'$\mathit{\tau_{post}}$')
		elif x == 'rho_neg':
			formated_x_labels.append(r'$\mathit{\rho_{neg}}$')
		elif x == 'rho_neg2':
			formated_x_labels.append(r'$\mathit{\rho_{neg2}}$')
		elif x == 'thr_post':
			formated_x_labels.append(r'$\mathit{\theta_{post}}$')
		elif x == 'thr_pre':
			formated_x_labels.append(r'$\mathit{\theta_{pre}}$')
		elif x == 'all':
			formated_x_labels.append(r'$\infty$')
		else:
			formated_x_labels.append(x)

	plt.yticks(size = s1)
	plt.xticks([r + bar_width for r in range(len(varying_params))], formated_x_labels, size = s1)

	plt.ylim(-max(all_counts_concatenated)/10, max(all_counts_concatenated)*1.1) # settings for find_wmax plot

	plt.tick_params(axis = 'both',which = 'major',width = lwdth,length = 5,pad = 10)

	plt.savefig(os.path.join(path_sim,sim_id + '_learning_performance.png'), bbox_inches = 'tight')

	return counts





