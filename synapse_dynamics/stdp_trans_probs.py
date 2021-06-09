# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:
- Run single synapse simulations to get transition probabilities for LTP and LTD.
- [REFACTOR]: use multithreading for frequencies loop.
"""
import setuptools
import os, sys, pickle
from brian2 import *
from scipy import *
from numpy import *

import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper modules
from load_parameters import *
from load_synapse_model import *
from run_single_synap import *

# == 1 - Simulation run variables ==========

sim_rep = 20

dt_resolution = 0.01		# 1ms | step of simulation time step resolution
t_run = 1					# simulation time (seconds)

N_Pre = 1
N_Post = 1

plasticity_rule = 'LR3'			# 'none', 'LR1', 'LR2', 'LR3'
parameter_set = '1.3'			# '2.1', '2.2', '2.4'

bistability = True
stoplearning = True

w_init = float(sys.argv[1])		# '0.0' to test LTP, '1.0' to test LTD

neuron_type = 'spikegenerator'	# 'poisson', 'LIF' , 'spikegenerator'
int_meth_syn = 'euler'			# Synaptic integration method

exp_type = 'stdp_trans_probabi_'

# Range of pre- and postsynaptic frequencies (Hz)
min_freq = 0
max_freq = 140

step = 10

# == 1.1 - neurons' firing rates

f_pre = np.arange(min_freq, max_freq + 0.1, step)
f_pos = np.arange(min_freq, max_freq + 0.1, step)

resul_per_pre_rate = []
transition_probabilities = []

# == 2 - Brian2 simulation settings ==========

# Starts a new scope for magic functions
start_scope()

tau_xstop = 0
xstop_jump = 0
thr_stop_h = 0
thr_stop_l = 0
xstop_max = 0
xstop_min = 0

thr_up_h = 0
thr_up_l = 0
thr_down_h = 0
thr_down_l = 0

tau_xstop = 400*ms
xstop_jump = 0.1
xstop_max = 1
xstop_min = 0
thr_stop_h = 0.7
thr_stop_l = 0.5

# loading learning rule parameters
[tau_xpre,
	tau_xpost,
	xpre_jump,
	xpost_jump,
	rho_neg,
	rho_neg2,
	rho_init,
	tau_rho,
	thr_post,
	thr_pre,
	thr_b_rho,
	rho_min,
	rho_max,
	alpha,
	beta,
	xpre_factor,
	w_max,
	xpre_min,
	xpost_min,
	xpost_max,
	xpre_max] = load_rule_params(
		plasticity_rule = plasticity_rule, 
		parameter_set = parameter_set,
		efficacy_init = w_init)

# loading synaptic rule equations
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability, stoplearning = stoplearning)

# [REFACTOR] - use multithreading for frequencies loop.
for x in range(0, len(f_pre)):
	if x != 0:
		outer_result = 0
		print('\npre: ', f_pre[x], 'Hz')

		for y in range(0, len(f_pos)):
			if y != 0:
				print(' - pos: ', f_pos[y], 'Hz')

				transition_event_count = 0

				for i in range(0, sim_rep):
					transition_event_count += run_single_synap(
						pre_rate = f_pre[x],
						post_rate = f_pos[y],
						t_run = t_run,
						dt_resolution = dt_resolution,
						plasticity_rule = plasticity_rule,
						neuron_type = neuron_type,
						bistability = bistability,
						N_Pre = N_Pre,
						N_Post = N_Post,
						tau_xpre = tau_xpre,
						tau_xpost = tau_xpost,
						xpre_jump = xpre_jump,
						xpost_jump = xpost_jump,
						rho_neg = rho_neg,
						rho_neg2 = rho_neg2,
						rho_init = rho_init,
						tau_rho = tau_rho,
						thr_post = thr_post,
						thr_pre = thr_pre,
						thr_b_rho = thr_b_rho,
						rho_min = rho_min,
						rho_max = rho_max,
						alpha = alpha,
						beta = beta,
						xpre_factor = xpre_factor,
						w_max = w_max,
						tau_xstop = tau_xstop,
						xstop_jump = xstop_jump,
						thr_up_h = thr_up_h,
						thr_up_l = thr_up_l,
						thr_down_h = thr_down_h,
						thr_down_l = thr_down_l,
						model_E_E = model_E_E,
						pre_E_E = pre_E_E,
						post_E_E = post_E_E,
						int_meth_syn = int_meth_syn,
						w_init = w_init,
						xpre_max = xpre_max)

				# transition prob for post- @ yHz
				transition_probabilities.append(transition_event_count/sim_rep)
			else:
				transition_probabilities.append(0.0)

		resul_per_pre_rate.append(transition_probabilities)
		transition_probabilities = []

fn = exp_type + '_' + \
	str(w_init) + '_' + \
	parameter_set + '_' + \
	str(bistability) + '_' + \
	'_last_rho.pickle'

fnopen = os.path.join(os.getcwd(), fn)

with open(fnopen,'wb') as f:
	pickle.dump((
		w_init,
		sim_rep,
		np.array(resul_per_pre_rate),
		f_pre,
		f_pos,
		min_freq,
		max_freq,
		step,
		exp_type,
		plasticity_rule,
		parameter_set,
		bistability,
		dt_resolution,
		t_run,
		int_meth_syn)
		, f)

print('\nstdp_trans_probs.py - END.\n')

		