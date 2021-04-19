# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Parameter set definitions.
"""
input:
- plasticity_rule():
- parameter_set():

output:
- tau_xpre():
- tau_xpost():
- xpre_jump():
- xpost_jump():
- rho_neg():
- rho_neg2():
- rho_init():
- tau_rho():
- thr_post():
- thr_pre():
- thr_b_rho():
- rho_min():
- rho_max():
- alpha():
- beta():
- xpre_factor():
- w_max():

Comments:
"""
def load_rule_params(plasticity_rule, parameter_set, efficacy_init = 0.5):
	from brian2 import ms, mV
	xpre_jump = 0.2 				# jump of x_pre
	xpost_jump = 0.2 				# jump of x_post
	rho_init = efficacy_init 		# initial rho value
	tau_rho = 350000*ms 			# rho time constant
	rho_min = 0.0 					# DOWN state
	rho_max = 1 					# UP state
	thr_b_rho = 0.5 				# bistability threshold
	alpha = 1 						# slope of bistability
	beta = alpha

	tau_xstop = 0*ms
	xstop_jump = 0
	thr_stop_h = 0
	thr_stop_l = 0

	if plasticity_rule == 'LR1':
		if parameter_set == '1.1':
			tau_xpre = 22*ms
			tau_xpost = 22*ms
			xpre_factor = 0.1
			thr_post = 0.5
			thr_pre = 0.0
			rho_neg = -0.05
			rho_neg2 = rho_neg
			tau_rho = 1000*ms
	elif plasticity_rule == 'LR2':
		if parameter_set =='2.1':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.1 # scaling factor positive efficacy change
			thr_post = 0.4 # threshold for x_post
			thr_pre = 0.2 # threshold for x_pre
			rho_neg = -0.05 # negative efficacy change
			rho_neg2 = rho_neg # additional negative efficacy change 
		elif parameter_set =='2.2':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.013 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_neg = -0.008
			rho_neg2 = rho_neg
			# rho_neg = -0.0008 # negative efficacy change
			# rho_neg2 = rho_neg*10 # additional negative efficacy change 
		elif parameter_set =='2.3':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.017 # scaling factor positive efficacy change
			thr_post = 0.4 # threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_neg = -0.00055 # negative efficacy change
			rho_neg2 = rho_neg*10 # additional negative efficacy change 
		elif parameter_set =='2.4':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.21 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_neg = -0.008 # negative efficacy change
			rho_neg2 = rho_neg*10 # additional negative efficacy change
	elif plasticity_rule == 'LR3_1':
		if parameter_set =='2.0':
			tau_xpre = 30*ms 
			tau_xpost = 30*ms
			tau_xstop = 55*ms
			xpre_factor = 0.05
			thr_post = 0.4
			thr_pre = 0.5
			rho_neg = -0.1
			rho_neg2 = rho_neg*10
			xstop_jump = 0.2
			thr_stop_h = 0.9
			thr_stop_l = 0.3
	elif plasticity_rule == 'LR3_2':
		if parameter_set =='3.0':
			tau_xpre = 30*ms 
			tau_xpost = 30*ms
			xpre_factor = 0.1
			rho_neg = -0.01
			xpre_jump = 0.12 				# jump of x_pre
			xpost_jump = 0.12 				# jump of x_post
			thr_post = 0.1
			thr_pre = 0.1
			thr_stop_h = 0.4
			thr_stop_l = 0.2
			xpost_max = 1.0
			xpre_max = 1.0
			xstop_jump = 0.12
			xstop_max = 1.0
			tau_xstop = 60*ms
		elif parameter_set =='3.1':
			tau_xpre = 30*ms 
			tau_xpost = 30*ms
			xpre_factor = 0.1
			rho_neg = -0.01
			xpre_jump = 0.12 				# jump of x_pre
			xpost_jump = 0.15 				# jump of x_post
			thr_post = 0.1
			thr_pre = 0.3
			thr_stop_h = 0.4
			thr_stop_l = 0.1
			xpost_max = 1.0
			xpre_max = 1.0
			xstop_jump = 0.08
			xstop_max = 1.0
			tau_xstop = 100*ms
		elif parameter_set =='3.2':
			tau_xpre = 30*ms 
			tau_xpost = 30*ms
			xpre_factor = 0.1
			rho_neg = -0.01
			xpre_jump = 0.12 				# jump of x_pre
			xpost_jump = 0.15 				# jump of x_post
			thr_post = 0.1
			thr_pre = 0.3
			thr_stop_h = 0.4
			thr_stop_l = 0.1
			xpost_max = 1.0
			xpre_max = 1.0
			xstop_jump = 0.08
			xstop_max = 1.0
			tau_xstop = 100*ms
	else: # default '2.1'
		tau_xpre = 13*ms
		tau_xpost = 33*ms
		xpre_factor = 0.1
		thr_post = 0.4
		thr_pre = 0.2
		rho_neg = -0.05
		rho_neg2 = rho_neg

	w_max = 1*mV
	rho_neg2 = rho_neg*10

	return tau_xpre,\
	tau_xpost,\
	xpre_jump,\
	xpost_jump,\
	rho_neg,\
	rho_neg2,\
	rho_init,\
	tau_rho,\
	thr_post,\
	thr_pre,\
	thr_b_rho,\
	rho_min,\
	rho_max,\
	alpha,\
	beta,\
	xpre_factor,\
	w_max,\
	tau_xstop,\
	xstop_jump,\
	thr_stop_h,\
	thr_stop_l,\
	xpost_max,\
	xpre_max,\
	xstop_max