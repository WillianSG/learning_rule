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
	xpre_jump = 1 # jump of x_pre
	xpost_jump = 1 # jump of x_post
	rho_init = efficacy_init # initial rho value
	tau_rho = 350000*ms # rho time constant
	rho_min = 0.0 # DOWN state
	rho_max = 1 # UP state
	thr_b_rho = 0.5 # bistability threshold
	alpha = 1 # slope of bistability
	beta = alpha

	xpre_min = 0.0
	xpost_min = 0.0
	xpost_max = 1.0
	xpre_max = 1.0

	rho_neg = -0.05
	rho_neg2 = -0.05

	thr_pre = 0.0

	if plasticity_rule == 'LR2':
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
			rho_neg = -0.0008 # -0.008
			rho_neg2 = rho_neg 
		elif parameter_set =='2.3':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.017 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
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
		elif parameter_set =='2.5':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.08 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_neg = -0.008 # -0.008
			rho_neg2 = rho_neg
	elif plasticity_rule == 'LR1':
		if parameter_set == '1.1':
			tau_xpre = 22*ms
			tau_xpost = 22*ms
			xpre_factor = 0.1
			thr_post = 0.5
			thr_pre = 0.0
			rho_neg = -0.05
			rho_neg2 = rho_neg
			tau_rho = 1000*ms
	elif plasticity_rule == 'LR3':
		if parameter_set == '0.0':
			tau_xpre = 50*ms
			tau_xpost = 50*ms
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.5
			thr_pre = 0.5
			rho_neg = -0.05
			rho_neg2 = -0.05
			xpre_factor = 0.1
		elif parameter_set == '0.1': # from 0.0
			tau_xpre = 50*ms
			tau_xpost = 40*ms #
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.5
			thr_pre = 0.3 #
			rho_neg = -0.05
			rho_neg2 = -0.05
			xpre_factor = 0.1
		elif parameter_set == '0.2': # from 0.1
			tau_xpre = 50*ms
			tau_xpost = 40*ms
			tau_rho = 1000*ms
			xpre_jump = 0.4 #
			xpost_jump = 0.5
			thr_post = 0.4 #
			thr_pre = 0.3
			rho_neg = -0.06 #
			rho_neg2 = -0.05
			xpre_factor = 0.1
		elif parameter_set == '0.21': # from 0.2
			tau_xpre = 50*ms
			tau_xpost = 40*ms
			tau_rho = 1000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.06
			rho_neg2 = -0.01 #
			xpre_factor = 0.1
		elif parameter_set == '0.22': # from 0.21
			tau_xpre = 50*ms
			tau_xpost = 40*ms
			tau_rho = 1000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.06
			rho_neg2 = -0.03 #
			xpre_factor = 0.1
		elif parameter_set == '0.23': # from 0.21
			tau_xpre = 55*ms
			tau_xpost = 30*ms # reduced
			tau_rho = 1000*ms
			xpre_jump = 0.6 # incresed
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.07 # reduced
			rho_neg2 = -0.1
			xpre_factor = 0.1
		elif parameter_set == '0.3': # from none
			tau_xpre = 48*ms
			tau_xpost = 40*ms
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.05
			rho_neg2 = -0.005
			xpre_factor = 0.05
		elif parameter_set == '0.31': # from 0.3
			tau_xpre = 48*ms
			tau_xpost = 38*ms #
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.05
			rho_neg2 = -0.005
			xpre_factor = 0.05
		elif parameter_set == '0.32': # from 0.31
			tau_xpre = 48*ms
			tau_xpost = 35*ms #
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.05
			rho_neg2 = -0.005
			xpre_factor = 0.05
		elif parameter_set == '0.33': # from 0.31
			tau_xpre = 50*ms #
			tau_xpost = 36*ms #
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.05
			rho_neg2 = -0.005
			xpre_factor = 0.05
		elif parameter_set == '0.34': # from 0.33
			tau_xpre = 50*ms #
			tau_xpost = 35*ms #
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.05
			rho_neg2 = -0.005
			xpre_factor = 0.05
		elif parameter_set == '0.4': # from 0.3
			tau_xpre = 48*ms
			tau_xpost = 40*ms
			tau_rho = 1000*ms
			xpre_jump = 0.5
			xpost_jump = 0.5
			thr_post = 0.4
			thr_pre = 0.3
			rho_neg = -0.03
			rho_neg2 = -0.005
			xpre_factor = 0.05
	else: # default '2.1'
		tau_xpre = 13*ms
		tau_xpost = 33*ms
		xpre_factor = 0.1
		thr_post = 0.4
		thr_pre = 0.2
		rho_neg = -0.05
		rho_neg2 = rho_neg

	w_max = 1*mV

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
	xpre_min,\
	xpost_min,\
	xpost_max,\
	xpre_max