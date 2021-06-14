# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Network proposed in "Learning Real-World Stimuli in a Neural Network with Spike-Driven Synaptic Dynamics".

Observations:
- Output population has to separated in pools, each pool with N_c neurons, where N_c is the total number of classes in the training data.
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
from random import uniform
import numpy as np
from time import localtime

# Helper Modules
from load_parameters import *
from load_synapse_model import *
from load_stimulus_Ninp import *

prefs.codegen.target = 'numpy'

class FeedforwardNetwork:
	def __init__(self):
		self.syn_delay_Vepsp_in_out = 0*ms
		# ----------- Network Parameters -----------
		self.network_id = ''
		self.simulation_path = ''

		# Learning Rule
		self.plasticity_rule = 'user-defined'
		self.parameter_set = 'user-defined'
		self.bistability = True

		# Neuron Groups
		self.neuron_type = 'user-defined'	# neuron model
		self.N_c = 1						# num. of classes in data

		# Synapses
		self.Einp_to_Eout_w = 7.5*mV
		self.Input_to_I_w = 1*mV
		self.Input_to_Einp_w = 1*mV
		self.I_to_Eout_w = 1*mV
		self.teacher_to_Eout_w = 1*mV
		self.spont_to_input_w = 1*mV

		self.w_e_e_max = 1*mV 		# upper boundary for excit. syn.

		# Inhibitory Population
		self.Vr_i = -60*mV 			# resting voltage
		self.Vrst_i = -60*mV 		# reset potential
		self.Vth_i = -40*mV 		# threshold voltage
		self.tau_i = 10*ms 			# membrane time constant
		self.tref_i = 1*ms 			# refractory period
		self.tau_epsp_i = 3.5*ms	# time constant of EPSP
		self.tau_ipsp_i = 5.5*ms 	# time constant of IPSP
		self.stim_freq_i = 0*Hz

		# Input (E) Population
		self.N_e = 45 				# num. of neurons
		self.Vr_e = -65*mV 			# resting potential
		self.Vrst_e = -65*mV 		# reset potential
		self.Vth_e_init = -58*mV 	# initial threshold voltage
		self.Vth_e_incr = 5*mV 		# post-spike threshold voltage increase
		self.tau_Vth_e = 20*ms 		# time constant of threshold decay
		self.tau_e = 20*ms 			# membrane time constant
		self.tref_e = 0*ms 			# refractory period
		self.tau_epsp_e = 3.5*ms 	# time constant of EPSP
		self.tau_ipsp_e = 5.5*ms 	# time constant of IPSP
		self.stim_freq_Ninp = 0*Hz	# firing freq. of input layer 
		self.stimulus_id = ''		# stimulus id
		self.stim_size = 0

		# Output Population
		self.N_e_outp = 1

		# Teacher
		self.stim_freq_teach = 2000*Hz

		# Spontaneus activity
		self.stim_freq_spont = 5*Hz

		# Rule parameters
		self.tau_rho = None
		self.thr_b_rho = None
		self.xpre_jump = None
		self.xpost_jump = None
		self.tau_xpre = None 
		self.tau_xpost = None
		self.alpha = None
		self.beta = None
		self.xpre_factor = None
		self.thr_post = None
		self.thr_pre = None
		self.rho_init = None
		self.rho_neg = None
		self.rho_neg2 = None
		self.rho_min = 0
		self.rho_max = 1
		self.tau_xstop = None
		self.xstop_jump = None
		self.thr_up_h = None
		self.thr_up_l = None
		self.thr_down_h = None
		self.thr_down_l = None

		self.xpre_max = 1

		# Stop-Learning
		self.tau_xstop = 300*ms
		self.xstop_jump = 0.09
		self.xstop_max = 1
		self.xstop_min = 0
		self.thr_stop_h = 0.6
		self.thr_stop_l = 0.55

		# General
		self.exp_type = 'user-defined' 		# type of experiment
		self.executed_at = '--'				# execution date and time

		self.t_run = 1*second 				# duration of simul.
		self.int_meth_neur = 'user-defined' # numerical integration method
		self.int_meth_syn = 'user-defined' 	# E_E syn. num. integration method
		self.dt_resolution = 0.001*second 	# Delta t of clock intervals
		self.mon_dt = 0.001*second				# Delta t for monitors clock

		self.stimulus_size = self.N_e		# Number of neurons in inp. layer
		self.stim_active_neur_size = 0		# Number of active ids within stim.
		self.coding_lvl = 0.00				# Percetange of active neurons

	def initialize_network_modules(self):
		self.set_inhibitory_pop()
		self.set_input_pop()
		self.set_output_pop()
		self.set_teach_pop()
		self.set_spont_activ_pop()
		self.set_synapses()
		self.set_weights()
		self.set_learning_rule_parameters()
		self.set_spike_mon()
		self.set_state_mon()
		self.set_stimulus_Ninp()
		self.set_stimulus_I()
		self.set_network_object()

	# ----------- Population Methods -----------

	def set_inhibitory_pop(self):
		self.eqs_I = Equations(''' 
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_i)) / tau_i : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt''',
			Vr_i = self.Vr_i,
			tau_i = self.tau_i,
			tau_epsp = self.tau_epsp_i,
			tau_ipsp = self.tau_ipsp_i)

		self.I = NeuronGroup(N = self.N_e_outp, model = self.eqs_I,
			reset = 'Vm = Vrst_i',
			threshold = 'Vm > Vth_i',
			refractory = self.tref_i,
			method = self.int_meth_neur, 
			name = 'I')

		self.Input_to_I = NeuronGroup(N = self.N_e_outp, 
			model = 'rates : Hz', 
			threshold = 'rand()<rates*dt', 
			name = 'Input_to_I')

		self.I.Vepsp = 0
		self.I.Vipsp = 0

		# Random membrane voltages
		self.I.Vm = (self.Vrst_i + rand(self.N_e_outp) * (self.Vth_i - self.Vr_i))

	def set_input_pop(self):
		self.eqs_E_inp = Equations('''
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / tau_e : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt
			dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt''',
			Vr_e = self.Vr_e,
			tau_e = self.tau_e,
			tau_epsp = self.tau_epsp_e,
			tau_ipsp = self.tau_ipsp_e,
			Vth_e_init = self.Vth_e_init,
			tau_Vth_e = self.tau_Vth_e)

		# Input layer
		self.E_inp = NeuronGroup(N = self.N_e, model = self.eqs_E_inp,
			reset = '''Vm = Vrst_e 
					Vth_e += Vth_e_incr''',
			threshold = 'Vm > Vth_e',
			refractory = self.tref_e,
			method = self.int_meth_neur, 
			name = 'E_inp')

		# "Virtual" neurons driving input layer
		self.Input_to_E_inp = NeuronGroup(N = self.N_e,
			model = 'rates : Hz',
			threshold = 'rand()<rates*dt', 
			name = 'Input_to_E_inp')

		self.E_inp.Vth_e = self.Vth_e_init

		# Random membrane voltages
		self.E_inp.Vm = (self.Vrst_e + rand(self.N_e) * (self.Vth_e_init - self.Vr_e))

	def set_output_pop(self):
		self.eqs_E_outp = Equations('''
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / tau_e : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt
			dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt''',
			Vr_e = self.Vr_e,
			tau_e = self.tau_e,
			tau_epsp = self.tau_epsp_e,
			tau_ipsp = self.tau_ipsp_e,
			Vth_e_init = self.Vth_e_init,
			tau_Vth_e = self.tau_Vth_e)

		# Output layer
		self.E_outp = NeuronGroup(N = self.N_e_outp, model = self.eqs_E_outp,
			reset = '''Vm = Vrst_e 
					Vth_e += Vth_e_incr''',
			threshold = 'Vm > Vth_e',
			refractory = self.tref_e,
			method = self.int_meth_neur, 
			name = 'E_outp')

		self.E_outp.Vth_e = self.Vth_e_init

		# Random membrane voltages
		self.E_outp.Vm = (self.Vrst_e + rand(self.N_e_outp) * (self.Vth_e_init - self.Vr_e))

	def set_teach_pop(self):
		self.teacher_pop = NeuronGroup(N = 1,
			model = 'rates : Hz',
			threshold = 'rand()<rates*dt', 
			name = 'teacher')

		self.teacher_pop.rates[range(0, 1)] = self.stim_freq_teach

	def set_teacher_signal(self, pattern_id):
		signal = 6*Hz

		if ((pattern_id+1) % 2) == 0:
			signal = 60*Hz

		self.teacher_pop.rates[range(0, 1)] = signal

	def set_spont_activ_pop(self):
		self.spontaneous_act_pop = NeuronGroup(N = self.N_e,
			model = 'rates : Hz',
			threshold = 'rand()<rates*dt', 
			name = 'spont_activity')

		self.spontaneous_act_pop.rates[range(0, self.N_e)] = self.stim_freq_spont


	# ----------- Synaptic Connections Methods -----------

	def set_synapses(self):
		[self.model_E_E, self.pre_E_E, self.post_E_E] = load_synapse_model(
			plasticity_rule = self.plasticity_rule,
			neuron_type = self.neuron_type,
			bistability = self.bistability)

		# between "input neurons" and 1st (E_inp) layer
		self.Input_1st_layer = Synapses(source = self.Input_to_E_inp, 
			target = self.E_inp, 
			model = 'w : volt', 
			on_pre = 'Vepsp += w', 
			name = 'Input_1st_layer')

		# between "inhibitory input" and inhibitory (I) population
		self.Input_I = Synapses(source = self.Input_to_I, target = self.I, 
			model = 'w : volt',
			on_pre = 'Vepsp += w',
			name = 'Input_I')

		# (plastic) between 1st (input) and output layer
		self.Input_to_Output = Synapses(
			source = self.E_inp, 
			target = self.E_outp,
			model = self.model_E_E,
			on_pre = self.pre_E_E,
			on_post = self.post_E_E,
			method = self.int_meth_syn,
			name = 'Input_to_Output')

		# between inhibitory (I) and output populations
		self.I_Eout = Synapses(source = self.I, target = self.E_outp, 
			model = 'w : volt',
			on_pre = 'Vipsp += w',
			name = 'I_Eout')

		# between teacher and output populations
		self.teacher_Eout = Synapses(
			source = self.teacher_pop,
			target = self.E_outp, 
			model = 'w : volt',
			on_pre = 'Vepsp += w',
			name = 'teacher_Eout')

		# spontaneous activity
		self.spont_input = Synapses(
			source = self.spontaneous_act_pop,
			target = self.E_inp, 
			model = 'w : volt',
			on_pre = 'Vepsp += w',
			name = 'spont_input')

		self.Input_1st_layer.connect(j = 'i')	# drives activity in 1st layer
		self.Input_I.connect(j = 'i')			# drives activity in inhi. pop.
		self.Input_to_Output.connect()			# feedforward connections
		self.I_Eout.connect(j = 'i')			# inhib. pop. -> output pop.

		self.teacher_Eout.connect(j = 'i')
		self.spont_input.connect(j = 'i')

		self.Input_to_Output.Vepsp_transmission.delay = self.syn_delay_Vepsp_in_out

		# saving connection matrix
		self.M_syn = np.full((len(self.E_inp), len(self.E_outp)), np.nan)
		self.M_syn[self.Input_to_Output.i[:], self.Input_to_Output.j[:]] = self.Input_to_Output.rho[:]

		self.randomize_synaptic_weights()

	def set_weights(self):
		self.Input_to_Output.w = self.Einp_to_Eout_w
		self.Input_I.w = self.Input_to_I_w
		self.Input_1st_layer.w = self.Input_to_Einp_w
		self.I_Eout.w = self.I_to_Eout_w
		self.teacher_Eout.w = self.teacher_to_Eout_w
		self.spont_input.w = self.spont_to_input_w

	def randomize_synaptic_weights(self):
		for pre_id in range(0, len(self.E_inp)):
			for post_id in range(0, len(self.E_outp)):
				if isnan(self.M_syn[pre_id][post_id]) == False:
					s = uniform(0, 1)					
					self.Input_to_Output.rho[pre_id, post_id] = round(s, 2)
					# self.Input_to_Output.rho[pre_id, post_id] = 1.0

		self.M_syn[self.Input_to_Output.i[:], self.Input_to_Output.j[:]] = self.Input_to_Output.rho[:]

	# ----------- Monitors Methods -----------

	def set_spike_mon(self):
		self.Input_to_E_inp_spkmon = SpikeMonitor(
			source = self.Input_to_E_inp, 
			record = True, 
			name = 'Input_to_E_inp_spkmon')

		self.E_inp_spkmon = SpikeMonitor(
			source = self.E_inp, 
			record = True, 
			name = 'E_inp_spkmon')

		self.E_outp_spkmon = SpikeMonitor(
			source = self.E_outp, 
			record = True, 
			name = 'E_outp_spkmon')

		self.Input_to_I_spkmon = SpikeMonitor(
			source = self.Input_to_I, 
			record = True, 
			name = 'Input_to_I_spkmon')

		self.I_spkmon = SpikeMonitor(
			source = self.I, 
			record = True, 
			name = 'I_spkmon')

		self.teacher_spkmon = SpikeMonitor(
			source = self.teacher_pop, 
			record = True, 
			name = 'teacher_spkmon')

		self.spont_spkmon = SpikeMonitor(
			source = self.spontaneous_act_pop, 
			record = True, 
			name = 'spont_spkmon')

	def set_state_mon(self):
		self.Input_1st_layer_stamon = StateMonitor(
			source = self.Input_1st_layer,
			variables = ('w'),
			record = False,
			dt = self.mon_dt,
			name = 'Input_1st_layer_stamon')

		self.Input_I_stamon = StateMonitor(
			source = self.Input_I,
			variables = ('w'),
			record = False,
			dt = self.mon_dt,
			name = 'Input_I_stamon')

		self.Eout_stamon = StateMonitor(
			source = self.E_outp,
			variables = ('Vm','Vepsp','Vipsp'),
			record = True,
			dt = self.mon_dt,
			name = 'Eout_stamon')

		self.Einp_stamon = StateMonitor(
			source = self.E_inp,
			variables = ('Vm','Vepsp','Vipsp'),
			record = True,
			dt = self.mon_dt,
			name = 'Einp_stamon')

		self.I_stamon = StateMonitor(
			source = self.I,
			variables = ('Vm','Vepsp','Vipsp'),
			record = True,
			dt = self.mon_dt,
			name = 'I_stamon')

		self.Input_to_Output_stamon = StateMonitor(
			source = self.Input_to_Output,
			variables = ('w', 'rho'),
			record = True,
			dt = self.mon_dt,
			name = 'Input_to_Output_stamon')

		self.I_Eout_stamon = StateMonitor(
			source = self.I_Eout,
			variables = ('w'),
			record = False,
			dt = self.mon_dt,
			name = 'I_Eout_stamon')

	# ----------- Network Operation Methods -----------

	def set_stimulus_Ninp(self):
		# Setting active neurons in the input layer
		self.Input_to_E_inp.rates = 0*Hz # reset input activity
		self.stimulus_ids_Ninp = load_stimulus_Ninp(self.stimulus_id, self.N_e)
		self.Input_to_E_inp.rates[self.stimulus_ids_Ninp] = self.stim_freq_Ninp

		self.stim_size = len(self.stimulus_ids_Ninp)

		# Saving *coding level* of input stimulus
		self.stim_active_neur_size = len(self.stimulus_ids_Ninp)
		self.coding_lvl = round(
			self.stim_active_neur_size / self.stimulus_size, 2)

		# Updating Inhibitory population activity lvl
		self.set_I_from_coding_lvl()

	def set_stimulus_dataset(self, pattern):
		# getting ids of active neurons in pattern
		active_ids = np.nonzero(pattern)
		
		# Setting active neurons in the input layer
		self.Input_to_E_inp.rates = 0*Hz # reset input activity
		self.stimulus_ids_Ninp = active_ids[0]
		self.Input_to_E_inp.rates[self.stimulus_ids_Ninp] = self.stim_freq_Ninp

		self.stim_size = len(self.stimulus_ids_Ninp)

		# Saving *coding level* of input stimulus
		self.stim_active_neur_size = len(self.stimulus_ids_Ninp)
		self.coding_lvl = round(
			self.stim_active_neur_size / self.stimulus_size, 2)

		# Updating Inhibitory population activity lvl
		self.set_I_from_coding_lvl()

	def set_stimulus_I(self):
		self.Input_to_I.rates[range(0, self.N_e_outp)] = self.stim_freq_i

	def set_I_from_coding_lvl(self):
		# Setting active neurons in the Inhibitory population
		self.Input_to_I.rates = 0*Hz # reset activity
		# self.Input_to_I.rates[range(0, self.N_e_outp)] = self.coding_lvl*Hz
		self.Input_to_I.rates[range(0, self.N_e_outp)] = self.stim_freq_i

	def set_namespace(self):
		self.namespace = {
			'Vrst_e' : self.Vrst_e,
			'Vth_e_init' : self.Vth_e_init,
			'Vrst_i' : self.Vrst_i,
			'Vth_i' : self.Vth_i,
			'Vth_e_incr' : self.Vth_e_incr,
			'w_max' : self.w_e_e_max,
			'xpre_jump' : self.xpre_jump,
			'xpost_jump' : self.xpost_jump,
			'tau_xpre' : self.tau_xpre,
			'tau_xpost' : self.tau_xpost,
			'tau_rho' : self.tau_rho,
			'rho_min' : self.rho_min,
			'rho_max' : self.rho_max,
			'alpha' : self.alpha,
			'beta' : self.beta,
			'xpre_factor' : self.xpre_factor,
			'thr_b_rho' : self.thr_b_rho,
			'rho_neg' : self.rho_neg,
			'rho_neg2' : self.rho_neg2,
			'thr_post' : self.thr_post,
			'thr_pre' : self.thr_pre,
			'tau_xstop' : self.tau_xstop,
			'xstop_jump' : self.xstop_jump,
			'thr_up_h' : self.thr_up_h,
			'thr_up_l' : self.thr_up_l,
			'thr_down_h' : self.thr_down_h,
			'thr_down_l' : self.thr_down_l,
			'tau_xstop' : self.tau_xstop ,
			'xstop_jump' : self.xstop_jump,
			'xstop_max' : self.xstop_max,
			'xstop_min' : self.xstop_min, 
			'thr_stop_h' : self.thr_stop_h,
			'thr_stop_l' : self.thr_stop_l,
			'xpre_max' : self.xpre_max,
			'xpre_min' : self.xpre_min}

		return self.namespace

	def set_learning_rule_parameters(self):
		if self.plasticity_rule == 'LR1' or self.plasticity_rule =='LR2' or self.plasticity_rule =='LR3':
			[self.tau_xpre,
			self.tau_xpost,
			self.xpre_jump,
			self.xpost_jump,
			self.rho_neg,
			self.rho_neg2,
			self.rho_init,
			self.tau_rho,
			self.thr_post,
			self.thr_pre,
			self.thr_b_rho,
			self.rho_min,
			self.rho_max,
			self.alpha,
			self.beta,
			self.xpre_factor,
			self.w_max,
			self.xpre_min,
			self.xpost_min,
			self.xpost_max,
			self.xpre_max] = load_rule_params(self.plasticity_rule, 
				self.parameter_set)

	def run_net(self, report = 'stdout', period = 1):
		# Running simulation
		self.net.run(
			self.t_run, 
			report = report, 
			report_period = period*second,
			namespace = self.set_namespace())

		self.net.stop()

	def set_network_object(self):
		defaultclock.dt = self.dt_resolution

		self.net = Network(
			self.E_inp,
			self.Input_to_E_inp,
			self.E_outp,
			self.I,
			self.teacher_pop,
			self.spontaneous_act_pop,
			self.Input_to_I,
			self.Input_1st_layer,
			self.Input_to_Output,
			self.Input_I,
			self.I_Eout,
			self.teacher_Eout,
			self.spont_input,
			self.Input_to_E_inp_spkmon,
			self.E_inp_spkmon,
			self.E_outp_spkmon,
			self.Input_to_I_spkmon,
			self.I_spkmon,
			self.teacher_spkmon,
			self.spont_spkmon,
			self.Input_1st_layer_stamon,
			self.Input_I_stamon,
			self.Eout_stamon,
			self.Einp_stamon,
			self.I_stamon,
			self.Input_to_Output_stamon,
			self.I_Eout_stamon,
			name = 'net')
			




