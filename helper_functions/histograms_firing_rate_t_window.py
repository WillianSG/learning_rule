# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- t_window (int): desired bin with in time (ms)
"""
import numpy as np
from brian2 import second, ms

def histograms_firing_rate_t_window(
	t_points, 
	sim_t = 0.0*second, 
	t_window = 10, 
	t_start = 0.0*second):
	
	sim_t = sim_t/second
	t_start = t_start/second

	sim_t_ms = (sim_t)*1000
	step_ms = t_window/1000

	bins = np.arange(t_start, t_start+sim_t+step_ms, step = step_ms)

	hist, bins = np.histogram(t_points, bins = bins)

	return hist, bins
