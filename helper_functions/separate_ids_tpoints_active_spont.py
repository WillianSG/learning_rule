# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- scripts receive as input arrays with spiking times and the respective ids of the neuron spiking. Returns spk times based on active/spontaneous neurons in the input layer.
"""

def separate_ids_tpoints_active_spont(
	input_tpoints, 
	input_ids, 
	active_input_ids):
	
	active_tpoints = []
	active_ids = []

	spontaneous_tpoints = []
	spontaneous_ids = []

	for a in range(0, len(input_ids)):
		if input_ids[a] in active_input_ids:
			active_ids.append(input_ids[a])
			active_tpoints.append(input_tpoints[a])
		else:
			spontaneous_ids.append(input_ids[a])
			spontaneous_tpoints.append(input_tpoints[a])

	return active_tpoints, active_ids, spontaneous_tpoints, spontaneous_ids
