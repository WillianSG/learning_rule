# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- set total # neurons to 400 (20x20 pattern)
"""
import setuptools
import os, sys, pickle, shutil
import numpy as np
from time import localtime, strftime
import random

def main():
	# timestamp
	idt = localtime()
	timestamp = strftime("%d%b%Y_%H-%M-%S", localtime())

	dataset_size = int(sys.argv[1]) # total number of patterns generated
	stimulus_size = int(sys.argv[2]) # total number of neurons in pattern

	active_neurons = int((stimulus_size*5)/100) # number of neurons active in pattern

	# according to Fusi's paper, coding lvl should relate to 5% of stim. size
	coding_lvl = active_neurons/stimulus_size

	meta_data = {
		'dataset_size': dataset_size,
		'stimulus_size': stimulus_size,
		'active_neurons': active_neurons,
		'coding_lvl': coding_lvl,
		'timestamp': timestamp
	}

	class_0 = []
	class_1 = []

	print('\n- generating patterns (coding lvl: ' + str(coding_lvl) + ')')
	for loop_c in range(0, dataset_size):
		print('  pattern #', loop_c+1)

		pattern = np.zeros(stimulus_size)

		temp_selected = []
		
		for pos in range(0, active_neurons):

			# position to be altered to active state (1)
			position = np.random.randint(stimulus_size, size = 1)

			while (position[0] in temp_selected):
				position = np.random.randint(stimulus_size, size = 1)

			temp_selected.append(position[0])

			pattern[position[0]] = 1

		# separatting patterns into two equal size classes
		if (loop_c+1) % 2 == 0:
			class_0.append(pattern)
		else:
			class_1.append(pattern)

	# making np array
	class_0 = np.array(class_0)
	class_1 = np.array(class_1)

	# pickling dataset

	parent_dir = os.path.dirname(os.getcwd())

	# dataset destination
	dataset_dir = os.path.join(parent_dir, 'dataset_F')

	if not(os.path.isdir(dataset_dir)):
		os.mkdir(dataset_dir)

	fn = os.path.join(dataset_dir, timestamp + '_dataset_Fusi.pickle')

	with open(fn, 'wb') as f:
		pickle.dump((
			meta_data,
			class_0,
			class_1
			), f)

if __name__ == "__main__":
	main()

	print("\n- random_patterns.py - END\n")