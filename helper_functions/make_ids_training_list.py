# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- dataset_size (int) = number of patterns in the dataset
- epoch (int) = training epoch of network

Functionality:
- Function returns the a shuffled list of pattern ids to be used during the training epoch 'epoch'. At each epoch eacher pattern presentation increases by one (e.g. in the 3rd epoch each pattern is randomly presented three times).
"""

import random

def make_ids_traning_list(dataset_size, epoch):
	original_ids_list = list(range(0, dataset_size))

	final_ids_list = []

	for i in range(0, epoch):
		final_ids_list += original_ids_list

	# random.shuffle(final_ids_list)

	return final_ids_list