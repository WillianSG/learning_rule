# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:
- Returns a list of input neuron indices that are active to form a stimulus pattern.

Comments:
"""

import random
import numpy as np
import sys

def load_stimulus_Ninp(stimulus_id):
	if stimulus_id == 'x':
		stimulus_neur_ids = [6,8,4,0,2]
	elif stimulus_id == '.':
		stimulus_neur_ids = [2, 4, 6]
	elif stimulus_id == '+':
		stimulus_neur_ids = [7,3,4,5,1]
	elif stimulus_id == 'ulc': # upper left corner
		stimulus_neur_ids = [6]
	elif stimulus_id == 'lrc': # lower right corner
		stimulus_neur_ids = [3]
	else:
		sys.exit('\n> ERROR - select valid stimulus (exiting)')

	return stimulus_neur_ids


