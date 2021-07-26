# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- automatic reshape based on stimulus size
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle, os

with open(
	'26Jul2021_20-52-29__LR3_1.C_bistTrue_dict_array_mi_plus_metadata.pickle',
	'rb') as f:(
	dict_array_mi,
	dataset_metadata,
	plasticity_rule,
	parameter_set,
	bistability,
	stoplearning,
	populations_biasing_dict) = pickle.load(f)

for item in dict_array_mi:
	print(item['mi_per_pattern_c1'])

# print('================== dataset metadata ==================')
for key, value in dict_array_mi[1].items():
	print(key, ':', value)
# print('======================================================\n')