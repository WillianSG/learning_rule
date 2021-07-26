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

# dict_mi = {
# 'pre_i': i, 
# 'post_j': j,
# 'rho': 0.0,
# 'num_patterns_c1': -1.0,
# 'num_patterns_c2': -1.0,
# 'mi_per_pattern_c1': [],
# 'pattern_c1_ids': [],
# 'mi_per_pattern_c2': [],
# 'pattern_c2_ids': [],
# 'avg_mi_c1': 0.0,
# 'avg_mi_c2': 0.0,
# 'std_mi_c1': 0.0,
# 'std_mi_c2': 0.0
# }

with open(
	'26Jul2021_22-03-29__LR3_1.C_bistTrue_dict_array_mi_plus_metadata.pickle',
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