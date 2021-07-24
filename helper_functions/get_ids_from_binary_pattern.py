# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Receive as input a binary array representing a pattern and return the IDs of the active neurons on the pattern.
"""

import numpy as np

def get_ids_from_binary_pattern(binarized_pattern):
	return np.nonzero(binarized_pattern)[0]