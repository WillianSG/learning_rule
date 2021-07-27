# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
"""
import numpy as np

def histograms_firing_rate2(t_points, pop_size, bins_edges = []):
	"""
	t_hist_count = values of histogram (# spks in each bar)
	t_hist_edges: respective edges for the histogram bins - used as x coordinate of the bars in 'plt.bar()'
	"""
	t_hist_count, t_hist_edges = np.histogram(a = t_points, bins = bins_edges)

	# calculating bin width from histogram edges
	t_hist_bin_widths = []
	for i in np.arange(0, len(bins_edges)-1, 1):
		bin_width_temp = t_hist_edges[i + 1] - t_hist_edges[i]
		t_hist_bin_widths.append(np.round(bin_width_temp, 3))

	

	# calculating "frequency" per bin
	t_hist_freq = t_hist_count/t_hist_bin_widths/pop_size

	"""matplotlib.pyplot.bar(x, height, width)
	x: x-coordinates of the bars [array -> t_hist_edges]
	heigh: height of each bar [array -> t_hist_freq]
	width: width of each bar [array -> t_hist_bin_widths]
	"""
	return t_hist_edges[:-1], t_hist_freq, t_hist_bin_widths
