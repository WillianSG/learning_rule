import numpy as np

def magnitude_change_percentage(initial_val, final_val):
	if final_val > initial_val:
		magnitude_change = final_val - initial_val
	else:
		magnitude_change = initial_val - final_val
		
	return np.round(((magnitude_change*100)/initial_val), 1)

