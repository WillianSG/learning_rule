import os, sys
from progress.bar import Bar

# epochs_list = [' 1', ' 2', ' 3', ' 4']
epochs_list = [' 4']
time_list = [' 1.0']

num_repetitions = 5

bar = Bar(
	'Training networks', 
	max = len(epochs_list)*len(time_list)*num_repetitions)

for time in time_list:
	for epoch in epochs_list:
		for rep in range(0, num_repetitions):
			run_script = 'python train_test_feedforward_net.py' +  time + epoch
			os.system(run_script)

			bar.next()
bar.finish()