import os, sys

epochs_list = [' 1', ' 2', ' 3', ' 4', ' 5']
time_list = [' 1.0']

num_repetitions = 10

for time in time_list:
	for epoch in epochs_list:
		for rep in range(0, num_repetitions):
			run_script = 'python train_test_feedforward_net.py' +  time + epoch
			os.system(run_script)