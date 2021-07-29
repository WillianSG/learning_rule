import os, pickle
import numpy as np
import matplotlib.pyplot as plt
 
# Get the list of all files and directories
sim_results_path = "C://Users//willi//PhD_Stuff//learning_rule//network_results_CR_varying_epoch_w_binary//feedforward_network"

dir_list = os.listdir(sim_results_path)

print('Number of folders: ', len(dir_list))

# pickled data format
# with open(fn, 'wb') as f:
# pickle.dump((
	# network.network_id,
	# network.t_run/second,
	# num_epochs,
	# correct_response,
	# wrong_response,
	# correct_rate
# 	), f)


cr_1_epoch = []
cr_2_epoch = []
cr_3_epoch = []
cr_4_epoch = []

for folder in dir_list:
	# open each folder
	sim_folder_n = os.path.join(
		sim_results_path,
		folder)

	folder_n_files_list = os.listdir(sim_folder_n)

	# open pickled data
	if len(folder_n_files_list) == 1:
		pickled_file = os.path.join(sim_folder_n, folder_n_files_list[0])

		with open(pickled_file,'rb') as f:(
			network_id,
			t_run,
			num_epochs,
			correct_response,
			wrong_response,
			correct_rate,
			plasticity_rule,
			parameter_set) = pickle.load(f)

		# append all CR of same epoch in diff arrays
		if num_epochs == 1:
			cr_1_epoch.append(correct_rate)
		if num_epochs == 2:
			cr_2_epoch.append(correct_rate)
		if num_epochs == 3:
			cr_3_epoch.append(correct_rate)
		if num_epochs == 4:
			cr_4_epoch.append(correct_rate)

# calc CR avg and std
cr_1_epoch_avg = np.mean(cr_1_epoch)
cr_2_epoch_avg = np.mean(cr_2_epoch)
cr_3_epoch_avg = np.mean(cr_3_epoch)
cr_4_epoch_avg = np.mean(cr_4_epoch)


cr_1_epoch_std = np.std(cr_1_epoch)
cr_2_epoch_std = np.std(cr_2_epoch)
cr_3_epoch_std = np.std(cr_3_epoch)
cr_4_epoch_std = np.std(cr_4_epoch)

# bar plot with error bar
x = [1, 2, 3, 4]
y = [cr_1_epoch_avg, cr_2_epoch_avg, cr_3_epoch_avg, cr_4_epoch_avg]

plt.bar(x, y)

err_bar = [cr_1_epoch_std, cr_2_epoch_std, cr_3_epoch_std, cr_4_epoch_std]

plt.errorbar(x, y, yerr = err_bar, fmt = 'o', color = 'r')

plt.ylabel('CR', size = 10)
plt.xlabel('Epochs', size = 10)
  
plt.show()