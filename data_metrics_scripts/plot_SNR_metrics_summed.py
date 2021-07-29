import os, pickle
import numpy as np
import matplotlib.pyplot as plt
 
# Get the list of all files and directories
sim_results_path = "C://Users//willi//PhD_Stuff//learning_rule//network_results_SNR_w_binarized_above0.5//feedforward_network"

dir_list = os.listdir(sim_results_path)

print('Number of folders: ', len(dir_list))

dataset_size = 10
dict_array_snr_summed = []

for pattern_id in range(0, dataset_size):
	dict_snr = {
	'pattern_id': pattern_id,
	'summed_snr_out1': 0.0,
	'summed_snr_out2': 0.0,
	'avg_snr_out1': 0.0,
	'avg_snr_out2': 0.0
	}

	dict_array_snr_summed.append(dict_snr)

def update_pattern_snr_sum(pattern_id, snr_out1, snr_out2, dict_array):
	for x in range(0, len(dict_array)):
		if dict_array[x]['pattern_id'] == pattern_id:
			dict_array[x]['summed_snr_out1'] += snr_out1
			dict_array[x]['summed_snr_out2'] += snr_out2
			break

def update_pattern_snr_avg(pattern_id, avg_snr_out1, avg_snr_out2, dict_array):
	for x in range(0, len(dict_array)):
		if dict_array[x]['pattern_id'] == pattern_id:
			dict_array[x]['avg_snr_out1'] = avg_snr_out1
			dict_array[x]['avg_snr_out2'] = avg_snr_out2
			break

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
			dict_array_snr,
			M_syn,
			correct_response,
			wrong_response,
			dataset_metadata,
			plasticity_rule,
			parameter_set,
			bistability,
			stoplearning,
			populations_biasing_dict) = pickle.load(f)

		#+++=================
		# a. open snr array
		for x in range(0, len(dict_array_snr)):
			# a.1 if len > 1 -> use arra[-1]
			#		else use arra[0]
			if len(dict_array_snr[x]['snr_ffrq_out1']) > 1:
				update_pattern_snr_sum(
					pattern_id = dict_array_snr[x]['pattern_id'], 
					snr_out1 = dict_array_snr[x]['snr_ffrq_out1'][-1], 
					snr_out2 = dict_array_snr[x]['snr_ffrq_out2'][-1], 
					dict_array = dict_array_snr_summed)
			else:
				update_pattern_snr_sum(
					pattern_id = dict_array_snr[x]['pattern_id'], 
					snr_out1 = dict_array_snr[x]['snr_ffrq_out1'][0], 
					snr_out2 = dict_array_snr[x]['snr_ffrq_out2'][0], 
					dict_array = dict_array_snr_summed)

			# b. sum SNR of each pattern per out neuron
			# c. avg them out (use counter)
			# d. plot avg SNR per pattern per out/class
			#+++=================

# plot_data[0] = out 1
plot_data_y = [ [], []]

plot_data_x = []

for x in range(0, len(dict_array_snr_summed)):
	update_pattern_snr_avg(
		pattern_id = dict_array_snr_summed[x]['pattern_id'], 
		avg_snr_out1 = dict_array_snr_summed[x]['summed_snr_out1']/len(dir_list), 
		avg_snr_out2 = dict_array_snr_summed[x]['summed_snr_out2']/len(dir_list), 
		dict_array = dict_array_snr_summed)

	plot_data_y[0].append(dict_array_snr_summed[x]['avg_snr_out1'])
	plot_data_y[1].append(dict_array_snr_summed[x]['avg_snr_out2'])

	plot_data_x.append(dict_array_snr_summed[x]['pattern_id'])

# # set width of bar
# barWidth = 0.25
# fig = plt.subplots(figsize =(12, 8))
 
# # set height of bar
# IT = plot_data_y[0]
# ECE = plot_data_y[1]
 
# # Set position of bar on X axis
# br1 = np.arange(len(IT))
# br2 = [x + barWidth for x in br1]
 
# # Make the plot
# plt.bar(br1, IT, color ='r', width = barWidth,
#         edgecolor ='grey', label ='IT')

# plt.bar(br2, ECE, color ='g', width = barWidth,
#         edgecolor ='grey', label ='ECE')
 
# # Adding Xticks
# plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
# plt.ylabel('Students passed', fontweight ='bold', fontsize = 15)
# plt.xticks([r + barWidth for r in range(len(IT))],
#         ['2015', '2016', '2017', '2018', '2019'])

x = np.arange(dataset_size)

plt.bar(x+0.1, plot_data_y[0], 0.2, label = 'out1', color = 'darkblue')
plt.bar(x-0.1, plot_data_y[1], 0.2, label = 'out2', color = 'tomato')
 
plt.legend()

plt.xticks(np.arange(
	0, 
	dataset_size,
	step = 1))

plt.ylabel('Average SNR', size = 10)
plt.xlabel('Pattern ID', size = 10)

# plt.xlim(0, dataset_size)

plt.show()

# print(plot_data_y[0])
# print(plot_data_y[1])










