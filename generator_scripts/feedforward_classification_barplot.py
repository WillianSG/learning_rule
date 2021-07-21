import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
correct = [10, , , ]
wrong = [10, , , ]
 
# Set position of bar on X axis
br1 = np.arange(len(correct))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, correct, color ='lightblue', width = barWidth,
        edgecolor ='k', label ='Correct')
plt.bar(br2, wrong, color ='tomato', width = barWidth,
        edgecolor ='k', label ='Wrong')
 
# Adding Xticks
plt.xlabel('# epochs', fontweight ='bold', fontsize = 8)
plt.ylabel('# patterns', fontweight ='bold', fontsize = 8)

plt.xticks([r + barWidth for r in range(len(correct))],
        ['1', '2', '3', '4'])
 
plt.legend()
plt.show()