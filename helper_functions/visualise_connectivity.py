import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import *

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)

    fig = plt.figure(figsize = (10, 4))
    plt.subplot(121)

    plt.plot(zeros(Ns), arange(Ns), 'ok', ms = 10)
    plt.plot(ones(Nt), arange(Nt), 'ok', ms = 10)

    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')

    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')

    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')

    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

    fig.suptitle(S.name.replace('_', ' ') + ' connections', fontsize = 10)

    plt.show()