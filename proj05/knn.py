import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Generate N samples in range [0, 1] in d dimensions
def gen_dcube_samples(N, d):
    return np.random.random((N, d))
    
def pairwise_distances(data):
    dists = []
    
    for i, a in enumerate(data):
        for j, b in enumerate(data):
            if i == j:
                continue          
            dists.append(np.linalg.norm(a-b))
            
    return dists
    

N = 100
dims = [2, 3, 10, 100, 1000, 10000]

fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(6, 6))

for i, dim in enumerate(dims[:3]):
    data = gen_dcube_samples(N, dim)
    dists = pairwise_distances(data)
    axs[0][i].hist(dists, bins=10, weights=np.ones(len(dists)) / len(dists), edgecolor='black',linewidth=1.2)
    axs[0][i].set_title('d=' + str(dim), size=9)
                                            
for i, dim in enumerate(dims[3:]):          
    data = gen_dcube_samples(N, dim)        
    dists = pairwise_distances(data)        
    axs[1][i].hist(dists, bins=10, weights=np.ones(len(dists)) / len(dists), edgecolor='black',linewidth=1.2)                   
    axs[1][i].set_title('d=' + str(dim), size=9)

fig.text(0.5, 0.03, 'Distance', ha='center')
fig.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical')
#plt.tight_layout()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()