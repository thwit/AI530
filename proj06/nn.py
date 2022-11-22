import numpy as np
import matplotlib.pyplot as plt

# Generate N samples in range [0, 1] in d dimensions from the hypercube
def gen_dcube_samples(N, d):
    return np.random.random((N, d))
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def NN(x1, x2, w1=0.65, w2=0.35):
    return sigmoid(w1 * x1 + w2 * x2)
    
def G(x1, x2):
    return 1 / np.sqrt(2*np.pi) * np.exp(-0.5 * (x1 * x1 + x2 * x2))
    

dataNN = []
x1s = [i for i in range(-10, 11)]
x2s = reversed(x1s)

for x1, x2 in zip(x1s, x2s):
    dataNN.append(NN(x1, x2))


#plt.plot(x1s, dataNN)
#plt.xlabel('w1 value (w2=1-w1)')
#plt.ylabel('NN')
#plt.show()

dataNN = []
dataG = []

minw1 = None
minw2 = None
minerror = np.inf

for i in range(100):
    w1 = i / 100
    for j in range(100):
        w2 = j / 100
        w2 = 1 - w1
        error = 0
        dcube = gen_dcube_samples(100, 2)
        for x1, x2 in dcube:
            error += NN(x1, x2, w1, w2) - G(x1, x2)
        
        if error < minerror:
            minerror = error
            minw1 = w1
            minw2 = w2
        
print()
print('Min. error: ' + str(minerror))
print('w1: ' + str(minw1))
print('w2: ' + str(minw2))