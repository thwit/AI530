import random
import numpy as np
import pia as pia

# Function which generates random gamma, R, and P
def gammaRP(N, M):
    gamma = random.uniform(0, 1)
    
    
    P = np.zeros([N, N, M])
    
    for n in range(N):
        for n_ in range(N):
            for m in range(M):
                # Random chance that given transition is not available
                if random.uniform(0, 1) > 0.15:
                    P[n, n_, m] = -1.
                    continue
                    
                # random probability between 0 and 1
                P[n, n_, m] = random.uniform(0, 1)
    
    R = np.zeros([N, N, M])
    
    for n in range(N):
        for n_ in range(N):
            for m in range(M):
                # If transition is not available, reward is 0
                if P[n, n_, m] < 0:
                    continue
                # random reward between 0 and 1
                R[n, n_, m] = random.uniform(0, 1)
                
    return gamma, R, P
    
N = random.randint(3, 20)
M = random.randint(3, 20)
gamma, R, P = gammaRP(N, M)

V = pia.pia(gamma, R, P, N, M)
