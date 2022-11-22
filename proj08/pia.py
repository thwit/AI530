import numpy as np
import random

# Initializes V and PI
def init(P, N, M):
    V = np.zeros(N)
    PI = np.zeros((N, M))
    
    # Initialize PI
    for s in range(N):
        for a in range(M):
            n_actions = sum(P[s, s_, a] > 0 for s_ in range(N))
            PI[s, a] = 1 / n_actions if n_actions > 0 else 0
    
    return V, PI
    
# Policy evaluation
def evaluate(V, PI, R, P, gamma, states, actions):
    delta = 0
    
    # Stop criterion
    EPS = 0.001
    while delta > EPS:
        for s in states:
            v = V[s]
            V[s] = 0
            
            # Update V
            for s_ in states:
                for a in actions:
                    r = R[s, s_, a]
                    p = P[s, s_, a]
                    if p < 0:
                        continue
                    V[s] += PI[s, a] * p * (r + gamma * V[s_])
                delta = max(delta, abs(v - V[s]))
    return V

def improve(V, PI, R, P, gamma, states, actions):
    EPS = 0.001
    
    # Stop criterion
    stable = False
    while not stable:
        delta = 0
        stable = True
        # Update PI
        for s in states:
            for a in actions:
                pi = PI[s, a]
                for s_ in states:
                    r = R[s, s_, a]
                    p = P[s, s_, a]
                    PI[s, a] += p * (r + gamma * V[s_])
                    
                if abs(pi - PI[s, a]) > EPS:
                    stable = False
                delta += abs(pi - PI[s, a])
        
        # Perform an evaluation
        evaluate(V, PI, R, P, gamma, states, actions)
        
    return V
    
            
# Main function to call
def pia(gamma, R, P, N, M):
    V, PI = init(P, N, M)
    states = [i for i in range(N)]
    actions = [i for i in range(M)]
    V = evaluate(V, PI, R, P, gamma, states, actions)
    V = improve(V, PI, R, P, gamma, states, actions)
    
    return V