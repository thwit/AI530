import numpy as np
import random

# Constant value deciding if 0 or -1 (or any other value) indicates no connection between two vertices
NOCON = 0

# Traveling Salesman Problem
# Author: Thomas Aamand Witting
# email:  wittingt@oregonstate.edu
##### DESCRIPTION #####
# The implemented algorithm is a greedy approach at solving the Traveling Salesman Problem.
# The algorithm will consider multiple visits to the same vertex if necessary - however, it
# will only do this in cases where it is strictly necessary - i.e. no other option is available.
# In a such case, it will always go to an unexplored vertex if possible, no matter if it is
# cheaper to do otherwise. In general, the algorithm will always travel along the cheapest edge
# which goes to an unvisited vertex.
#
# Arguments:
#   M: weighted adjacency matrix, numpy array of dim(n, n)
#   K: Max computational steps, int
def tsp(M, K):
    optimal_tour = [0]
    optimal_value = 0
    
    # Current city, initialize to 0 (city 1)
    c_cur = 0
    
    # Blacklisted city used when backtracking
    backtrack = False
    blacklist = -1
    
    for _ in range(K):        
        min_cost = np.inf
        min_city = -1
        
        # Find closest relevant neighbour
        for j in range(len(M)):
            # If we just backtracked we do not want to explore a
            # previously discovered route
            if backtrack and j == blacklist:
                continue
            
            # Check if neighbour is closer than previous neighbours
            if M[c_cur][j] != NOCON and j not in optimal_tour and M[c_cur][j] < min_cost:
                min_cost = M[c_cur][j]
                min_city = j
           
        # Reset backtracking variables
        backtrack = False
        blacklist = -1
        
        # If no relevant neighbour is found, it either means
        # we have visited all cities or that we are stuck
        if min_city == -1:
            # If we have visited all cities
            if len(optimal_tour) >= len(M):
                tour_idx = len(optimal_tour) - 1
                city = c_cur
                while tour_idx >= 0:
                    # See if we can go directly back to initial city: if yes, we're done.
                    # Otherwise go back through optimal_tour until we are able to (but keeping the already visited cities in the tour)
                    if M[city][0] != NOCON:
                        optimal_tour.append(0)
                        optimal_value = 0
                        
                        # Recompute optimal value along the tour
                        for i in range(1, len(optimal_tour)):
                            optimal_value += M[optimal_tour[i-1]][optimal_tour[i]]
    
                        # return optimal tour and its cost. (adds 1 to each city 'name' to be consistent with the
                        # assignment notation (1-index based)
                        return [[c + 1 for c in optimal_tour], optimal_value]
                    else:
                        optimal_tour.append(optimal_tour[tour_idx-1])
                        optimal_value = M[city][optimal_tour[tour_idx-1]]
                        tour_idx -= 1
                        city = optimal_tour[tour_idx]
                        
                raise Exception("Could not find a route in Graph")
                 
            # If we are stuck
            else:
                # Backtrack to random city
                # Get random index in optimal_tour that is not
                # the latest city
                idx = random.randint(0, len(optimal_tour) - 2)
                backtrack = True
                
                # Blacklist the closest neighbour to the random city
                blacklist = optimal_tour[idx+1]
                
                # Update variables
                c_cur = optimal_tour[idx]
                optimal_tour = optimal_tour[:idx+1]
                optimal_value = 0
                
                for i in range(len(optimal_tour) - 1):
                    optimal_value += M[i][optimal_tour[i+1]]
                
                # Continue search
                continue
                
        c_next = min_city
            
        # Update tour and cost of tour so far
        optimal_tour.append(c_next)
        optimal_value += M[c_cur][c_next]
        
        c_cur = c_next
        
    raise Exception("Could not find a route in K steps")
                
                

# Actual cheapest cost: 88
# Found cost: 88
M1 = np.array([[0, 1, 0, 0, 1], 
[1, 0, 42, 1, 0], 
[0, 42, 0, 0, 0], 
[0, 1, 0, 0, 1], 
[1, 0, 0, 1, 0]])

# Actual cheapest cost: 5
# Found cost: 5
M2 = np.array([[0, 1, 0, 0, 1], 
[1, 0, 42, 1, 1], 
[0, 42, 0, 1, 1], 
[0, 1, 1, 0, 1], 
[1, 1, 1, 1, 0]])

# Actual cheapest cost: 37
# Found cost: can't solve in reasonable K? (bug with code is likely)
M3 = np.array([[0, 1, 0, 0, 1, 0, 1, 0], 
[1, 0, 42, 1, 1, 10, 0, 0], 
[0, 42, 0, 1, 1, 0, 0, 1], 
[0, 1, 1, 0, 1, 0, 0, 0], 
[1, 1, 1, 1, 0, 4, 0, 0], 
[0, 10, 0, 0, 4, 0, 0, 0], 
[1, 0, 0, 0, 0, 0, 0, 10], 
[0, 0, 1, 0, 0, 0, 10, 0]])


# Example of "weakness"
# "Weakness" as it depends on context:
# Do we want cheaper cost or less vertex visits on the tour?
# Actual cheapest cost: 18
# Found cost: 113
M4 = np.array([[0, 1, 0, 0, 0, 100, 0, 0, 0, 0], 
[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
[0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
[100, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
[0, 0, 0, 0, 0, 0, 1, 0, 1, 0], 
[0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])



K = 1000000
tour, cost = tsp(M5, K)
print("Cost of tour: " + str(cost))
print("Optimal tour: " + str(tour))