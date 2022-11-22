import cvxpy as cp
import numpy as np

# Constants
S = [[100, 200, 300], [50, 201, 200]]
C = [[50, 55, 75], [52.4, 55, 75]]

for (S1, S2, S3), (C1, C2, C3) in zip(S, C):
    # Variables
    # x1, x2, x3
    n = 3
    x = cp.Variable(shape=(n), name="x")

    # Constraints
    # x1 >= 0
    # x2 >= 0
    # x3 >= 0
    # x1*C1 + x2*C2 + x3 * C3 <= 3000000

    # x must be integers as well, but this can be ignored for this assignment
    # Set constraints
    constraints = np.array([x[0] >= 0,
                            x[1] >= 0,
                            x[2] >= 0,
                           (x[0] * C1 + x[1] * C2 + x[2] * C3) <= 3000000])

    # Set objective
    objective = x[0] * (S1 - C1) + x[1] * (S2 - C2) + x[2] * (S3 - C3)

    # Create problem
    prob = cp.Problem(cp.Maximize(objective), constraints)

    # Solve problem
    prob.solve()

    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)