"""User defined initial Decision Tree for various environments."""

import numpy as np
from agents.dt.completeDT import genDT

def DT_cart():
    """
    Return a DT with 16 leaves for cartpole environment. Number of leaves is hard coded.
    # TODO: Take command line args input for number of leaves
    # W and B gets orthogonal initialization
    """
    dim_in, dim_out, W, B, init_leaves = genDT(env_name='cart', num_leaf=16)
    return dim_in, dim_out, W, B, init_leaves

# proposed in prolonet
def DT_cart_v0():
    """
    User defined DT for cartpole environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """

    dim_in = 4 # state dimension
    dim_out = 2 # action dimension
    num_nodes = 11

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: x0 + 1 > 0
    W[0, 0] = 1
    B[0] = 1

    # node 1: -x0 + 1 > 0
    W[1, 0] = -1
    B[1] = 1

    # node 2: -x2 + 0 > 0
    W[2, 2] = -1
    B[2] = 0

    # node 3: -x2 + 0 > 0
    W[3, 2] = -1
    B[3] = 0

    # node 4:  -x2 + 0 > 0
    W[4, 2] = -1
    B[4] = 0

    # node 5: -x1 + 0 > 0
    W[5, 1] = -1
    B[5] = 0

    # node 6: x1 + 0 > 0
    W[6, 1] = 1
    B[6] = 0

    # node 7: x3 + 0 > 0
    W[7, 3] = 1
    B[7] = 0

    # node 8: -x2 + 0 > 0
    W[8, 2] = -1

    # node 9: -x3 + 0 > 0
    W[9, 3] = -1
    B[9] = 0

    # node 10: -x2 + 0 > 0
    W[10, 2] = -1
    B[10] = 0

    # 2. Define Leaf Nodes and Paths to leaf nodes
    num_leaves = 12

    # SPP: Format: [True Conditions, False Conditions, Action]
    init_leaves = []
    l1 = [[], [0, 2], [0, 0]]
    l1[-1][1] = 1  # Right
    init_leaves.append(l1)

    l2 = [[0, 1, 3], [], [0, 0]]
    l2[-1][0] = 1  # Left
    init_leaves.append(l2)

    l3 = [[0, 1], [3], [0, 0]]
    l3[-1][1] = 1  # Right
    init_leaves.append(l3)

    l4 = [[0, 4], [1], [0, 0]]
    l4[-1][0] = 1  # Left
    init_leaves.append(l4)

    l5 = [[2, 5, 7], [0], [0, 0]]
    l5[-1][1] = 1  # Right
    init_leaves.append(l5)

    l6 = [[2, 5], [0, 7], [0, 0]]
    l6[-1][0] = 1  # Left
    init_leaves.append(l6)

    l7 = [[2, 8], [0, 5], [0, 0]]
    l7[-1][0] = 1  # Left
    init_leaves.append(l7)

    l8 = [[2], [0, 5, 8], [0, 0]]
    l8[-1][1] = 1  # Right
    init_leaves.append(l8)

    l9 = [[0, 6, 9], [1, 4], [0, 0]]
    l9[-1][0] = 1  # Left
    init_leaves.append(l9)

    l10 = [[0, 6], [1, 4, 9], [0, 0]]
    l10[-1][1] = 1  # Right
    init_leaves.append(l10)

    l11 = [[0, 10], [1, 4, 6], [0, 0]]
    l11[-1][0] = 1  # Left
    init_leaves.append(l11)

    l12 = [[0], [1, 4, 6, 10], [0, 0]]
    l12[-1][1] = 1  # Right
    init_leaves.append(l12)


    return dim_in, dim_out, W, B, init_leaves

