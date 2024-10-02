"""User defined initial Decision Tree for various environments.

DT Structure is similar to what defines in ProLoNets.
# TODO: More elegant way to define DTs.

"""

import numpy as np
from agents.dt.completeDT import genDT

def DT_lunar():
    """
    Original DT written by ProLonet Authors.
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 14

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 1.1

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -0.2

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -0.1

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 0.1

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -0.9

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -0.1

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -0.9

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -0.9

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -0.2

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -0.9

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -0.2

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 0.1

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -0.2

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -0.2


    # 2. Define leaf nodes
    l0 = [[0, 1], [3], [0, 0, 0, 0]]
    l0[-1][3] = 1

    l1 = [[0, 4], [1], [0, 0, 0, 0]]
    l1[-1][0] = 1

    l2 = [[6], [0, 2], [0, 0, 0, 0]]
    l2[-1][0] = 1

    l3 = [[], [0, 2, 6], [0, 0, 0, 0]]
    l3[-1][3] = 1

    l4 = [[0, 1, 3, 7], [], [0, 0, 0, 0]]
    l4[-1][0] = 1

    l5 = [[0, 8], [1, 4], [0, 0, 0, 0]]
    l5[-1][1] = 1

    l6 = [[2, 5, 9], [0], [0, 0, 0, 0]]
    l6[-1][0] = 1

    l7 = [[2, 5], [0, 9], [0, 0, 0, 0]]
    l7[-1][1] = 1

    l8 = [[2, 10], [0, 5], [0, 0, 0, 0]]
    l8[-1][1] = 1

    l9 = [[0, 1, 3, 11], [7], [0, 0, 0, 0]]
    l9[-1][2] = 1

    l10 = [[0, 1, 3], [7, 11], [0, 0, 0, 0]]
    l10[-1][1] = 1

    l11 = [[0, 12], [1, 4, 8], [0, 0, 0, 0]]
    l11[-1][3] = 1

    l12 = [[0], [1, 4, 8, 12], [0, 0, 0, 0]]
    l12[-1][0] = 1

    l13 = [[2, 13], [0, 5, 10], [0, 0, 0, 0]]
    l13[-1][3] = 1

    l14 = [[2], [0, 5, 10, 13], [0, 0, 0, 0]]
    l14[-1][0] = 1

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14
    ]

    num_decision_nodes = 14
    num_action_nodes = 0

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves

def DT_lunar2():
    """
    TEST Version. Not used in the experiments.
    ?Leaf Nodes are update according to the trained PROLONET Model?
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 14

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 1.1

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -0.2

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -0.1

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 0.1

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -0.9

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -0.1

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -0.9

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -0.9

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -0.2

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -0.9

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -0.2

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 0.1

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -0.2

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -0.2

    # 2. Define leaf nodes
    l0 = [[0, 1], [3], [0, 0, 0, 0]]
    l0[-1][3] = 1

    l1 = [[0, 4], [1], [0, 0, 0, 0]]
    l1[-1][1] = 1

    l2 = [[6], [0, 2], [0, 0, 0, 0]]
    l2[-1][0] = 1

    l3 = [[], [0, 2, 6], [0, 0, 0, 0]]
    l3[-1][3] = 1

    l4 = [[0, 1, 3, 7], [], [0, 0, 0, 0]]
    l4[-1][3] = 1

    l5 = [[0, 8], [1, 4], [0, 0, 0, 0]]
    l5[-1][0] = 1

    l6 = [[2, 5, 9], [0], [0, 0, 0, 0]]
    l6[-1][0] = 1

    l7 = [[2, 5], [0, 9], [0, 0, 0, 0]]
    l7[-1][1] = 1

    l8 = [[2, 10], [0, 5], [0, 0, 0, 0]]
    l8[-1][1] = 1

    l9 = [[0, 1, 3, 11], [7], [0, 0, 0, 0]]
    l9[-1][2] = 1

    l10 = [[0, 1, 3], [7, 11], [0, 0, 0, 0]]
    l10[-1][0] = 1

    l11 = [[0, 12], [1, 4, 8], [0, 0, 0, 0]]
    l11[-1][3] = 1

    l12 = [[0], [1, 4, 8, 12], [0, 0, 0, 0]]
    l12[-1][0] = 1

    l13 = [[2, 13], [0, 5, 10], [0, 0, 0, 0]]
    l13[-1][2] = 1

    l14 = [[2], [0, 5, 10, 13], [0, 0, 0, 0]]
    l14[-1][2] = 1

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14
    ]

    return dim_in, dim_out, W, B, init_leaves

def DT_lunar_minimal(nodes=31):
    '''Which DT to use? It will be a complete binary tree with following number of nodes
    1. 31
    2. 63
    3. 127
    '''

    assert nodes in [31, 63, 127], "Invalid number of nodes"
    if nodes == 31:
        return DT_lunar_minimal_0()
    elif nodes == 63:
        return DT_lunar_minimal_1()
    elif nodes == 127:
        return DT_lunar_minimal_2()
    else:
        raise NotImplementedError

# SPP: with 31 nodes
def DT_lunar_minimal_0():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 31

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 1.1

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -0.2

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -0.1

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 0.1

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -0.9

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -0.1

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -0.9

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -0.9

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -0.2

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -0.9

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -0.2

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 0.1

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -0.2

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -0.2

    # node 14: -x0 - 0.2 > 0
    W[14, 0] = 1
    B[14] = 0.5

    # node 15: -x1 + 1.1 > 0
    W[15, 1] = -1
    B[15] = 1.1

    # node 16: -x3 - 0.2 > 0
    W[16, 3] = -1
    B[16] = -0.2

    # node 17: x5 - 0.1 > 0
    W[17, 5] = 1
    B[17] = -0.1

    # node 18: -x5 + 0.1 > 0
    W[18, 5] = -1
    B[18] = 0.1

    # node 19: x6 + x7 - 0.9 > 0
    W[19, 6] = 1
    W[19, 7] = 1
    B[19] = -0.9

    # node 20: -x5 -0.1 > 0
    W[20, 5] = -1
    B[20] = -0.1

    # node 21: x6 + x7 - 0.9 > 0
    W[21, 6] = 1
    W[21, 7] = 1
    B[21] = -0.9

    # node 22: x6 + x7 - 0.9 > 0
    W[22, 6] = 1
    W[22, 7] = 1
    B[22] = -0.9

    # node 23: x0 - 0.2 > 0
    W[23, 0] = 1
    B[23] = -0.2

    # node 24: x6 + x7 - 0.9 > 0
    W[24, 6] = 1
    W[24, 7] = 1
    B[24] = -0.9

    # node 25: x0 - 0.2 > 0
    W[25, 0] = 1
    B[25] = -0.2

    # node 26: x5 + 0.1 > 0
    W[26, 5] = 1
    B[26] = 0.1

    # node 27: -x0 - 0.2 > 0
    W[27, 0] = -1
    B[27] = -0.2

    # node 28: -x0 - 0.2 > 0
    W[28, 0] = -1
    B[28] = -0.2

    # node 29: -x0 - 0.2 > 0
    W[29, 0] = 1
    B[29] = 0.5

    # node 30: x5 + 0.1 > 0
    W[30, 5] = 1
    B[30] = 0.1

    # 2. Define leaf nodes [T] [F] [Action]
    l0 = [[0, 1, 3, 7, 15], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7], [15], [0, 1, 0, 0]]
    l2 = [[0, 1, 3, 16], [7], [0, 0, 1, 0]]
    l3 = [[0, 1, 3], [7, 16], [0, 0, 0, 1]]

    l4 = [[0, 1, 8, 17], [3], [1, 0, 0, 0]]
    l5 = [[0, 1, 8], [3, 17], [0, 1, 0, 0]]
    l6 = [[0, 1, 18], [3, 8], [0, 0, 1, 0]]
    l7 = [[0, 1], [3, 8, 18], [0, 0, 0, 1]]

    l8 = [[0, 4, 9, 19], [1], [1, 0, 0, 0]]
    l9 = [[0, 4, 9], [1, 19], [0, 1, 0, 0]]
    l10 = [[0, 4, 20], [1, 9], [0, 0, 1, 0]]
    l11 = [[0, 4], [1, 9, 20], [0, 0, 0, 1]]

    l12 = [[0, 10, 21], [1, 4], [1, 0, 0, 0]]
    l13 = [[0, 10], [1, 4, 21], [0, 1, 0, 0]]
    l14 = [[0, 22], [1, 4, 10], [0, 0, 1, 0]]
    l15 = [[0], [1, 4, 10, 22], [0, 0, 0, 1]]

    l16 = [[2, 5, 11, 23], [0], [1, 0, 0, 0]]
    l17 = [[2, 5, 11], [0, 23], [0, 1, 0, 0]]
    l18 = [[2, 5, 24], [0, 11], [0, 0, 1, 0]]
    l19 = [[2, 5], [0, 11, 24], [0, 0, 0, 1]]

    l20 = [[2, 12, 25], [0, 5], [1, 0, 0, 0]]
    l21 = [[2, 12], [0, 5, 25], [0, 1, 0, 0]]
    l22 = [[2, 26], [0, 5, 12], [0, 0, 1, 0]]
    l23 = [[2], [0, 5, 12, 26], [0, 0, 0, 1]]

    l24 = [[6, 13, 27], [0, 2], [1, 0, 0, 0]]
    l25 = [[6, 13], [0, 2, 27], [0, 1, 0, 0]]
    l26 = [[6, 28], [0, 2, 13], [0, 0, 1, 0]]
    l27 = [[6], [0, 2, 13, 28], [0, 0, 0, 1]]

    l28 = [[14, 29], [0, 2, 6], [1, 0, 0, 0]]
    l29 = [[14], [0, 2, 6, 29], [0, 1, 0, 0]]
    l30 = [[30], [0, 2, 6, 14], [0, 0, 1, 0]]
    l31 = [[], [0, 2, 6, 14, 30], [0, 0, 0, 1]]


    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27,
        l28, l29, l30, l31
    ]



    return dim_in, dim_out, W, B, init_leaves

# SPP: minimal with 63 internal nodes
def DT_lunar_minimal_1():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 63

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 0] = 1


    # node 1 to 2
    for i in range(1,3):
        W[i, 1] = 1


    # node 3 to 6
    for i in range(3, 7):
        W[i, 2] = 1


    # node 7 to 14
    for i in range(7, 15):
        W[i, 3] = 1


    # node 15 to 30
    for i in range(15, 31):
        W[i, 4] = 1


    # node 31 to 62
    for i in range(31, 63):
        W[i, 5] = 1


    # total of 63 internal decision nodes


    # 2. Define leaf nodes [[True Nodes] [False Nodes] [Action]]
    l0 = [[0, 1, 3, 7, 15, 31], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7, 15], [31], [0, 0, 0, 1]]
    l2 = [[0, 1, 3, 7, 32], [15], [0, 1, 0, 0]]
    l3 = [[0, 1, 3, 7], [15, 32], [0, 0, 1, 0]]

    l4 = [[0, 1, 3, 16, 33], [7], [1, 0, 0, 0]]
    l5 = [[0, 1, 3, 16], [7, 33], [0, 0, 0, 1]]
    l6 = [[0, 1, 3, 34], [7, 16], [0, 1, 0, 0]]
    l7 = [[0, 1, 3], [7, 16, 34], [0, 0, 1, 0]]

    l8 = [[0, 1, 8, 17, 35], [3], [1, 0, 0, 0]]
    l9 = [[0, 1, 8, 17], [3, 35], [0, 0, 0, 1]]
    l10 = [[0, 1, 8, 36], [3, 17], [0, 1, 0, 0]]
    l11 = [[0, 1, 8], [3, 17, 36], [0, 0, 1, 0]]

    l12 = [[0, 1, 18, 37], [3, 8], [1, 0, 0, 0]]
    l13 = [[0, 1, 18], [3, 8, 37], [0, 0, 0, 1]]
    l14 = [[0, 1, 38], [3, 8, 18], [0, 1, 0, 0]]
    l15 = [[0, 1], [3, 8, 18, 38], [0, 0, 1, 0]]

    l16 = [[0, 4, 9, 19, 39], [1], [1, 0, 0, 0]]
    l17 = [[0, 4, 9, 19], [1, 39], [0, 0, 0, 1]]
    l18 = [[0, 4, 9, 40], [1, 19], [0, 1, 0, 0]]
    l19 = [[0, 4, 9], [1, 19, 40], [0, 0, 1, 0]]

    l20 = [[0, 4, 20, 41], [1, 9], [1, 0, 0, 0]]
    l21 = [[0, 4, 20], [1, 9, 41], [0, 0, 0, 1]]
    l22 = [[0, 4, 42], [1, 9, 20], [0, 1, 0, 0]]
    l23 = [[0, 4], [1, 9, 20, 42], [0, 0, 1, 0]]

    l24 = [[0, 10, 21, 43], [1, 4], [1, 0, 0, 0]]
    l25 = [[0, 10, 21], [1, 4, 43], [0, 0, 0, 1]]
    l26 = [[0, 10, 44], [1, 4, 21], [0, 1, 0, 0]]
    l27 = [[0, 10], [1, 4, 21, 44], [0, 0, 1, 0]]

    l28 = [[0, 22, 45], [1, 4, 10], [1, 0, 0, 0]]
    l29 = [[0, 22], [1, 4, 10, 45], [0, 0, 0, 1]]
    l30 = [[0, 46], [1, 4, 10, 22], [0, 1, 0, 0]]
    l31 = [[0], [1, 4, 10, 22, 46], [0, 0, 1, 0]]

    l32 = [[2, 5, 11, 23, 47], [0], [1, 0, 0, 0]]
    l33 = [[2, 5, 11, 23], [0, 47], [0, 0, 0, 1]]
    l34 = [[2, 5, 11, 48], [0, 23], [0, 1, 0, 0]]
    l35 = [[2, 5, 11], [0, 23, 48], [0, 0, 1, 0]]

    l36 = [[2, 5, 24, 49], [0, 11], [1, 0, 0, 0]]
    l37 = [[2, 5, 24], [0, 11, 49], [0, 0, 0, 1]]
    l38 = [[2, 5, 50], [0, 11, 24], [0, 1, 0, 0]]
    l39 = [[2, 5], [0, 11, 24, 50], [0, 0, 1, 0]]

    l40 = [[2, 12, 25, 51], [0, 5], [1, 0, 0, 0]]
    l41 = [[2, 12, 25], [0, 5, 51], [0, 0, 0, 1]]
    l42 = [[2, 12, 52], [0, 5, 25], [0, 1, 0, 0]]
    l43 = [[2, 12], [0, 5, 25, 52], [0, 0, 1, 0]]

    l44 = [[2, 26, 53], [0, 5, 12], [1, 0, 0, 0]]
    l45 = [[2, 26], [0, 5, 12, 53], [0, 0, 0, 1]]
    l46 = [[2, 54], [0, 5, 12, 26], [0, 1, 0, 0]]
    l47 = [[2], [0, 5, 12, 26, 54], [0, 0, 1, 0]]

    l48 = [[6, 13, 27, 55], [0, 2], [1, 0, 0, 0]]
    l49 = [[6, 13, 27], [0, 2, 55], [0, 0, 0, 1]]
    l50 = [[6, 13, 56], [0, 2, 27], [0, 1, 0, 0]]
    l51 = [[6, 13], [0, 2, 27, 56], [0, 0, 1, 0]]

    l52 = [[6, 28, 57], [0, 2, 13], [1, 0, 0, 0]]
    l53 = [[6, 28], [0, 2, 13, 57], [0, 0, 0, 1]]
    l54 = [[6, 58], [0, 2, 13, 28], [0, 1, 0, 0]]
    l55 = [[6], [0, 2, 13, 28, 58], [0, 0, 1, 0]]

    l56 = [[14, 29, 59], [0, 2, 6], [1, 0, 0, 0]]
    l57 = [[14, 29], [0, 2, 6, 59], [0, 0, 0, 1]]
    l58 = [[14, 60], [0, 2, 6, 29], [0, 1, 0, 0]]
    l59 = [[14], [0, 2, 6, 29, 60], [0, 0, 1, 0]]

    l60 = [[30, 61], [0, 2, 6, 14], [1, 0, 0, 0]]
    l61 = [[30], [0, 2, 6, 14, 61], [0, 0, 0, 1]]
    l62 = [[62], [0, 2, 6, 14, 30], [0, 1, 0, 0]]
    l63 = [[], [0, 2, 6, 14, 30, 62], [0, 0, 1, 0]]

    # 64 leaf nodes



    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59, l60, l61, l62, l63
    ]

    return dim_in, dim_out, W, B, init_leaves

# SPP: minimal with 127 nodes
def DT_lunar_minimal_2():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 127

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 0] = 1


    # node 1 to 2
    for i in range(1,3):
        W[i, 1] = 1


    # node 3 to 6
    for i in range(3, 7):
        W[i, 2] = 1


    # node 7 to 14
    for i in range(7, 15):
        W[i, 3] = 1


    # node 15 to 30
    for i in range(15, 31):
        W[i, 4] = 1


    # node 31 to 62
    for i in range(31, 63):
        W[i, 5] = 1

    # node 63 to 126
    for i in range(63, 127):
        W[i, 6] = 1


    # total of 63 internal decision nodes


    # 2. Define leaf nodes [[True Nodes] [False Nodes] [Action]]
    l0 = [[0, 1, 3, 7, 15, 31], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7, 15], [31], [0, 0, 0, 1]]
    l2 = [[0, 1, 3, 7, 32], [15], [0, 1, 0, 0]]
    l3 = [[0, 1, 3, 7], [15, 32], [0, 0, 1, 0]]

    l4 = [[0, 1, 3, 16, 33], [7], [1, 0, 0, 0]]
    l5 = [[0, 1, 3, 16], [7, 33], [0, 0, 0, 1]]
    l6 = [[0, 1, 3, 34], [7, 16], [0, 1, 0, 0]]
    l7 = [[0, 1, 3], [7, 16, 34], [0, 0, 1, 0]]

    l8 = [[0, 1, 8, 17, 35], [3], [1, 0, 0, 0]]
    l9 = [[0, 1, 8, 17], [3, 35], [0, 0, 0, 1]]
    l10 = [[0, 1, 8, 36], [3, 17], [0, 1, 0, 0]]
    l11 = [[0, 1, 8], [3, 17, 36], [0, 0, 1, 0]]

    l12 = [[0, 1, 18, 37], [3, 8], [1, 0, 0, 0]]
    l13 = [[0, 1, 18], [3, 8, 37], [0, 0, 0, 1]]
    l14 = [[0, 1, 38], [3, 8, 18], [0, 1, 0, 0]]
    l15 = [[0, 1], [3, 8, 18, 38], [0, 0, 1, 0]]

    l16 = [[0, 4, 9, 19, 39], [1], [1, 0, 0, 0]]
    l17 = [[0, 4, 9, 19], [1, 39], [0, 0, 0, 1]]
    l18 = [[0, 4, 9, 40], [1, 19], [0, 1, 0, 0]]
    l19 = [[0, 4, 9], [1, 19, 40], [0, 0, 1, 0]]

    l20 = [[0, 4, 20, 41], [1, 9], [1, 0, 0, 0]]
    l21 = [[0, 4, 20], [1, 9, 41], [0, 0, 0, 1]]
    l22 = [[0, 4, 42], [1, 9, 20], [0, 1, 0, 0]]
    l23 = [[0, 4], [1, 9, 20, 42], [0, 0, 1, 0]]

    l24 = [[0, 10, 21, 43], [1, 4], [1, 0, 0, 0]]
    l25 = [[0, 10, 21], [1, 4, 43], [0, 0, 0, 1]]
    l26 = [[0, 10, 44], [1, 4, 21], [0, 1, 0, 0]]
    l27 = [[0, 10], [1, 4, 21, 44], [0, 0, 1, 0]]

    l28 = [[0, 22, 45], [1, 4, 10], [1, 0, 0, 0]]
    l29 = [[0, 22], [1, 4, 10, 45], [0, 0, 0, 1]]
    l30 = [[0, 46], [1, 4, 10, 22], [0, 1, 0, 0]]
    l31 = [[0], [1, 4, 10, 22, 46], [0, 0, 1, 0]]

    l32 = [[2, 5, 11, 23, 47], [0], [1, 0, 0, 0]]
    l33 = [[2, 5, 11, 23], [0, 47], [0, 0, 0, 1]]
    l34 = [[2, 5, 11, 48], [0, 23], [0, 1, 0, 0]]
    l35 = [[2, 5, 11], [0, 23, 48], [0, 0, 1, 0]]

    l36 = [[2, 5, 24, 49], [0, 11], [1, 0, 0, 0]]
    l37 = [[2, 5, 24], [0, 11, 49], [0, 0, 0, 1]]
    l38 = [[2, 5, 50], [0, 11, 24], [0, 1, 0, 0]]
    l39 = [[2, 5], [0, 11, 24, 50], [0, 0, 1, 0]]

    l40 = [[2, 12, 25, 51], [0, 5], [1, 0, 0, 0]]
    l41 = [[2, 12, 25], [0, 5, 51], [0, 0, 0, 1]]
    l42 = [[2, 12, 52], [0, 5, 25], [0, 1, 0, 0]]
    l43 = [[2, 12], [0, 5, 25, 52], [0, 0, 1, 0]]

    l44 = [[2, 26, 53], [0, 5, 12], [1, 0, 0, 0]]
    l45 = [[2, 26], [0, 5, 12, 53], [0, 0, 0, 1]]
    l46 = [[2, 54], [0, 5, 12, 26], [0, 1, 0, 0]]
    l47 = [[2], [0, 5, 12, 26, 54], [0, 0, 1, 0]]

    l48 = [[6, 13, 27, 55], [0, 2], [1, 0, 0, 0]]
    l49 = [[6, 13, 27], [0, 2, 55], [0, 0, 0, 1]]
    l50 = [[6, 13, 56], [0, 2, 27], [0, 1, 0, 0]]
    l51 = [[6, 13], [0, 2, 27, 56], [0, 0, 1, 0]]

    l52 = [[6, 28, 57], [0, 2, 13], [1, 0, 0, 0]]
    l53 = [[6, 28], [0, 2, 13, 57], [0, 0, 0, 1]]
    l54 = [[6, 58], [0, 2, 13, 28], [0, 1, 0, 0]]
    l55 = [[6], [0, 2, 13, 28, 58], [0, 0, 1, 0]]

    l56 = [[14, 29, 59], [0, 2, 6], [1, 0, 0, 0]]
    l57 = [[14, 29], [0, 2, 6, 59], [0, 0, 0, 1]]
    l58 = [[14, 60], [0, 2, 6, 29], [0, 1, 0, 0]]
    l59 = [[14], [0, 2, 6, 29, 60], [0, 0, 1, 0]]

    l60 = [[30, 61], [0, 2, 6, 14], [1, 0, 0, 0]]
    l61 = [[30], [0, 2, 6, 14, 61], [0, 0, 0, 1]]
    l62 = [[62], [0, 2, 6, 14, 30], [0, 1, 0, 0]]
    l63 = [[], [0, 2, 6, 14, 30, 62], [0, 0, 1, 0]]

    # 64 leaf nodes
    lf_actions = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59, l60, l61, l62, l63
    ]

    # add 2 more leaf to each leaf node to increase the number of layers
    new_init_leaves = []
    for i in range(64):
        t1 = init_leaves[i][0]
        t2 = init_leaves[i][1]
        t3 = t1.copy()
        t4 = t2.copy()

        t1.append(63 + i)
        t4.append(63 + i)

        left_leaf = [t1, t2, lf_actions[ 2*(i%2) ]]
        right_leaf = [t3, t4, lf_actions[2*(i%2) + 1]]

        new_init_leaves.append(left_leaf)
        new_init_leaves.append(right_leaf)

    init_leaves = new_init_leaves



    return dim_in, dim_out, W, B, init_leaves


def DT_lunar_v1():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.
    This is an extented version of original DT where the number of decision nodes are 14 and the number of action nodes are 45,
    hence total number of nodes are 59. Total number of leaf nodes are 60.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        num_decision_nodes: Number of decision nodes
        num_action_nodes: Number of action nodes
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_decision_nodes = 14
    num_action_nodes = 45
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # ========== Decision Nodes ==========
    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 1.1

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -0.2

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -0.1

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 0.1

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -0.9

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -0.1

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -0.9

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -0.9

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -0.2

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -0.9

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -0.2

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 0.1

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -0.2

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -0.2

    # ========== Action Nodes ==========
    B[14] = -0.1
    B[20] = -0.1

    B[16] = 0.1
    B[27] = 0.1

    B[17] = -0.1
    B[30] = -0.1

    B[15] = 0.1
    B[21] = 0.1

    B[23] = 0.1
    B[35] = -0.1

    B[24] = 0.1
    B[39] = 0.1

    B[25] = 0.1
    B[41] = -0.1

    B[26] = 0.1
    B[43] = -0.1

    B[18] = 0.1
    B[31] = 0.1

    B[33] = -0.1
    B[48] = 0.1

    B[34] = 0.1
    B[49] = -0.1

    B[37] = -0.1
    B[52] = -0.1

    B[38] = 0.1
    B[53] = 0.1

    B[45] = -0.1
    B[56] = -0.1

    B[46] = 0.1
    B[57] = 0.1





    # 59 nodes in total


    # 2. Define leaf nodes [[True Nodes], [False Nodes], [Action]]
    l0 = [[0, 1, 3, 7, 18, 31], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7, 18], [31], [0, 1, 0, 0]]
    l2 = [[0, 1, 3, 7, 32], [18], [0, 0, 1, 0]]
    l3 = [[0, 1, 3, 7], [18, 32], [0, 0, 0, 1]]

    l4 = [[0,1,14,19], [3], [1, 0, 0, 0]]
    l5 = [[0, 1, 14], [3, 19], [0, 1, 0, 0]]
    l6 = [[0, 1, 20], [3, 14], [0, 0, 1, 0]]
    l7 = [[0, 1], [3, 14, 20], [0, 0, 0, 1]]

    l8 = [[0, 4, 15, 21], [1], [1, 0, 0, 0]]
    l9 = [[0, 4, 15], [1, 21], [0, 1, 0, 0]]
    l10 = [[0, 4, 22], [1, 15], [0, 0, 1, 0]]
    l11 = [[0, 4], [1, 15, 22], [0, 0, 0, 1]]

    l12 = [[6, 16, 27], [0, 2], [1, 0, 0, 0]]
    l13 = [[6, 16], [0, 2, 27], [0, 1, 0, 0]]
    l14 = [[6, 28], [0, 2, 16], [0, 0, 1, 0]]
    l15 = [[6], [0, 2, 16, 28], [0, 0, 0, 1]]

    l16 = [[17, 29], [0, 2, 6], [1, 0, 0, 0]]
    l17 = [[17], [0, 2, 6, 29], [0, 1, 0, 0]]
    l18 = [[30], [0, 2, 6, 17], [0, 0, 1, 0]]
    l19 = [[], [0, 2, 6, 17, 30], [0, 0, 0, 1]]

    l20 = [[0, 8, 23, 35], [1, 4], [1, 0, 0, 0]]
    l21 = [[0, 8, 23], [1, 4, 35], [0, 1, 0, 0]]
    l22 = [[0, 8, 36], [1, 4, 23], [0, 0, 1, 0]]
    l23 = [[0, 8], [1, 4, 23, 36], [0, 0, 0, 1]]

    l24 = [[2, 5, 9, 24, 39], [0], [1, 0, 0, 0]]
    l25 = [[2, 5, 9, 24], [0, 39], [0, 1, 0, 0]]
    l26 = [[2, 5, 9, 40], [0, 24], [0, 0, 1, 0]]
    l27 = [[2, 5, 9], [0, 24, 40], [0, 0, 0, 1]]

    l28 = [[2, 5, 25, 41], [0, 9], [1, 0, 0, 0]]
    l29 = [[2, 5, 25], [0, 9, 41], [0, 1, 0, 0]]
    l30 = [[2, 5, 42], [0, 9, 25], [0, 0, 1, 0]]
    l31 = [[2, 5], [0, 9, 25, 42], [0, 0, 0, 1]]

    l32 = [[2, 10, 26, 43], [0, 5], [1, 0, 0, 0]]
    l33 = [[2, 10, 26], [0, 5, 43], [0, 1, 0, 0]]
    l34 = [[2, 10, 44], [0, 5, 26], [0, 0, 1, 0]]
    l35 = [[2, 10], [0, 5, 26, 44], [0, 0, 0, 1]]

    l36 = [[0, 1, 3, 11, 33, 47], [7], [1, 0, 0, 0]]
    l37 = [[0, 1, 3, 11, 33], [7, 47], [0, 1, 0, 0]]
    l38 = [[0, 1, 3, 11, 48], [33], [0, 0, 1, 0]]
    l39 = [[0, 1, 3, 11], [33, 48], [0, 0, 0, 1]]

    l40 = [[0, 1, 3, 34, 49], [7, 11], [1, 0, 0, 0]]
    l41 = [[0, 1, 3, 34], [7, 11, 49], [0, 1, 0, 0]]
    l42 = [[0, 1, 3, 50], [7, 11, 34], [0, 0, 1, 0]]
    l43 = [[0, 1, 3], [7, 11, 34, 50], [0, 0, 0, 1]]

    l44 = [[0, 12, 37, 51], [1, 4, 8], [1, 0, 0, 0]]
    l45 = [[0, 12, 37], [1, 4, 8, 51], [0, 1, 0, 0]]
    l46 = [[0, 12, 52], [1, 4, 8, 37], [0, 0, 1, 0]]
    l47 = [[0, 12], [1, 4, 8, 37, 52], [0, 0, 0, 1]]

    l48 = [[0, 38, 53], [1, 4, 8, 12], [1, 0, 0, 0]]
    l49 = [[0, 38], [1, 4, 8, 12, 53], [0, 1, 0, 0]]
    l50 = [[0, 54], [1, 4, 8, 12, 38], [0, 0, 1, 0]]
    l51 = [[0], [1, 4, 8, 12, 38, 54], [0, 0, 0, 1]]

    l52 = [[2, 13, 45, 55], [0, 5, 10], [1, 0, 0, 0]]
    l53 = [[2, 13, 45], [0, 5, 10, 55], [0, 1, 0, 0]]
    l54 = [[2, 13, 56], [0, 5, 10, 45], [0, 0, 1, 0]]
    l55 = [[2, 13], [0, 5, 10, 45, 56], [0, 0, 0, 1]]

    l56 = [[2, 46, 57], [0, 5, 10, 13], [1, 0, 0, 0]]
    l57 = [[2, 46], [0, 5, 10, 13, 57], [0, 1, 0, 0]]
    l58 = [[2, 58], [0, 5, 10, 13, 46], [0, 0, 1, 0]]
    l59 = [[2], [0, 5, 10, 13, 46, 58], [0, 0, 0, 1]]


    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59
    ]

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves


## DT action nodes at leaf corrected
def DT_lunar_minimal_0_v1():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 31

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 1.1

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -0.2

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -0.1

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 0.1

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -0.9

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -0.1

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -0.9

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -0.9

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -0.2

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -0.9

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -0.2

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 0.1

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -0.2

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -0.2

    # node 14: -x0 - 0.2 > 0
    W[14, 0] = 1
    B[14] = 0.5

    # node 15: -x1 + 1.1 > 0
    W[15, 1] = -1
    B[15] = 1.1

    # node 16: -x3 - 0.2 > 0
    W[16, 3] = -1
    B[16] = -0.2

    # node 17: x5 - 0.1 > 0
    W[17, 5] = 1
    B[17] = -0.1

    # node 18: -x5 + 0.1 > 0
    W[18, 5] = -1
    B[18] = 0.1

    # node 19: x6 + x7 - 0.9 > 0
    W[19, 6] = 1
    W[19, 7] = 1
    B[19] = -0.9

    # node 20: -x5 -0.1 > 0
    W[20, 5] = -1
    B[20] = -0.1

    # node 21: x6 + x7 - 0.9 > 0
    W[21, 6] = 1
    W[21, 7] = 1
    B[21] = -0.9

    # node 22: x6 + x7 - 0.9 > 0
    W[22, 6] = 1
    W[22, 7] = 1
    B[22] = -0.9

    # node 23: x0 - 0.2 > 0
    W[23, 0] = 1
    B[23] = -0.2

    # node 24: x6 + x7 - 0.9 > 0
    W[24, 6] = 1
    W[24, 7] = 1
    B[24] = -0.9

    # node 25: x0 - 0.2 > 0
    W[25, 0] = 1
    B[25] = -0.2

    # node 26: x5 + 0.1 > 0
    W[26, 5] = 1
    B[26] = 0.1

    # node 27: -x0 - 0.2 > 0
    W[27, 0] = -1
    B[27] = -0.2

    # node 28: -x0 - 0.2 > 0
    W[28, 0] = -1
    B[28] = -0.2

    # node 29: -x0 - 0.2 > 0
    W[29, 0] = 1
    B[29] = 0.5

    # node 30: x5 + 0.1 > 0
    W[30, 5] = 1
    B[30] = 0.1

    # 2. Define leaf nodes [T] [F] [Action]
    l0 = [[0, 1, 3, 7, 15], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7], [15], [0, 0, 1, 0]]
    l2 = [[0, 1, 3, 16], [7], [0, 1, 0, 0]]
    l3 = [[0, 1, 3], [7, 16], [0, 0, 0, 1]]

    l4 = [[0, 1, 8, 17], [3], [1, 0, 0, 0]]
    l5 = [[0, 1, 8], [3, 17], [0, 0, 1, 0]]
    l6 = [[0, 1, 18], [3, 8], [0, 1, 0, 0]]
    l7 = [[0, 1], [3, 8, 18], [0, 0, 0, 1]]

    l8 = [[0, 4, 9, 19], [1], [1, 0, 0, 0]]
    l9 = [[0, 4, 9], [1, 19], [0, 0, 1, 0]]
    l10 = [[0, 4, 20], [1, 9], [0, 1, 0, 0]]
    l11 = [[0, 4], [1, 9, 20], [0, 0, 0, 1]]

    l12 = [[0, 10, 21], [1, 4], [1, 0, 0, 0]]
    l13 = [[0, 10], [1, 4, 21], [0, 0, 1, 0]]
    l14 = [[0, 22], [1, 4, 10], [0, 1, 0, 0]]
    l15 = [[0], [1, 4, 10, 22], [0, 0, 0, 1]]

    l16 = [[2, 5, 11, 23], [0], [1, 0, 0, 0]]
    l17 = [[2, 5, 11], [0, 23], [0, 0, 1, 0]]
    l18 = [[2, 5, 24], [0, 11], [0, 1, 0, 0]]
    l19 = [[2, 5], [0, 11, 24], [0, 0, 0, 1]]

    l20 = [[2, 12, 25], [0, 5], [1, 0, 0, 0]]
    l21 = [[2, 12], [0, 5, 25], [0, 0, 1, 0]]
    l22 = [[2, 26], [0, 5, 12], [0, 1, 0, 0]]
    l23 = [[2], [0, 5, 12, 26], [0, 0, 0, 1]]

    l24 = [[6, 13, 27], [0, 2], [1, 0, 0, 0]]
    l25 = [[6, 13], [0, 2, 27], [0, 0, 1, 0]]
    l26 = [[6, 28], [0, 2, 13], [0, 1, 0, 0]]
    l27 = [[6], [0, 2, 13, 28], [0, 0, 0, 1]]

    l28 = [[14, 29], [0, 2, 6], [1, 0, 0, 0]]
    l29 = [[14], [0, 2, 6, 29], [0, 0, 1, 0]]
    l30 = [[30], [0, 2, 6, 14], [0, 1, 0, 0]]
    l31 = [[], [0, 2, 6, 14, 30], [0, 0, 0, 1]]

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31
    ]

    return dim_in, dim_out, W, B, init_leaves


# SPP: minimal with 63 internal nodes
def DT_lunar_minimal_1_v1():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 8
    dim_out = 4
    num_nodes = 63

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 0] = 1

    # node 1 to 2
    for i in range(1, 3):
        W[i, 1] = 1

    # node 3 to 6
    for i in range(3, 7):
        W[i, 2] = 1

    # node 7 to 14
    for i in range(7, 15):
        W[i, 3] = 1

    # node 15 to 30
    for i in range(15, 31):
        W[i, 4] = 1

    # node 31 to 62
    for i in range(31, 63):
        W[i, 5] = 1

    # total of 63 internal decision nodes

    # 2. Define leaf nodes [[True Nodes] [False Nodes] [Action]]
    l0 = [[0, 1, 3, 7, 15, 31], [], [1, 0, 0, 0]]
    l1 = [[0, 1, 3, 7, 15], [31], [0, 0, 1, 0]]
    l2 = [[0, 1, 3, 7, 32], [15], [0, 1, 0, 0]]
    l3 = [[0, 1, 3, 7], [15, 32], [0, 0, 0, 1]]

    l4 = [[0, 1, 3, 16, 33], [7], [1, 0, 0, 0]]
    l5 = [[0, 1, 3, 16], [7, 33], [0, 0, 1, 0]]
    l6 = [[0, 1, 3, 34], [7, 16], [0, 1, 0, 0]]
    l7 = [[0, 1, 3], [7, 16, 34], [0, 0, 0, 1]]

    l8 = [[0, 1, 8, 17, 35], [3], [1, 0, 0, 0]]
    l9 = [[0, 1, 8, 17], [3, 35], [0, 0, 1, 0]]
    l10 = [[0, 1, 8, 36], [3, 17], [0, 1, 0, 0]]
    l11 = [[0, 1, 8], [3, 17, 36], [0, 0, 0, 1]]

    l12 = [[0, 1, 18, 37], [3, 8], [1, 0, 0, 0]]
    l13 = [[0, 1, 18], [3, 8, 37], [0, 0, 1, 0]]
    l14 = [[0, 1, 38], [3, 8, 18], [0, 1, 0, 0]]
    l15 = [[0, 1], [3, 8, 18, 38], [0, 0, 0, 1]]

    l16 = [[0, 4, 9, 19, 39], [1], [1, 0, 0, 0]]
    l17 = [[0, 4, 9, 19], [1, 39], [0, 0, 1, 0]]
    l18 = [[0, 4, 9, 40], [1, 19], [0, 1, 0, 0]]
    l19 = [[0, 4, 9], [1, 19, 40], [0, 0, 0, 1]]

    l20 = [[0, 4, 20, 41], [1, 9], [1, 0, 0, 0]]
    l21 = [[0, 4, 20], [1, 9, 41], [0, 0, 1, 0]]
    l22 = [[0, 4, 42], [1, 9, 20], [0, 1, 0, 0]]
    l23 = [[0, 4], [1, 9, 20, 42], [0, 0, 0, 1]]

    l24 = [[0, 10, 21, 43], [1, 4], [1, 0, 0, 0]]
    l25 = [[0, 10, 21], [1, 4, 43], [0, 0, 1, 0]]
    l26 = [[0, 10, 44], [1, 4, 21], [0, 1, 0, 0]]
    l27 = [[0, 10], [1, 4, 21, 44], [0, 0, 0, 1]]

    l28 = [[0, 22, 45], [1, 4, 10], [1, 0, 0, 0]]
    l29 = [[0, 22], [1, 4, 10, 45], [0, 0, 1, 0]]
    l30 = [[0, 46], [1, 4, 10, 22], [0, 1, 0, 0]]
    l31 = [[0], [1, 4, 10, 22, 46], [0, 0, 0, 1]]

    l32 = [[2, 5, 11, 23, 47], [0], [1, 0, 0, 0]]
    l33 = [[2, 5, 11, 23], [0, 47], [0, 0, 1, 0]]
    l34 = [[2, 5, 11, 48], [0, 23], [0, 1, 0, 0]]
    l35 = [[2, 5, 11], [0, 23, 48], [0, 0, 0, 1]]

    l36 = [[2, 5, 24, 49], [0, 11], [1, 0, 0, 0]]
    l37 = [[2, 5, 24], [0, 11, 49], [0, 0, 1, 0]]
    l38 = [[2, 5, 50], [0, 11, 24], [0, 1, 0, 0]]
    l39 = [[2, 5], [0, 11, 24, 50], [0, 0, 0, 1]]

    l40 = [[2, 12, 25, 51], [0, 5], [1, 0, 0, 0]]
    l41 = [[2, 12, 25], [0, 5, 51], [0, 0, 1, 0]]
    l42 = [[2, 12, 52], [0, 5, 25], [0, 1, 0, 0]]
    l43 = [[2, 12], [0, 5, 25, 52], [0, 0, 0, 1]]

    l44 = [[2, 26, 53], [0, 5, 12], [1, 0, 0, 0]]
    l45 = [[2, 26], [0, 5, 12, 53], [0, 0, 1, 0]]
    l46 = [[2, 54], [0, 5, 12, 26], [0, 1, 0, 0]]
    l47 = [[2], [0, 5, 12, 26, 54], [0, 0, 0, 1]]

    l48 = [[6, 13, 27, 55], [0, 2], [1, 0, 0, 0]]
    l49 = [[6, 13, 27], [0, 2, 55], [0, 0, 1, 0]]
    l50 = [[6, 13, 56], [0, 2, 27], [0, 1, 0, 0]]
    l51 = [[6, 13], [0, 2, 27, 56], [0, 0, 0, 1]]

    l52 = [[6, 28, 57], [0, 2, 13], [1, 0, 0, 0]]
    l53 = [[6, 28], [0, 2, 13, 57], [0, 0, 1, 0]]
    l54 = [[6, 58], [0, 2, 13, 28], [0, 1, 0, 0]]
    l55 = [[6], [0, 2, 13, 28, 58], [0, 0, 0, 1]]

    l56 = [[14, 29, 59], [0, 2, 6], [1, 0, 0, 0]]
    l57 = [[14, 29], [0, 2, 6, 59], [0, 0, 1, 0]]
    l58 = [[14, 60], [0, 2, 6, 29], [0, 1, 0, 0]]
    l59 = [[14], [0, 2, 6, 29, 60], [0, 0, 0, 1]]

    l60 = [[30, 61], [0, 2, 6, 14], [1, 0, 0, 0]]
    l61 = [[30], [0, 2, 6, 14, 61], [0, 0, 1, 0]]
    l62 = [[62], [0, 2, 6, 14, 30], [0, 1, 0, 0]]
    l63 = [[], [0, 2, 6, 14, 30, 62], [0, 0, 0, 1]]

    # 64 leaf nodes

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59, l60, l61, l62, l63
    ]

    return dim_in, dim_out, W, B, init_leaves
