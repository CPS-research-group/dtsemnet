"""User defined initial Decision Tree for various environments."""

import numpy as np
from agents.dt.completeDT import genDT


def DT_acrobot_v1():
    """
    User defined DT for Mountain Car environment. Returns node and leaf weights which represent a DT.
    DT with 7 decision nodes and 4 action nodes.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        num_decision_nodes: Number of decision nodes
        num_action_nodes: Number of action nodes
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 6
    dim_out = 3
    num_decision_nodes = 7
    num_action_nodes = 4
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # ========== Decision Nodes ==========
    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 0

    # node 1: -x1 + 0 > 0
    W[1, 0] = -1
    B[1] = 0

    # node 2:
    W[2, 0] = 1
    B[2] = -0.1

    # node 3:
    W[3, 1] = 1
    B[3] = 0.1

    # node 4:
    W[4, 1] = 1
    B[4] = -0.2

    # node 5:
    W[5, 1] = -1
    B[5] = -0.1

    # node 6:
    W[6, 1] = 1
    B[6] = 0.3

    # ========== Action Nodes ==========
    B[7] = 0.1
    B[8] = 0.1
    B[9] = 0.1
    B[10] = 0.1
    # 11 nodes in total

    # 2. Define leaf nodes [[True Nodes], [False Nodes], [Action]]
    l0 = [[0, 1, 3], [], [1, 0, 0]]
    l1 = [[0, 1, 7], [3], [0, 1, 0]]
    l2 = [[0, 1], [3, 7], [0, 0, 1]]

    l3 = [[0, 4], [1], [1, 0, 0]]
    l4 = [[0, 8], [1, 4], [0, 1, 0]]
    l5 = [[0], [1, 4, 8], [0, 0, 1]]

    l6 = [[2, 5], [0], [1, 0, 0]]
    l7 = [[2, 9], [0, 5], [0, 1, 0]]
    l8 = [[2], [0, 5, 9], [0, 0, 1]]

    l9 = [[6], [0, 2], [1, 0, 0]]
    l10 = [[10], [0, 2, 6], [0, 1, 0]]
    l11 = [[], [0, 2, 6, 10], [0, 0, 1]]

    # 12 leaf nodes

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
    ]

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves

def DT_acrobot_minimal():
    return genDT(env_name='acrobot', num_leaf=16)

# ProloNet DT
def DT_acrobot_minimal_v0():
    """
    User defined DT for Mountain Car environment. Returns node and leaf weights which represent a DT.
    DT with 7 decision nodes and 4 action nodes.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        num_decision_nodes: Number of decision nodes
        num_action_nodes: Number of action nodes
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 6
    dim_out = 3
    num_nodes = 11

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # ========== Decision Nodes ==========
    W[0, 0] = 1
    B[0] = 0

    W[1, 2] = 1
    B[1] = 0

    W[2, 2] = 1
    B[2] = 0

    # nodes from 3 to 6
    for i in range(3, 7):
        W[i, 4] = 1
        B[i] = 0

    # nodes from 7 to 10
    for i in range(7, 11):
        W[i, 5] = 1
        B[i] = 0


    # ========== Leaf Nodes ==========
    # 2. Define leaf nodes [[True Nodes], [False Nodes], [Action]]
    l0 = [[0, 1, 3], [], [0, 1, 0]]
    l1 = [[0, 1, 7], [3], [1, 0, 0]]
    l2 = [[0, 1], [3, 7], [0, 0, 1]]

    l3 = [[0, 4], [1], [0, 1, 0]]
    l4 = [[0, 8], [1, 4], [1, 0, 0]]
    l5 = [[0], [1, 4, 8], [0, 0, 1]]

    l6 = [[2, 5], [0], [0, 1, 0]]
    l7 = [[2, 9], [0, 5], [1, 0, 0]]
    l8 = [[2], [0, 5, 9], [0, 0, 1]]

    l9 = [[6], [0, 2], [0, 1, 0]]
    l10 = [[10], [0, 2, 6], [1, 0, 0]]
    l11 = [[], [0, 2, 6, 10], [0, 0, 1]]

    # 12 leaf nodes

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
    ]

    return dim_in, dim_out, W, B, init_leaves
