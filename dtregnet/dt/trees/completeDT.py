"""Code to generate a complete binary tree with given leaf nodes."""

import math
import numpy as np

def generate_complete_binary_tree(num_leaf):
    dim_out = num_leaf
    leaf_action = [0] * dim_out #info: number of controllers is same as number of leaf nodes
    height = math.ceil(math.log2(num_leaf))
    leaf_nodes_lists = []
    stack = [(0, [], [])]

    contoller_num = 0

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[contoller_num] = 1
            contoller_num += 1
            leaf_nodes_lists.append([left_parents, right_parents, leaf_act])
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'
    return leaf_nodes_lists

def genDT(env_name ,num_leaf=8):
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Lunar lander with 7 internal nodes and 8 controllers

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    # info: Add environment input
    if env_name == 'lunar':
        dim_in = 8
        num_controls = 2
    elif env_name == 'cart':
        dim_in = 4
        num_controls = 1
    elif env_name == 'lane_keeping':
        dim_in = 12
        num_controls = 1
    elif env_name == 'ring_accel':
        dim_in = 44
        num_controls = 1
    elif env_name in ['highway', 'intersection', 'racetrack']:
        dim_in = 25
        num_controls = 2
    elif env_name in ['walker']:
        dim_in = 24
        num_controls = 4
    elif env_name == 'ugrid':
        dim_in = 6
        num_controls = 1


    dim_out = num_leaf
    num_nodes = dim_out - 1


    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.random.rand(num_nodes, dim_in)  # each row represents a node
    B = np.random.rand(num_nodes)  # Biases of each node

    init_leaves = generate_complete_binary_tree(num_leaf=dim_out)


    return dim_in, dim_out, W, B, init_leaves, num_controls
