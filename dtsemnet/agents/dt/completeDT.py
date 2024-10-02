import math
import numpy as np

def generate_complete_binary_tree(num_leaf, dim_out):
    leaf_action = [0] * dim_out #info: number of controllers is same as number of leaf nodes
    height = math.ceil(math.log2(num_leaf))
    leaf_nodes_lists = []
    stack = [(0, [], [])]

    controller_num = 0

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[controller_num] = 1
            controller_num += 1
            if controller_num == dim_out:
                controller_num = 0
            leaf_nodes_lists.append([left_parents, right_parents, leaf_act])
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'
    return leaf_nodes_lists

def genDT(env_name = 'lunar', num_leaf=8):
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
    if env_name == 'lunar':
        dim_in = 8
        dim_out = 4
    elif env_name == 'cart':
        dim_in = 4
        dim_out = 2
    elif env_name == 'zerlings':
        dim_in = 38
        dim_out = 10
    elif env_name == 'acrobot':
        dim_in = 6
        dim_out = 3
    else:
        raise ValueError('Invalid env_name')

   
    num_nodes = num_leaf - 1


    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.random.rand(num_nodes, dim_in)  # each row represents a node
    B = np.random.rand(num_nodes)  # Biases of each node

    init_leaves = generate_complete_binary_tree(num_leaf=num_leaf, dim_out=dim_out)


    return dim_in, dim_out, W, B, init_leaves