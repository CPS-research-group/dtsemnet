"""Converts DT to ProLoNet
Copied from ProloNet repo.
"""
import numpy as np
import sys

from agents.prolonet.arch import ProLoNet
import torch

from utils.sc_build_hellions_helpers import EXPANDED_TYPES


def init_cart_nets(distribution,
                   use_gpu=False,
                   vectorized=False,
                   randomized=False):
    dim_in = 4
    dim_out = 2
    w1 = np.zeros(dim_in)
    c1 = np.ones(dim_in) * 2
    w1[0] = 1  # cart position
    c1[0] = -1  # > -1

    w2 = np.zeros(dim_in)
    c2 = np.ones(dim_in) * 2
    w2[0] = -1  # negative position
    c2[0] = -1  # < 1  (so if positive < 4)

    w3 = np.zeros(dim_in)
    c3 = np.ones(dim_in) * 2
    w3[2] = -1  # pole angle
    c3[2] = 0  # < 0

    w4 = np.zeros(dim_in)
    c4 = np.ones(dim_in) * 2
    w4[2] = -1
    c4[2] = 0  # < 0

    w5 = np.zeros(dim_in)
    c5 = np.ones(dim_in) * 2
    w5[2] = -1
    c5[2] = 0  # < 0

    w6 = np.zeros(dim_in)
    c6 = np.ones(dim_in) * 2
    w6[1] = -1  # cart velocity
    c6[1] = 0  # < 0

    w7 = np.zeros(dim_in)
    c7 = np.ones(dim_in) * 2
    w7[1] = 1  # cart velocity
    c7[1] = 0  # > 0

    w8 = np.zeros(dim_in)
    c8 = np.ones(dim_in) * 2
    w8[3] = 1  # pole rate
    c8[3] = 0  # > 0

    w9 = np.zeros(dim_in)
    c9 = np.ones(dim_in) * 2
    w9[2] = -1
    c9[2] = 0

    w10 = np.zeros(dim_in)
    c10 = np.ones(dim_in) * 2
    w10[3] = -1
    c10[3] = 0

    w11 = np.zeros(dim_in)
    c11 = np.ones(dim_in) * 2
    w11[2] = -1
    c11[2] = 0

    init_weights = [
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]

    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * 2

    # SPP: Format: [True Conditions, False Conditions, Action]
    l1 = [[], [0, 2], leaf_base.copy()]
    l1[-1][1] = leaf_target_init_val  # Right

    l2 = [[0, 1, 3], [], leaf_base.copy()]
    l2[-1][0] = leaf_target_init_val  # Left

    l3 = [[0, 1], [3], leaf_base.copy()]
    l3[-1][1] = leaf_target_init_val  # Right

    l4 = [[0, 4], [1], leaf_base.copy()]
    l4[-1][0] = leaf_target_init_val  # Left

    l5 = [[2, 5, 7], [0], leaf_base.copy()]
    l5[-1][1] = leaf_target_init_val  # Right

    l6 = [[2, 5], [0, 7], leaf_base.copy()]
    l6[-1][0] = leaf_target_init_val  # Left

    l7 = [[2, 8], [0, 5], leaf_base.copy()]
    l7[-1][0] = leaf_target_init_val  # Left

    l8 = [[2], [0, 5, 8], leaf_base.copy()]
    l8[-1][1] = leaf_target_init_val  # Right

    l9 = [[0, 6, 9], [1, 4], leaf_base.copy()]
    l9[-1][0] = leaf_target_init_val  # Left

    l10 = [[0, 6], [1, 4, 9], leaf_base.copy()]
    l10[-1][1] = leaf_target_init_val  # Right

    l11 = [[0, 10], [1, 4, 6], leaf_base.copy()]
    l11[-1][0] = leaf_target_init_val  # Left

    l12 = [[0], [1, 4, 6, 10], leaf_base.copy()]
    l12[-1][1] = leaf_target_init_val  # Right

    init_leaves = [
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
        l12,
    ]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators
                            ]  # SPP: why vectorized? It's just s>c
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None
        init_comparators = None
        init_leaves = 4


    # SPP: weights, comparators, leaves are passed to ProLoNet
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False,
                              device='cuda' if use_gpu else 'cpu',
                              vectorized=vectorized)
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True,
                             device='cuda' if use_gpu else 'cpu',
                             vectorized=vectorized)
    if use_gpu:
        action_network = action_network.cuda()
        value_network = value_network.cuda()
    return action_network, value_network


def add_level(pro_lo_net, split_noise_scale=0.2, use_gpu=False):
    old_weights = pro_lo_net.layers  # Get the weights out
    new_weights = [
        weight.detach().clone().data.cpu().numpy() for weight in old_weights
    ]

    old_comparators = pro_lo_net.comparators  # get the comparator values out
    new_comparators = [
        comp.detach().clone().data.cpu().numpy() for comp in old_comparators
    ]
    old_leaf_information = pro_lo_net.leaf_init_information  # get the leaf init info out

    new_leaf_information = []

    for leaf_index in range(len(old_leaf_information)):
        prior_leaf = pro_lo_net.action_probs[leaf_index].detach().clone().cpu(
        ).numpy()
        leaf_information = old_leaf_information[leaf_index]
        left_path = leaf_information[0]
        right_path = leaf_information[1]
        # This hideousness is to handle empty sequences... get the index of the node this used to split at
        weight_index = max(max(max(left_path, [-1])),
                           max(max(right_path, [-1])))

        new_weight = np.random.normal(scale=split_noise_scale,
                                      size=old_weights[weight_index].size()[0])
        new_comparator = np.random.normal(
            scale=split_noise_scale,
            size=old_comparators[weight_index].size()[0])

        new_weights.append(new_weight)  # Add it to the list of nodes
        new_comparators.append(new_comparator)  # Add it to the list of nodes

        new_node_ind = len(new_weights) - 1  # Remember where we put it

        # Create our two new leaves
        new_leaf1 = np.random.normal(scale=split_noise_scale,
                                     size=prior_leaf.shape)
        new_leaf2 = np.random.normal(scale=split_noise_scale,
                                     size=prior_leaf.shape)

        # Create the paths, which are copies of the old path but now with a left / right at the new node
        new_leaf1_left = left_path.copy()
        new_leaf1_right = right_path.copy()
        new_leaf2_left = left_path.copy()
        new_leaf2_right = right_path.copy()
        # Leaf 1 goes left at the new node, leaf 2 goes right
        new_leaf1_left.append(new_node_ind)
        new_leaf2_right.append(new_node_ind)

        new_leaf_information.append(
            [new_leaf1_left, new_leaf1_right, new_leaf1])
        new_leaf_information.append(
            [new_leaf2_left, new_leaf2_right, new_leaf2])

        new_network = ProLoNet(input_dim=pro_lo_net.input_dim,
                               weights=new_weights,
                               comparators=new_comparators,
                               leaves=new_leaf_information,
                               alpha=pro_lo_net.alpha.item(),
                               is_value=pro_lo_net.is_value,
                               device='cuda' if use_gpu else 'cpu',
                               vectorized=pro_lo_net.vectorized,
                               output_dim=pro_lo_net.output_dim)
        if use_gpu:
            new_network = new_network.cuda()
    return new_network


def swap_in_node(network, deeper_network, leaf_index, use_gpu=False):
    """
    Duplicates the network and returns a new one, where the node at leaf_index as been turned into a splitting node
    with two leaves that are slightly noisy copies of the previous node
    :param network: prolonet in
    :param deeper_network: deeper_network to take the new node / leaves from
    :param leaf_index: index of leaf to turn into a split
    :return: new prolonet (value or normal)
    """
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    leaf_information = network.leaf_init_information[
        leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]
    if deeper_network is not None:
        deeper_weights = [
            weight.detach().clone().data.cpu().numpy()
            for weight in deeper_network.layers
        ]

        deeper_comparators = [
            comp.detach().clone().data.cpu().numpy()
            for comp in deeper_network.comparators
        ]

        deeper_leaf_info = deeper_network.leaf_init_information[leaf_index * 2]
        deeper_left_path = deeper_leaf_info[0]
        deeper_right_path = deeper_leaf_info[1]
        deeper_weight_index = max(max(max(deeper_left_path, [-1])),
                                  max(max(deeper_right_path, [-1])))

        # Make a new weight vector, mostly the same as the old one

        new_weight = deeper_weights[deeper_weight_index]
        new_comparator = deeper_comparators[deeper_weight_index]
        new_leaf1 = deeper_network.action_probs[
            leaf_index * 2].detach().clone().data.cpu().numpy()
        new_leaf2 = deeper_network.action_probs[
            leaf_index * 2 + 1].detach().clone().data.cpu().numpy()
    else:
        new_weight = np.random.normal(scale=0.2, size=old_weights[0].size()[0])
        new_comparator = np.random.normal(scale=0.2,
                                          size=old_comparators[0].size()[0])
        new_leaf1 = np.random.normal(
            scale=0.2, size=network.action_probs[leaf_index].size()[0])
        new_leaf2 = np.random.normal(
            scale=0.2, size=network.action_probs[leaf_index].size()[0])

    new_weights = [
        weight.detach().clone().data.cpu().numpy() for weight in old_weights
    ]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comparators = [
        comp.detach().clone().data.cpu().numpy() for comp in old_comparators
    ]
    new_comparators.append(new_comparator)  # Add it to the list of nodes

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = network.leaf_init_information
    for index, leaf_prob_vec in enumerate(
            network.action_probs):  # Copy over the learned leaf weight
        new_leaf_information[index][-1] = leaf_prob_vec.detach().clone(
        ).data.cpu().numpy()
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]
    new_network = ProLoNet(input_dim=network.input_dim,
                           weights=new_weights,
                           comparators=new_comparators,
                           leaves=new_leaf_information,
                           alpha=network.alpha.item(),
                           is_value=network.is_value,
                           device='cuda' if use_gpu else 'cpu',
                           vectorized=network.vectorized,
                           output_dim=network.output_dim)
    if use_gpu:
        new_network = new_network.cuda()
    return new_network


# def init_lander_nets_aaai21(distribution,
#                      use_gpu=False,
#                      vectorized=False,
#                      randomized=False):
#     dim_in = 8
#     dim_out = 4
#     w0 = np.zeros(dim_in)
#     c0 = np.ones(dim_in) * 2
#     w0[1] = -1
#     c0[1] = -1.1  # < 1.1

#     w1 = np.zeros(dim_in)
#     c1 = np.ones(dim_in) * 2
#     w1[3] = -1
#     c1[3] = 0.2  # < -0.2

#     w2 = np.zeros(dim_in)
#     c2 = np.ones(dim_in) * 2
#     w2[5] = 1
#     c2[5] = 0.1  # > 0.1

#     w3 = np.zeros(dim_in)
#     c3 = np.ones(dim_in) * 2
#     w3[5] = -1
#     c3[5] = -0.1  # < 0.1

#     w4 = np.zeros(dim_in)
#     c4 = np.ones(dim_in) * 2
#     w4[6] = 1
#     w4[7] = 1
#     c4[6] = .9  # both 6 & 7 == 1
#     # c4[7] = .9

#     w5 = np.zeros(dim_in)
#     c5 = np.ones(dim_in) * 2
#     w5[5] = -1
#     c5[5] = 0.1  # < -0.1

#     w6 = np.zeros(dim_in)
#     c6 = np.ones(dim_in) * 2
#     w6[6] = 1
#     w6[7] = 1
#     c6[6] = .9  # both 6 & 7 == 1
#     # c6[7] = .9

#     w7 = np.zeros(dim_in)
#     c7 = np.ones(dim_in) * 2
#     w7[6] = 1
#     w7[7] = 1
#     c7[7] = .9  # both 6 & 7 == 1
#     # c7[7] = .9

#     w8 = np.zeros(dim_in)
#     c8 = np.ones(dim_in) * 2
#     w8[0] = 1
#     c8[0] = 0.2  # > 0

#     w9 = np.zeros(dim_in)
#     c9 = np.ones(dim_in) * 2
#     w9[6] = 1
#     w9[7] = 1
#     c9[6] = .9  # both 6 & 7 == 1
#     # c9[7] = .9

#     w10 = np.zeros(dim_in)
#     c10 = np.ones(dim_in) * 2
#     w10[0] = 1
#     c10[0] = 0.2

#     w11 = np.zeros(dim_in)
#     c11 = np.ones(dim_in) * 2
#     w11[5] = 1
#     c11[5] = -0.1  # > -0.1

#     w12 = np.zeros(dim_in)
#     c12 = np.ones(dim_in) * 2
#     w12[0] = -1
#     c12[0] = 0.2  # < -0.2

#     w13 = np.zeros(dim_in)
#     c13 = np.ones(dim_in) * 2
#     w13[0] = -1
#     c13[0] = 0.2  # < -0.2

#     init_weights = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13]
#     init_comparators = [
#         c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13
#     ]
#     if distribution == 'one_hot':
#         leaf_base_init_val = 0.
#         leaf_target_init_val = 1.
#     elif distribution == 'soft_hot':
#         leaf_base_init_val = 0.1 / dim_out
#         leaf_target_init_val = 0.9
#     else:  # uniform
#         leaf_base_init_val = 1.0 / dim_out
#         leaf_target_init_val = 1.0 / dim_out
#     leaf_base = [leaf_base_init_val] * dim_out

#     l0 = [[0, 1], [3], leaf_base.copy()]
#     l0[-1][3] = leaf_target_init_val

#     l1 = [[0, 4], [1], leaf_base.copy()]
#     l1[-1][0] = leaf_target_init_val

#     l2 = [[6], [0, 2], leaf_base.copy()]
#     l2[-1][0] = leaf_target_init_val

#     l3 = [[], [0, 2, 6], leaf_base.copy()]
#     l3[-1][3] = leaf_target_init_val

#     l4 = [[0, 1, 3, 7], [], leaf_base.copy()]
#     l4[-1][0] = leaf_target_init_val

#     l5 = [[0, 8], [1, 4], leaf_base.copy()]
#     l5[-1][1] = leaf_target_init_val

#     l6 = [[2, 5, 9], [0], leaf_base.copy()]
#     l6[-1][0] = leaf_target_init_val

#     l7 = [[2, 5], [0, 9], leaf_base.copy()]
#     l7[-1][1] = leaf_target_init_val

#     l8 = [[2, 10], [0, 5], leaf_base.copy()]
#     l8[-1][1] = leaf_target_init_val

#     l9 = [[0, 1, 3, 11], [7], leaf_base.copy()]
#     l9[-1][2] = leaf_target_init_val

#     l10 = [[0, 1, 3], [7, 11], leaf_base.copy()]
#     l10[-1][1] = leaf_target_init_val

#     l11 = [[0, 12], [1, 4, 8], leaf_base.copy()]
#     l11[-1][3] = leaf_target_init_val

#     l12 = [[0], [1, 4, 8, 12], leaf_base.copy()]
#     l12[-1][0] = leaf_target_init_val

#     l13 = [[2, 13], [0, 5, 10], leaf_base.copy()]
#     l13[-1][3] = leaf_target_init_val

#     l14 = [[2], [0, 5, 10, 13], leaf_base.copy()]
#     l14[-1][0] = leaf_target_init_val

#     init_leaves = [
#         l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14
#     ]
#     if not vectorized:
#         init_comparators = [comp[comp != 2] for comp in init_comparators]
#     if randomized:
#         init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
#         init_comparators = [
#             np.random.normal(0, 0.1, c.shape) for c in init_comparators
#         ]
#         init_leaves = [[l[0], l[1],
#                         np.random.normal(0, 0.1, dim_out)]
#                        for l in init_leaves]
#         init_weights = None  # None for random init within network
#         init_comparators = None
#         init_leaves = 16  # This is one more than intelligent, but best way to get close with selector
#     action_network = ProLoNet(input_dim=dim_in,
#                               output_dim=dim_out,
#                               weights=init_weights,
#                               comparators=init_comparators,
#                               leaves=init_leaves,
#                               alpha=1.,
#                               is_value=False,
#                               vectorized=vectorized,
#                               device='cuda' if use_gpu else 'cpu')
#     value_network = ProLoNet(input_dim=dim_in,
#                              output_dim=dim_out,
#                              weights=init_weights,
#                              comparators=init_comparators,
#                              leaves=init_leaves,
#                              alpha=1.,
#                              is_value=True,
#                              vectorized=vectorized,
#                              device='cuda' if use_gpu else 'cpu')
#     return action_network, value_network

# Lunar Lander with 31 internal decision nodes
# Default Implementation for lunar lander
def init_lander_nets(distribution,
                     use_gpu=False,
                     vectorized=False,
                     randomized=False):
    dim_in = 8
    dim_out = 4

    # 31 internal decision nodes
    w0 = np.zeros(dim_in)
    c0 = np.ones(dim_in) * 2
    w0[1] = -1
    c0[1] = -1.1  # < 1.1

    w1 = np.zeros(dim_in)
    c1 = np.ones(dim_in) * 2
    w1[3] = -1
    c1[3] = 0.2  # < -0.2

    w2 = np.zeros(dim_in)
    c2 = np.ones(dim_in) * 2
    w2[5] = 1
    c2[5] = 0.1  # > 0.1

    w3 = np.zeros(dim_in)
    c3 = np.ones(dim_in) * 2
    w3[5] = -1
    c3[5] = -0.1  # < 0.1

    w4 = np.zeros(dim_in)
    c4 = np.ones(dim_in) * 2
    w4[6] = 1
    w4[7] = 1
    c4[6] = .9  # both 6 & 7 == 1
    # c4[7] = .9

    w5 = np.zeros(dim_in)
    c5 = np.ones(dim_in) * 2
    w5[5] = -1
    c5[5] = 0.1  # < -0.1

    w6 = np.zeros(dim_in)
    c6 = np.ones(dim_in) * 2
    w6[6] = 1
    w6[7] = 1
    c6[6] = .9  # both 6 & 7 == 1
    # c6[7] = .9

    w7 = np.zeros(dim_in)
    c7 = np.ones(dim_in) * 2
    w7[6] = 1
    w7[7] = 1
    c7[7] = .9  # both 6 & 7 == 1
    # c7[7] = .9

    w8 = np.zeros(dim_in)
    c8 = np.ones(dim_in) * 2
    w8[0] = 1
    c8[0] = 0.2  # > 0

    w9 = np.zeros(dim_in)
    c9 = np.ones(dim_in) * 2
    w9[6] = 1
    w9[7] = 1
    c9[6] = .9  # both 6 & 7 == 1
    # c9[7] = .9

    w10 = np.zeros(dim_in)
    c10 = np.ones(dim_in) * 2
    w10[0] = 1
    c10[0] = 0.2

    w11 = np.zeros(dim_in)
    c11 = np.ones(dim_in) * 2
    w11[5] = 1
    c11[5] = -0.1  # > -0.1

    w12 = np.zeros(dim_in)
    c12 = np.ones(dim_in) * 2
    w12[0] = -1
    c12[0] = 0.2  # < -0.2

    w13 = np.zeros(dim_in)
    c13 = np.ones(dim_in) * 2
    w13[0] = -1
    c13[0] = 0.2  # < -0.2

    # new internal nodes
    w14 = np.zeros(dim_in)
    c14 = np.ones(dim_in) * 2
    w14[0] = 1
    c14[0] = -0.5

    w15 = np.zeros(dim_in)
    c15 = np.ones(dim_in) * 2
    w15[1] = -1
    c15[1] = -1.1

    w16 = np.zeros(dim_in)
    c16 = np.ones(dim_in) * 2
    w16[3] = -1
    c16[1] = 0.2

    w17 = np.zeros(dim_in)
    c17 = np.ones(dim_in) * 2
    w17[5] = 1
    c17[5] = 0.1

    w18 = np.zeros(dim_in)
    c18 = np.ones(dim_in) * 2
    w18[5] = -1
    c18[5] = -0.1

    w19 = np.zeros(dim_in)
    c19 = np.ones(dim_in) * 2
    w19[6] = 1
    w19[7] = 1
    c19[6] = .9

    w20 = np.zeros(dim_in)
    c20 = np.ones(dim_in) * 2
    w20[5] = -1
    c20[5] = 0.1

    w21 = np.zeros(dim_in)
    c21 = np.ones(dim_in) * 2
    w21[6] = 1
    w21[7] = 1
    c21[6] = .9

    w22 = np.zeros(dim_in)
    c22 = np.ones(dim_in) * 2
    w22[6] = 1
    w22[7] = 1
    c22[6] = .9

    w23 = np.zeros(dim_in)
    c23 = np.ones(dim_in) * 2
    w23[0] = 1
    c23[0] = 0.2

    w24 = np.zeros(dim_in)
    c24 = np.ones(dim_in) * 2
    w24[6] = 1
    w24[7] = 1
    c24[6] = .9

    w25 = np.zeros(dim_in)
    c25 = np.ones(dim_in) * 2
    w25[0] = 1
    c25[0] = 0.2

    w26 = np.zeros(dim_in)
    c26 = np.ones(dim_in) * 2
    w26[5] = 1
    c26[5] = -0.1

    w27 = np.zeros(dim_in)
    c27 = np.ones(dim_in) * 2
    w27[0] = -1
    c27[0] = 0.2

    w28 = np.zeros(dim_in)
    c28 = np.ones(dim_in) * 2
    w28[0] = -1
    c28[0] = 0.2

    w29 = np.zeros(dim_in)
    c29 = np.ones(dim_in) * 2
    w29[0] = 1
    c29[0] = -0.5

    w30 = np.zeros(dim_in)
    c30 = np.ones(dim_in) * 2
    w30[5] = 1
    c30[5] = -0.1

    init_weights = [
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15,
        w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29,
        w30
    ]
    init_comparators = [
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
        c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
        c30
    ]
    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

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
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14,
        l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26,
        l27, l28, l29, l30, l31
    ]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None  # None for random init within network
        init_comparators = None
        init_leaves = 33  # This is one more than intelligent, but best way to get close with selector
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1.,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1.,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')

    return action_network, value_network


# Lunar Lander with 63 internal nodes
from agents.dt.arch.lunar import DT_lunar_minimal
# def init_lander_nets_63N(distribution,
#                          use_gpu=False,
#                          vectorized=False,
#                          randomized=False):

#     # total of 63 internal decision nodes
#     dim_in, dim_out, W, B, init_leaves = DT_lunar_minimal()
#     init_weights = [w for w in W]
#     init_comparators = [np.array([-1 * b])
#                         for b in B]  # B is the negative of init_comparators

#     if randomized:
#         init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
#         init_comparators = [
#             np.random.normal(0, 0.1, c.shape) for c in init_comparators
#         ]
#         init_leaves = [[l[0], l[1],
#                         np.random.normal(0, 0.1, dim_out)]
#                        for l in init_leaves]
#         init_weights = None  # None for random init within network
#         init_comparators = None
#         init_leaves = 64  # This is one more than intelligent, but best way to get close with selector
#     action_network = ProLoNet(input_dim=dim_in,
#                               output_dim=dim_out,
#                               weights=init_weights,
#                               comparators=init_comparators,
#                               leaves=init_leaves,
#                               alpha=1.,
#                               is_value=False,
#                               vectorized=vectorized,
#                               device='cuda' if use_gpu else 'cpu')
#     value_network = ProLoNet(input_dim=dim_in,
#                              output_dim=dim_out,
#                              weights=init_weights,
#                              comparators=init_comparators,
#                              leaves=init_leaves,
#                              alpha=1.,
#                              is_value=True,
#                              vectorized=vectorized,
#                              device='cuda' if use_gpu else 'cpu')

#     return action_network, value_network


# SPP: Acrobot implementation with 11 internal nodes
from agents.dt.arch.acrobot import DT_acrobot_minimal
def init_acrobot_nets(distribution,
                     use_gpu=False,
                     vectorized=False,
                     randomized=False):

    # total of 63 internal decision nodes
    dim_in, dim_out, W, B, init_leaves = DT_acrobot_minimal()

    init_weights = [w for w in W]
    init_comparators = [np.array([-1*b]) for b in B]  # B is the negative of init_comparators


    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None  # None for random init within network
        init_comparators = None
        init_leaves = 12  # This is one more than intelligent, but best way to get close with selector
        # I think this is a bug in the original code
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1.,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1.,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')

    return action_network, value_network


def init_adversarial_net(adv_type='cart',
                         distribution_in='one_hot',
                         adv_prob=0.05,
                         use_gpu=False):
    """
    initialize networks intelligently but also wrongly
    with p=adv_prob negate weights
    with p=adv_prob negate comparators
    with p=adv_prob negate leaf probabilities
    :param adv_type: which env is this for: cart, lunar, sc
    :param distribution_in: same as init_cart_nets or init_lander_nets. one_hot, soft_hot, or other...
    :param adv_prob: probability to negate the things above
    :return: actor, critic
    """
    cart_act = None
    if adv_type == 'cart':
        cart_act, cart_value = init_cart_nets(distribution=distribution_in)
    elif adv_type == 'lunar':
        cart_act, cart_value = init_lander_nets(distribution=distribution_in)
    elif adv_type == 'sc':
        cart_act, cart_value = init_sc_nets(distribution_in)
    elif adv_type == 'micro':
        cart_act, cart_value = init_micro_net(distribution=distribution_in)
    if cart_act is None:
        return -1
    old_weights = cart_act.layers  # Get the weights out
    new_weights = [
        weight.detach().clone().data.cpu().numpy() for weight in old_weights
    ]

    old_comparators = cart_act.comparators  # get the comparator values out
    new_comparators = [
        comp.detach().clone().data.cpu().numpy() for comp in old_comparators
    ]

    prob_flip = adv_prob
    old_leaf_information = cart_act.leaf_init_information  # get the leaf init info out
    weight_flips = 0
    comp_flips = 0
    leaf_flips = 0
    quarter_weights = len(new_weights) * (prob_flip * 2)
    quarter_leaves = len(old_leaf_information) * (prob_flip * 2)
    for index in range(len(new_weights)):
        if np.random.random() < prob_flip and weight_flips < quarter_weights:
            new_weights[index] *= -1
            weight_flips += 1
        if np.random.random() < prob_flip and comp_flips < quarter_weights:
            new_comparators[index] *= -1
            comp_flips += 1
    for index in range(len(old_leaf_information)):
        if np.random.random() < prob_flip and leaf_flips < quarter_leaves:
            new_leaf_info = np.array(old_leaf_information[index][-1]) * -1
            old_leaf_information[index][-1] = new_leaf_info.tolist()
            leaf_flips += 1

    new_actor = ProLoNet(input_dim=cart_act.input_dim,
                         weights=new_weights,
                         comparators=new_comparators,
                         leaves=old_leaf_information,
                         alpha=cart_act.alpha.item(),
                         is_value=False,
                         output_dim=cart_act.output_dim)
    new_value = ProLoNet(input_dim=cart_act.input_dim,
                         weights=new_weights,
                         comparators=new_comparators,
                         leaves=old_leaf_information,
                         alpha=cart_act.alpha.item(),
                         is_value=True,
                         output_dim=cart_act.output_dim)

    return new_actor, new_value


def init_sc_nets(dist='one_hot',
                 use_gpu=False,
                 vectorized=False,
                 randomized=False):
    if vectorized:
        return init_sc_nets_vectorized(dist, use_gpu, randomized)
    else:
        return init_sc_nets_nonvec(dist, use_gpu, randomized)


def init_sc_build_hellions_net(dist='one_hot',
                               use_gpu=False,
                               vectorized=False,
                               randomized=False):
    if vectorized:
        print('Vectorized not supported.')
    else:
        return init_sc_build_hellions_net_novec(dist, use_gpu, randomized)


def init_sc_build_hellions_net_novec(dist='one_hot',
                                     use_gpu=False,
                                     randomized=False):
    dim_in = len(EXPANDED_TYPES)
    dim_out = 12

    #food_cap - food_used < 4 go left
    #-food_cap + food_used > -4 go left
    #-food_cap + food_used-4 > go left
    w0 = np.zeros(dim_in)
    w0[EXPANDED_TYPES.FOOD_CAP.value] = -1  # negative food capacity
    w0[EXPANDED_TYPES.FOOD_USED.
       value] = 1  # plus food used = negative food available
    c0 = [-8]  # > -4  (so if positive < 4)

    #scv_count < 18 go left
    #-scv_count > -18 go left
    #-scv_count-18 > 0 go left
    w1 = np.zeros(dim_in)
    w1[EXPANDED_TYPES.SCV.value] = -1
    c1 = [-16]

    #refinery < 1 go left
    w2 = np.zeros(dim_in)
    w2[EXPANDED_TYPES.REFINERY.value] = -1
    c2 = [-1]

    #barracks < 1 go left
    #-barracks > -1 go left
    w3 = np.zeros(dim_in)
    w3[EXPANDED_TYPES.BARRACKS.value] = -1
    c3 = [-0.5]

    #factory < 1 go left
    w4 = np.zeros(dim_in)
    w4[EXPANDED_TYPES.FACTORY.value] = -1
    c4 = [-1]

    #factory not busy go left
    w5 = np.zeros(dim_in)
    w5[EXPANDED_TYPES.FACTORY.value] = 1
    w5[EXPANDED_TYPES.PENDING_HELLION.value] = -1
    c5 = [-0.5]

    #either go back to mining or build more factory with equal probability
    w6 = np.zeros(dim_in)
    c6 = [0]

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
    ]

    if dist == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif dist == 'soft_hot':
        leaf_base_init_val = 0.1 / (max(dim_out - 1, 1))
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    # Supply
    # left from [0]     (Food)
    # right from []
    l0 = [[0], [], leaf_base.copy()]
    l0[-1][1] = leaf_target_init_val

    # SCV
    # left from [1]     (SCV)
    # right from []
    l1 = [[1], [0], leaf_base.copy()]
    l1[-1][8] = leaf_target_init_val

    # Refinery
    # left from [2]     (Refinery)
    # right from [0, 1] (Food, SCV)
    l2 = [[2], [0, 1], leaf_base.copy()]
    l2[-1][2] = leaf_target_init_val

    # Barracks?
    # left from [3]     (Build Barracks)
    # right from [0, 1, 2]  (Food, SCV, Refinery)
    l3 = [[3], [0, 1, 2], leaf_base.copy()]
    l3[-1][3] = leaf_target_init_val

    # Factory
    # left from [4] (Build factory)
    # right from [0, 1, 2, 3]   (all previous)
    l4 = [[4], [0, 1, 2, 3], leaf_base.copy()]
    l4[-1][6] = leaf_target_init_val

    # Hellion
    # left from [5]  (Build Hellion)
    # right from [0, 1, 2, 3, 4]
    l5 = [[5], [0, 1, 2, 3, 4], leaf_base.copy()]
    l5[-1][10] = leaf_target_init_val

    # Back to mining
    # left from [6] (Return to mining)
    # right from [0, 1, 2, 3, 4, 5]
    l6 = [[6], [0, 1, 2, 3, 4, 5], leaf_base.copy()]
    l6[-1][11] = leaf_target_init_val

    # Build another Factory
    # left from []
    # right from [0, 1, 2, 3, 4, 5, 6]
    l7 = [[], [0, 1, 2, 3, 4, 5, 6], leaf_base.copy()]
    l7[-1][6] = leaf_target_init_val

    init_leaves = [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
    ]

    if randomized:
        init_weights = None
        init_comparators = None
        init_leaves = 16

    actor = ProLoNet(input_dim=dim_in,
                     output_dim=dim_out,
                     weights=init_weights,
                     comparators=init_comparators,
                     leaves=init_leaves,
                     alpha=2,
                     vectorized=False,
                     device='cuda' if use_gpu else 'cpu',
                     is_value=False)
    critic = ProLoNet(input_dim=dim_in,
                      output_dim=dim_out,
                      weights=init_weights,
                      comparators=init_comparators,
                      leaves=init_leaves,
                      alpha=1,
                      vectorized=False,
                      device='cuda' if use_gpu else 'cpu',
                      is_value=True)
    return actor, critic


def init_sc_nets_nonvec(dist='one_hot', use_gpu=False, randomized=False):
    dim_in = 194
    dim_out = 44

    w0 = np.zeros(dim_in)
    w0[10] = 1  # zealot
    w0[22] = 1  # voidray
    c0 = [8]  # > 8

    w1 = np.zeros(dim_in)
    w1[45:65] = 1  # Enemy Protoss non-buildings
    w1[82:99] = 1  # Enemy Terran non-buildings
    w1[118:139] = 1  # Enemy Zerg non-buildings
    c1 = [2]  # > 4 enemies, potentially under attack?

    w2 = np.zeros(dim_in)
    w2[4] = 1  # idle workers
    c2 = [0.5]  # > 0.5

    w3 = np.zeros(dim_in)
    w3[65:82] = 1  # Enemy Protoss non-buildings
    w3[99:118] = 1  # Enemy Terran non-buildings
    w3[139:157] = 1  # Enemy Zerg non-buildings
    c3 = [0]  # > 0  # know where some enemy structures are

    w4 = np.zeros(dim_in)
    w4[2] = -1  # negative food capacity
    w4[3] = 1  # plus food used = negative food available
    c4 = [-4]  # > -4  (so if positive < 4)

    w5 = np.zeros(dim_in)
    w5[9] = -1  # probes
    w5[157] = -1
    c5 = [-20]  # > -16 == #probes<16

    w6 = np.zeros(dim_in)
    w6[30] = 1  # ASSIMILATOR
    w6[178] = 1
    c6 = [0.5]  # > 0.5

    w7 = np.zeros(dim_in)
    w7[38] = 1  # STARGATE
    w7[186] = 1
    c7 = [0.5]  # > 1.5

    w8 = np.zeros(dim_in)
    w8[31] = 1  # GATEWAY
    w8[179] = 1
    c8 = [0.5]  # > 0.5

    w9 = np.zeros(dim_in)
    w9[22] = 1  # VOIDRAY
    w9[170] = 1
    c9 = [7]  # > 7

    w10 = np.zeros(dim_in)
    w10[10] = 1  # zealot
    w10[158] = 1
    c10 = [2]  # > 3

    w11 = np.zeros(dim_in)
    w11[34] = 10  # CYBERNETICSCORE
    w11[182] = 10
    c11 = [0.5]  # > 0.5

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]
    if dist == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif dist == 'soft_hot':
        leaf_base_init_val = 0.1 / (max(dim_out - 1, 1))
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0, 1], [], leaf_base.copy()]
    l0[-1][41] = leaf_target_init_val  # Defend
    l0[-1][39] = leaf_target_init_val

    l1 = [[2], [0], leaf_base.copy()]
    l1[-1][40] = leaf_target_init_val  # Mine

    l2 = [[0, 1], [3], leaf_base.copy()]
    l2[-1][39] = leaf_target_init_val  # Attack

    l3 = [[0], [1, 3], leaf_base.copy()]
    l3[-1][42] = leaf_target_init_val  # Scout

    l4 = [[4], [0, 2], leaf_base.copy()]
    l4[-1][1] = leaf_target_init_val  # Pylon

    l5 = [[5], [0, 2, 4], leaf_base.copy()]
    l5[-1][16] = leaf_target_init_val  # Probe

    l6 = [[], [0, 2, 4, 5, 6, 8], leaf_base.copy()]
    l6[-1][3] = leaf_target_init_val  # Gateway

    l7 = [[6, 7, 9], [0, 2, 4, 5], leaf_base.copy()]
    l7[-1][39] = leaf_target_init_val  # Attack

    l8 = [[6, 7], [0, 2, 4, 5, 9], leaf_base.copy()]
    l8[-1][29] = leaf_target_init_val  # Voidray

    l9 = [[6, 10], [0, 2, 4, 5, 7], leaf_base.copy()]
    l9[-1][40] = leaf_target_init_val  # Mine (Get vespene)

    l10 = [[6], [0, 2, 4, 5, 7], leaf_base.copy()]
    l10[-1][10] = leaf_target_init_val  # Stargate

    l11 = [[8], [0, 2, 4, 5, 6, 10], leaf_base.copy()]
    l11[-1][17] = leaf_target_init_val  # Zealot

    l12 = [[8, 10, 11], [0, 2, 4, 5, 6], leaf_base.copy()]
    l12[-1][2] = leaf_target_init_val  # Assimilator

    l13 = [[8, 10], [0, 2, 4, 5, 6, 11], leaf_base.copy()]
    l13[-1][6] = leaf_target_init_val  # Cybernetics Core

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l10,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
    ]
    if randomized:
        init_weights = None
        init_comparators = None
        init_leaves = 16
    actor = ProLoNet(input_dim=dim_in,
                     output_dim=dim_out,
                     weights=init_weights,
                     comparators=init_comparators,
                     leaves=init_leaves,
                     alpha=2,
                     vectorized=False,
                     device='cuda' if use_gpu else 'cpu',
                     is_value=False)
    critic = ProLoNet(input_dim=dim_in,
                      output_dim=dim_out,
                      weights=init_weights,
                      comparators=init_comparators,
                      leaves=init_leaves,
                      alpha=1,
                      vectorized=False,
                      device='cuda' if use_gpu else 'cpu',
                      is_value=True)
    return actor, critic


def init_sc_nets_vectorized(dist='one_hot', use_gpu=False, randomized=False):
    dim_in = 194
    dim_out = 44

    w0 = np.zeros(dim_in)
    c0 = np.ones(dim_in) * 5
    w0[10] = 1  # zealot
    w0[22] = 1  # voidray
    c0[10] = 3  # > 3
    c0[22] = 5

    w1 = np.zeros(dim_in)
    c1 = np.zeros(dim_in) * 5
    w1[45:65] = 1  # Enemy Protoss non-buildings
    w1[82:99] = 1  # Enemy Terran non-buildings
    w1[118:139] = 1  # Enemy Zerg non-buildings
    c1[45:65] = 1  # > 4 enemies, potentially under attack?
    c1[82:99] = 1  # Enemy Terran non-buildings
    c1[118:139] = 1  # Enemy Zerg non-buildings

    w2 = np.zeros(dim_in)
    c2 = np.zeros(dim_in) * 5
    w2[4] = 1  # idle workers
    c2[4] = 0.5  # > 0.5

    w3 = np.zeros(dim_in)
    c3 = np.zeros(dim_in) * 5
    w3[65:82] = 1  # Enemy Protoss non-buildings
    w3[99:118] = 1  # Enemy Terran non-buildings
    w3[139:157] = 1  # Enemy Zerg non-buildings
    c3[65:82] = 0  # Enemy Protoss non-buildings
    c3[99:118] = 0  # Enemy Terran non-buildings
    c3[139:157] = 0  # Enemy Zerg non-buildings

    w4 = np.zeros(dim_in)
    c4 = np.zeros(dim_in) * 5
    w4[2] = -1  # negative food capacity
    w4[3] = 1  # plus food used = negative food available
    c4[2] = -50  # > -50  (so if positive < 50)
    c4[3] = 50

    w5 = np.zeros(dim_in)
    c5 = np.zeros(dim_in) * 5
    w5[9] = -1  # probes
    w5[157] = -1
    c5[9] = -20  # > -16 == #probes<16

    w6 = np.zeros(dim_in)
    c6 = np.zeros(dim_in) * 5
    w6[30] = 1  # ASSIMILATOR
    w6[178] = 1
    c6[30] = 0.5  # > 0.5

    w7 = np.zeros(dim_in)
    c7 = np.ones(dim_in) * 5
    w7[38] = 1  # STARGATE
    w7[186] = 1
    c7[38] = 0.5  # > 1.5

    w8 = np.zeros(dim_in)
    c8 = np.ones(dim_in) * 5
    w8[31] = 1  # GATEWAY
    w8[179] = 1
    c8[31] = 0.5  # > 0.5

    w9 = np.zeros(dim_in)
    c9 = np.ones(dim_in) * 5
    w9[22] = 1  # VOIDRAY
    w9[170] = 1
    c9[22] = 7  # > 7

    w10 = np.zeros(dim_in)
    c10 = np.ones(dim_in) * 5
    w10[10] = 1  # zealot
    w10[158] = 1
    c10[10] = 2  # > 3

    w11 = np.zeros(dim_in)
    c11 = np.ones(dim_in) * 5
    w11[34] = 10  # CYBERNETICSCORE
    w11[182] = 10
    c11[34] = 0.5  # > 0.5

    init_weights = [
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7,
        w8,
        w9,
        w10,
        w11,
    ]
    init_comparators = [
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ]
    if dist == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif dist == 'soft_hot':
        leaf_base_init_val = 0.1 / (max(dim_out - 1, 1))
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0, 1], [], leaf_base.copy()]
    l0[-1][41] = leaf_target_init_val  # Defend
    l0[-1][39] = leaf_target_init_val

    l1 = [[2], [0], leaf_base.copy()]
    l1[-1][40] = leaf_target_init_val  # Mine

    l2 = [[0, 1], [3], leaf_base.copy()]
    l2[-1][39] = leaf_target_init_val  # Attack

    l3 = [[0], [1, 3], leaf_base.copy()]
    l3[-1][42] = leaf_target_init_val  # Scout

    l4 = [[4], [0, 2], leaf_base.copy()]
    l4[-1][1] = leaf_target_init_val  # Pylon

    l5 = [[5], [0, 2, 4], leaf_base.copy()]
    l5[-1][16] = leaf_target_init_val  # Probe

    l6 = [[], [0, 2, 4, 5, 6, 8], leaf_base.copy()]
    l6[-1][3] = leaf_target_init_val  # Gateway

    l7 = [[6, 7, 9], [0, 2, 4, 5], leaf_base.copy()]
    l7[-1][39] = leaf_target_init_val  # Attack

    l8 = [[6, 7], [0, 2, 4, 5, 9], leaf_base.copy()]
    l8[-1][29] = leaf_target_init_val  # Voidray

    l9 = [[6, 10], [0, 2, 4, 5, 7], leaf_base.copy()]
    l9[-1][40] = leaf_target_init_val  # Mine (Get vespene)

    l10 = [[6], [0, 2, 4, 5, 7], leaf_base.copy()]
    l10[-1][10] = leaf_target_init_val  # Stargate

    l11 = [[8], [0, 2, 4, 5, 6, 10], leaf_base.copy()]
    l11[-1][17] = leaf_target_init_val  # Zealot

    l12 = [[8, 10, 11], [0, 2, 4, 5, 6], leaf_base.copy()]
    l12[-1][2] = leaf_target_init_val  # Assimilator

    l13 = [[8, 10], [0, 2, 4, 5, 6, 11], leaf_base.copy()]
    l13[-1][6] = leaf_target_init_val  # Cybernetics Core

    init_leaves = [
        l1,
        l2,
        l3,
        l4,
        l5,
        l10,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
    ]
    init_selectors = []
    for weight_vec in init_weights:
        sel = np.zeros(dim_in)  # Initialize selector to 0
        sel[weight_vec != 0] = 1  # Set to 1 where there's a weight
        init_selectors.append(sel)
    if randomized:
        init_weights = None
        init_comparators = None
        init_selectors = None
        init_leaves = 16

    actor = ProLoNet(input_dim=dim_in,
                     output_dim=dim_out,
                     weights=init_weights,
                     comparators=init_comparators,
                     selectors=init_selectors,
                     leaves=init_leaves,
                     alpha=1,
                     vectorized=True,
                     device='cuda' if use_gpu else 'cpu',
                     is_value=False)
    critic = ProLoNet(input_dim=dim_in,
                      output_dim=dim_out,
                      weights=init_weights,
                      comparators=init_comparators,
                      selectors=init_selectors,
                      leaves=init_leaves,
                      alpha=1,
                      vectorized=True,
                      device='cuda' if use_gpu else 'cpu',
                      is_value=True)
    return actor, critic


def save_prolonet(fn, model):
    checkpoint = dict()
    mdl_data = dict()
    mdl_data['weights'] = model.layers
    mdl_data['comparators'] = model.comparators
    mdl_data['leaf_init_information'] = model.leaf_init_information
    mdl_data['action_probs'] = model.action_probs
    mdl_data['alpha'] = model.alpha
    mdl_data['input_dim'] = model.input_dim
    mdl_data['is_value'] = model.is_value
    checkpoint['model_data'] = mdl_data
    torch.save(checkpoint, fn)


def load_prolonet(fn, use_gpu):
    model_checkpoint = torch.load(fn, map_location='cpu')
    model_data = model_checkpoint['model_data']
    init_weights = [
        weight.detach().clone().data.cpu().numpy()
        for weight in model_data['weights']
    ]
    init_comparators = [
        comp.detach().clone().data.cpu().numpy()
        for comp in model_data['comparators']
    ]
    new_model = ProLoNet(input_dim=model_data['input_dim'],
                         weights=init_weights,
                         comparators=init_comparators,
                         leaves=model_data['leaf_init_information'],
                         alpha=model_data['alpha'].item(),
                         is_value=model_data['is_value'],
                         device='cuda' if use_gpu else 'cpu')
    new_model.action_probs = model_data['action_probs']
    return new_model


def init_shallow_cart_nets(distribution,
                           use_gpu=False,
                           vectorized=False,
                           randomized=False):
    dim_in = 4
    dim_out = 2
    w0 = np.zeros(dim_in)
    c0 = np.ones(dim_in) * 2
    w0[2] = -1  # pole angle
    c0[2] = 0  # < 0
    init_weights = [w0]
    init_comparators = [c0]
    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l0 = [[0], [], leaf_base.copy()]
    l0[-1][1] = leaf_target_init_val  # Right

    l1 = [[], [0], leaf_base.copy()]
    l1[-1][0] = leaf_target_init_val  # Right

    init_leaves = [l0, l1]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None  # None for random init within network
        init_comparators = None
        init_leaves = 2  # This is one more than intelligent, but best way to get close with selector
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu',
                              alpha=1,
                              is_value=False)
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu',
                             alpha=1,
                             is_value=True)
    return action_network, value_network


# def init_micro_net_6N(distribution='one_hot',
#                    use_gpu=False,
#                    vectorized=False,
#                    randomized=False):
#     dim_in = 37
#     dim_out = 10

#     w4 = np.zeros(dim_in)
#     c4 = np.ones(dim_in) * 2
#     w4[14] = 1  # zergling health
#     c4[14] = 0

#     w5 = np.zeros(dim_in)
#     c5 = np.ones(dim_in) * 2
#     w5[1] = 1  # y position
#     c5[1] = 30  # > 30

#     w6 = np.zeros(dim_in)
#     c6 = np.ones(dim_in) * 2
#     w6[0] = -1  # y position
#     c6[0] = -20  # < 20

#     w7 = np.zeros(dim_in)
#     c7 = np.ones(dim_in) * 2
#     w7[1] = 1  # y position
#     c7[1] = 18  # < 20

#     w8 = np.zeros(dim_in)
#     c8 = np.ones(dim_in) * 2
#     w8[0] = 1  # y position
#     c8[0] = 40  # < 20

#     w9 = np.zeros(dim_in)
#     c9 = np.ones(dim_in) * 2
#     w9[0] = -1  # y position
#     c9[0] = -40  # < 20

#     init_weights = [w4, w5, w6, w7, w8, w9]
#     init_comparators = [c4, c5, c6, c7, c8, c9]

#     if distribution == 'one_hot':
#         leaf_base_init_val = 0.
#         leaf_target_init_val = 1.
#     elif distribution == 'soft_hot':
#         leaf_base_init_val = 0.1 / dim_out
#         leaf_target_init_val = 0.9
#     else:  # uniform
#         leaf_base_init_val = 1.0 / dim_out
#         leaf_target_init_val = 1.0 / dim_out
#     leaf_base = [leaf_base_init_val] * dim_out

#     l4 = [[0], [], leaf_base.copy()]
#     l4[-1][4] = leaf_target_init_val  # Attack

#     l5 = [[1, 2], [0], leaf_base.copy()]
#     l5[-1][2] = leaf_target_init_val  # S

#     l6 = [[3], [0, 1], leaf_base.copy()]
#     l6[-1][2] = leaf_target_init_val  # S

#     l7 = [[1, 4], [0, 2], leaf_base.copy()]
#     l7[-1][0] = leaf_target_init_val  # N

#     l8 = [[1], [0, 2, 4], leaf_base.copy()]
#     l8[-1][3] = leaf_target_init_val  # N

#     l9 = [[5], [0, 1, 3], leaf_base.copy()]
#     l9[-1][1] = leaf_target_init_val  # E

#     l10 = [[], [0, 1, 3, 5], leaf_base.copy()]
#     l10[-1][0] = leaf_target_init_val  # N

#     init_leaves = [l4, l5, l6, l7, l8, l9, l10]
#     if not vectorized:
#         init_comparators = [comp[comp != 2] for comp in init_comparators]
#     if randomized:
#         init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
#         init_comparators = [
#             np.random.normal(0, 0.1, c.shape) for c in init_comparators
#         ]
#         init_leaves = [[l[0], l[1],
#                         np.random.normal(0, 0.1, dim_out)]
#                        for l in init_leaves]
#         init_weights = None
#         init_comparators = None
#         init_leaves = 8
#     action_network = ProLoNet(input_dim=dim_in,
#                               output_dim=dim_out,
#                               weights=init_weights,
#                               comparators=init_comparators,
#                               leaves=init_leaves,
#                               alpha=1,
#                               is_value=False,
#                               vectorized=vectorized,
#                               device='cuda' if use_gpu else 'cpu')
#     value_network = ProLoNet(input_dim=dim_in,
#                              output_dim=dim_out,
#                              weights=init_weights,
#                              comparators=init_comparators,
#                              leaves=init_leaves,
#                              alpha=1,
#                              is_value=True,
#                              vectorized=vectorized,
#                              device='cuda' if use_gpu else 'cpu')
#     return action_network, value_network

# Find and Defeat with 8 Nodes
def init_micro_net_8N(distribution='one_hot',
                   use_gpu=False,
                   vectorized=False,
                   randomized=False):
    dim_in = 37
    dim_out = 10

    w4 = np.zeros(dim_in)
    c4 = np.ones(dim_in) * 2
    w4[14] = 1  # zergling health
    c4[14] = 0

    w5 = np.zeros(dim_in)
    c5 = np.ones(dim_in) * 2
    w5[1] = 1  # y position
    c5[1] = -30  # > 30

    w6 = np.zeros(dim_in)
    c6 = np.ones(dim_in) * 2
    w6[0] = -1  # y position
    c6[0] = 20  # < 20

    w7 = np.zeros(dim_in)
    c7 = np.ones(dim_in) * 2
    w7[1] = 1  # y position
    c7[1] = -18  # < 20

    w8 = np.zeros(dim_in)
    c8 = np.ones(dim_in) * 2
    w8[0] = 1  # y position
    c8[0] = -40  # < 20

    w9 = np.zeros(dim_in)
    c9 = np.ones(dim_in) * 2
    w9[0] = -1  # y position
    c9[0] = 40  # < 20

    w10 = np.zeros(dim_in)
    c10 = np.ones(dim_in) * 2
    w10[1] = 1  # y position
    c10[1] = -20  # < 20

    w11 = np.zeros(dim_in)
    c11 = np.ones(dim_in) * 2
    w11[1] = -1  # y position
    c11[1] = 20  # < 20

    init_weights = [w4, w5, w6, w7, w8, w9, w10, w11]
    init_comparators = [c4, c5, c6, c7, c8, c9, c10, c11]

    if distribution == 'one_hot':
        leaf_base_init_val = 0.
        leaf_target_init_val = 1.
    elif distribution == 'soft_hot':
        leaf_base_init_val = 0.1 / dim_out
        leaf_target_init_val = 0.9
    else:  # uniform
        leaf_base_init_val = 1.0 / dim_out
        leaf_target_init_val = 1.0 / dim_out
    leaf_base = [leaf_base_init_val] * dim_out

    l4 = [[0], [], leaf_base.copy()]
    l4[-1][4] = leaf_target_init_val  # Attack

    l5 = [[1, 2], [0], leaf_base.copy()]
    l5[-1][2] = leaf_target_init_val  # S

    l6 = [[3], [0, 1], leaf_base.copy()]
    l6[-1][2] = leaf_target_init_val  # S

    l7 = [[1, 4], [0, 2], leaf_base.copy()]
    l7[-1][0] = leaf_target_init_val  # N

    l8 = [[1], [0, 2, 4], leaf_base.copy()]
    l8[-1][3] = leaf_target_init_val  # N

    l9 = [[5, 6], [0, 1, 3], leaf_base.copy()]
    l9[-1][1] = leaf_target_init_val  # E

    l10 = [[5], [0, 1, 3, 6], leaf_base.copy()]
    l10[-1][3] = leaf_target_init_val  # N

    l11 = [[7], [0, 1, 3, 5], leaf_base.copy()]
    l11[-1][0] = leaf_target_init_val  # N

    l12 = [[], [0, 1, 3, 5, 7], leaf_base.copy()]
    l12[-1][2] = leaf_target_init_val  # S

    init_leaves = [l4, l5, l6, l7, l8, l9, l10, l11, l12]
    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None
        init_comparators = None
        # init_leaves = 10 # SPP: Why?
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')
    return action_network, value_network


from agents.dt.arch.zerlings import DT_zerlings_v1 # carefully import DT with 10 N
def init_micro_net_10N(distribution='one_hot',
                   use_gpu=False,
                   vectorized=False,
                   randomized=False):

    dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves = DT_zerlings_v1(num_nodes='10n')

    init_weights = [w for w in W]
    init_comparators = [np.array([-1 * b])
                        for b in B]  # B is the negative of init_comparators

    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None
        init_comparators = None
        init_leaves = 11
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')
    return action_network, value_network


# Minimal tree: Reused from the definition of the DT in dt.arch.zerlings
from agents.dt.arch.zerlings import DT_zerlings_minimal
def init_micro_net(distribution='one_hot',
                       use_gpu=False,
                       vectorized=False,
                       randomized=False):

    dim_in, dim_out, W, B, init_leaves = DT_zerlings_minimal()

    init_weights = [w for w in W]
    init_comparators = [np.array([-1 * b])
                        for b in B]  # B is the negative of init_comparators

    if not vectorized:
        init_comparators = [comp[comp != 2] for comp in init_comparators]
    if randomized:
        init_weights = [np.random.normal(0, 0.1, dim_in) for w in init_weights]
        init_comparators = [
            np.random.normal(0, 0.1, c.shape) for c in init_comparators
        ]
        init_leaves = [[l[0], l[1],
                        np.random.normal(0, 0.1, dim_out)]
                       for l in init_leaves]
        init_weights = None
        init_comparators = None
        init_leaves = 32
    action_network = ProLoNet(input_dim=dim_in,
                              output_dim=dim_out,
                              weights=init_weights,
                              comparators=init_comparators,
                              leaves=init_leaves,
                              alpha=1,
                              is_value=False,
                              vectorized=vectorized,
                              device='cuda' if use_gpu else 'cpu')
    value_network = ProLoNet(input_dim=dim_in,
                             output_dim=dim_out,
                             weights=init_weights,
                             comparators=init_comparators,
                             leaves=init_leaves,
                             alpha=1,
                             is_value=True,
                             vectorized=vectorized,
                             device='cuda' if use_gpu else 'cpu')
    return action_network, value_network


def init_fire_nets(distribution='one_hot',
                   use_gpu=False,
                   vectorized=False,
                   randomized=False,
                   bot_name='usermadeprolo'):
    if randomized:
        dim_in = 6
        dim_out = 5
        init_weights = None
        init_comparators = None
        init_leaves = 8
        action_network = ProLoNet(input_dim=dim_in,
                                  output_dim=dim_out,
                                  weights=init_weights,
                                  comparators=init_comparators,
                                  leaves=init_leaves,
                                  alpha=1,
                                  is_value=False,
                                  vectorized=vectorized,
                                  device='cuda' if use_gpu else 'cpu')
        value_network = ProLoNet(input_dim=dim_in,
                                 output_dim=dim_out,
                                 weights=init_weights,
                                 comparators=init_comparators,
                                 leaves=init_leaves,
                                 alpha=1,
                                 is_value=True,
                                 vectorized=vectorized,
                                 device='cuda' if use_gpu else 'cpu')
    else:
        act_fn = f'../study_models/finished_models/{bot_name}FireSim_actor_.pth.tar'
        val_fn = f'../study_models/finished_models/{bot_name}FireSim_critic_.pth.tar'
        action_network = load_prolonet(act_fn)
        value_network = load_prolonet(val_fn)
    return action_network, value_network
