"""User defined initial Decision Tree for find and defeat Zerlings."""

import numpy as np

###################
# 0: N
# 1: E
# 2: S
# 3: W
# 4: Attack N
# 5: Attack E
# 6: Attack S
# 7: Attack W
# 8: No move
# 9: No attack
###################
def DT_zerlings_v1(num_nodes='8nr'):
    """Single implimentation which calls the other implimentations according to specifies number of nodes"""
    assert num_nodes in ['8nr', '8nl', '6n', '10n'], 'Invalid number of nodes'
    if num_nodes == '8nr':
        return DT_zerlings_v1_8nr()
    elif num_nodes == '8nl':
        return DT_zerlings_v1_8nl()
    elif num_nodes == '6n':
        return DT_zerlings_v1_6n()
    elif num_nodes == '10n':
        return DT_zerlings_v1_10n()
    else:
        raise NotImplementedError

def DT_zerlings_v1_6n():
    """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT.
    Original DT with 6 decision nodes.
    """
    dim_in = 37
    dim_out = 10

    num_decision_nodes = 6
    num_action_nodes = 63 # 7 * 9
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 14] = 1
    B[0] = 0

    # node 1
    W[1, 1] = 1
    B[1] = 30

    # node 2:
    W[2, 0] = -1
    B[2] = -20

    # node 3:
    W[3, 1] = 1
    B[3] = 18

    # node 4:
    W[4, 0] = 1
    B[4] = 40

    # node 5:
    W[5, 0] = -1
    B[5] = -40


    # ========== Action Nodes ==========
    B[6] = 1
    B[7] = 1

    B[15] = 1
    B[16] = 1
    B[19] = 1

    B[24] = 1
    B[25] = -1
    B[28] = -1

    B[33] = -1
    B[36] = 1
    B[41] = 1

    B[34] = -1
    B[38] = 1
    B[45] = 1

    B[51] = -1
    B[54] = 1
    B[59] = -1

    B[52] = -1
    B[56] = 1
    B[63] = -1


    # ========== Leaves ==========
    leaf_base = [0] * dim_out

    l0 = [[0, 6, 7, 9], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 6, 7], [9], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 6, 10], [7], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 6], [7, 10], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 8, 11], [6], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 8], [6, 11], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 12, 13], [6, 8], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 12], [6, 8, 13], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 14], [6, 8, 12], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0], [6, 8, 12, 14], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[1, 2, 15, 16, 18], [0], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[1, 2, 15, 16], [0, 18], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[1, 2, 15, 19], [0, 16], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[1, 2, 15], [0, 16, 19], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[1, 2, 17, 20], [0, 15], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[1, 2, 17], [0, 15, 20], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[1, 2, 21, 22], [0, 15, 17], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[1, 2, 21], [0, 15, 17, 22], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[1, 2, 23], [0, 15, 17, 21], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[1, 2], [0, 15, 17, 21, 23], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[3, 24, 25, 27], [0, 1,], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[3, 24, 25], [0, 1, 27], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[3, 24, 28], [0, 1, 25], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[3, 24], [0, 1, 25, 28], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[3, 26, 29], [0, 1, 24], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[3, 26], [0, 1, 24, 29], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[3, 30, 31], [0, 1, 24, 26], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[3, 30], [0, 1, 24, 26, 31], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[3, 32], [0, 1, 24, 26, 30], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[3], [0, 1, 24, 26, 30, 32], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[1, 4, 33, 35, 39], [0, 2], leaf_base.copy()]
    l30[-1][0] = 1
    l31 = [[1, 4, 33, 35], [0, 2, 39], leaf_base.copy()]
    l31[-1][1] = 1
    l32 = [[1, 4, 33, 40], [0, 2, 35], leaf_base.copy()]
    l32[-1][2] = 1
    l33 = [[1, 4, 33], [0, 2, 35, 40], leaf_base.copy()]
    l33[-1][3] = 1
    l34 = [[1, 4, 36, 41], [0, 2, 33], leaf_base.copy()]
    l34[-1][4] = 1
    l35 = [[1, 4, 36], [0, 2, 33, 41], leaf_base.copy()]
    l35[-1][5] = 1
    l36 = [[1, 4, 42, 47], [0, 2, 33, 36], leaf_base.copy()]
    l36[-1][6] = 1
    l37 = [[1, 4, 42], [0, 2, 33, 36, 47], leaf_base.copy()]
    l37[-1][7] = 1
    l38 = [[1, 4, 48], [0, 2, 33, 36, 42], leaf_base.copy()]
    l38[-1][8] = 1
    l39 = [[1, 4], [0, 2, 33, 36, 42, 48], leaf_base.copy()]
    l39[-1][9] = 1

    l40 = [[1, 34, 37, 43], [0, 2, 4], leaf_base.copy()]
    l40[-1][0] = 1
    l41 = [[1, 34, 37], [0, 2, 4, 43], leaf_base.copy()]
    l41[-1][1] = 1
    l42 = [[1, 34, 44], [0, 2, 4, 37], leaf_base.copy()]
    l42[-1][2] = 1
    l43 = [[1, 34], [0, 2, 4, 37, 44], leaf_base.copy()]
    l43[-1][3] = 1
    l44 = [[1, 38, 45], [0, 2, 4, 34], leaf_base.copy()]
    l44[-1][4] = 1
    l45 = [[1, 38], [0, 2, 4, 34, 45], leaf_base.copy()]
    l45[-1][5] = 1
    l46 = [[1, 46, 49], [0, 2, 4, 34, 38], leaf_base.copy()]
    l46[-1][6] = 1
    l47 = [[1, 46], [0, 2, 4, 34, 38, 49], leaf_base.copy()]
    l47[-1][7] = 1
    l48 = [[1, 50], [0, 2, 4, 34, 38, 46], leaf_base.copy()]
    l48[-1][8] = 1
    l49 = [[1], [0, 2, 4, 34, 38, 46, 50], leaf_base.copy()]
    l49[-1][9] = 1

    l50 = [[5, 51, 53, 57], [0, 1, 3], leaf_base.copy()]
    l50[-1][0] = 1
    l51 = [[5, 51, 53], [0, 1, 3, 57], leaf_base.copy()]
    l51[-1][1] = 1
    l52 = [[5, 51, 58], [0, 1, 3, 53], leaf_base.copy()]
    l52[-1][2] = 1
    l53 = [[5, 51], [0, 1, 3, 53, 58], leaf_base.copy()]
    l53[-1][3] = 1
    l54 = [[5, 54, 59], [0, 1, 3, 51], leaf_base.copy()]
    l54[-1][4] = 1
    l55 = [[5, 54], [0, 1, 3, 51, 59], leaf_base.copy()]
    l55[-1][5] = 1
    l56 = [[5, 60, 65], [0, 1, 3, 51, 54], leaf_base.copy()]
    l56[-1][6] = 1
    l57 = [[5, 60], [0, 1, 3, 51, 54, 65], leaf_base.copy()]
    l57[-1][7] = 1
    l58 = [[5, 66], [0, 1, 3, 51, 54, 60], leaf_base.copy()]
    l58[-1][8] = 1
    l59 = [[5], [0, 1, 3, 51, 54, 60, 66], leaf_base.copy()]
    l59[-1][9] = 1

    l60 = [[52, 55, 61], [0, 1, 3, 5], leaf_base.copy()]
    l60[-1][0] = 1
    l61 = [[52, 55], [0, 1, 3, 5, 61], leaf_base.copy()]
    l61[-1][1] = 1
    l62 = [[52, 62], [0, 1, 3, 5, 55], leaf_base.copy()]
    l62[-1][2] = 1
    l63 = [[52], [0, 1, 3, 5, 55, 62], leaf_base.copy()]
    l63[-1][3] = 1
    l64 = [[56, 63], [0, 1, 3, 5, 52], leaf_base.copy()]
    l64[-1][4] = 1
    l65 = [[56], [0, 1, 3, 5, 52, 63], leaf_base.copy()]
    l65[-1][5] = 1
    l66 = [[64, 67], [0, 1, 3, 5, 52, 56], leaf_base.copy()]
    l66[-1][6] = 1
    l67 = [[64], [0, 1, 3, 5, 52, 56, 67], leaf_base.copy()]
    l67[-1][7] = 1
    l68 = [[68], [0, 1, 3, 5, 52, 56, 64], leaf_base.copy()]
    l68[-1][8] = 1
    l69 = [[], [0, 1, 3, 5, 52, 56, 64, 68], leaf_base.copy()]
    l69[-1][9] = 1

    # 70 leaves



    init_leaves = [l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10,
                   l11, l12, l13, l14, l15, l16, l17, l18, l19, l20,
                   l21, l22, l23, l24, l25, l26, l27, l28, l29, l30,
                   l31, l32, l33, l34, l35, l36, l37, l38, l39, l40,
                   l41, l42, l43, l44, l45, l46, l47, l48, l49, l50,
                   l51, l52, l53, l54, l55, l56, l57, l58, l59, l60,
                   l61, l62, l63, l64, l65, l66, l67, l68, l69]



    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves

# 8 nodes, 2 node at right -- WORKS GOOD
def DT_zerlings_v1_8nr():
    """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT.
    8 nodes. 2 nodes are appended to right most node
    """

    dim_in = 37
    dim_out = 10

    num_decision_nodes = 8
    num_action_nodes = 81  # 9 * 9
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 14] = 1
    B[0] = 0

    # node 1
    W[1, 1] = 1
    B[1] = 30

    # node 2:
    W[2, 0] = -1
    B[2] = -20

    # node 3:
    W[3, 1] = 1
    B[3] = 18

    # node 4:
    W[4, 0] = 1
    B[4] = 40

    # node 5:
    W[5, 0] = -1
    B[5] = -40

    # node 6:
    W[6, 1] = 1
    B[6] = 20

    #node 7:
    W[7, 1] = -1
    B[7] = -20

    # ========== Action Nodes ==========


    B[8] = -10
    B[10] = 10
    B[13] = 10
    B[15] = 10

    B[17] = 10
    B[18] = -10
    B[21] = 10

    # new addition after training
    B[26] = 10
    B[27] = -10
    B[30] = 10

    B[35] = 10
    B[36] = 10
    B[38] = 10


    B[44] = 10
    B[45] = -10
    B[48] = -10 # used to be 10 (new addition to the training)

    B[53] = 10
    B[54] = 10
    B[56] = -10

    B[62] = 10
    B[63] = -10
    B[66] = -10

    B[71] = 10
    B[72] = 10
    B[74] = 10

    B[80] = 10
    B[81] = -10
    B[84] = 10

    # ========== Leaves ==========
    leaf_base = [0] * dim_out

    l0 = [[0, 8, 9, 11], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 8, 9], [11], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 8, 12], [9], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 8], [9, 12], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 10, 13, 15], [8], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 10, 13], [8, 15], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 10, 16], [8, 13 ], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 10], [8, 13, 16], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 14], [8, 10], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0], [8, 10, 14], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[1, 2, 17, 18, 20], [0], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[1, 2, 17, 18], [0, 20], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[1, 2, 17, 21], [0, 18], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[1, 2, 17], [0, 18, 21], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[1, 2, 19, 22, 24], [0, 17], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[1, 2, 19, 22], [0, 17, 24], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[1, 2, 19, 25], [0, 17, 22], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[1, 2, 19], [0, 17, 22, 25], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[1, 2, 23], [0, 17, 19], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[1, 2], [0, 17, 19, 23], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[3, 26, 27, 29], [0, 1], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[3, 26, 27], [0, 1, 29], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[3, 26, 30], [0, 1, 27], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[3, 26], [0, 1, 27, 30], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[3, 28, 31, 33], [0, 1, 26], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[3, 28, 31], [0, 1, 26, 33], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[3, 28, 34], [0, 1, 26, 31], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[3, 28], [0, 1, 26, 31, 34], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[3, 32], [0, 1, 26, 28], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[3], [0, 1, 26, 28, 32], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[1, 4, 35, 36, 38], [0, 2], leaf_base.copy()]
    l30[-1][0] = 1
    l31 = [[1, 4, 35, 36], [0, 2, 38], leaf_base.copy()]
    l31[-1][1] = 1
    l32 = [[1, 4, 35, 39], [0, 2, 36], leaf_base.copy()]
    l32[-1][2] = 1
    l33 = [[1, 4, 35], [0, 2, 36, 39], leaf_base.copy()]
    l33[-1][3] = 1
    l34 = [[1, 4, 37, 40, 42], [0, 2, 35], leaf_base.copy()]
    l34[-1][4] = 1
    l35 = [[1, 4, 37, 40], [0, 2, 35, 42], leaf_base.copy()]
    l35[-1][5] = 1
    l36 = [[1, 4, 37, 43], [0, 2, 35, 40], leaf_base.copy()]
    l36[-1][6] = 1
    l37 = [[1, 4, 37], [0, 2, 35, 40, 43], leaf_base.copy()]
    l37[-1][7] = 1
    l38 = [[1, 4, 41], [0, 2, 35, 37], leaf_base.copy()]
    l38[-1][8] = 1
    l39 = [[1, 4], [0, 2, 35, 37, 41], leaf_base.copy()]
    l39[-1][9] = 1

    l40 = [[1, 44, 45, 47], [0, 2, 4], leaf_base.copy()]
    l40[-1][0] = 1
    l41 = [[1, 44, 45], [0, 2, 4, 47], leaf_base.copy()]
    l41[-1][1] = 1
    l42 = [[1, 44, 48], [0, 2, 4, 45], leaf_base.copy()]
    l42[-1][2] = 1
    l43 = [[1, 44], [0, 2, 4, 45, 48], leaf_base.copy()]
    l43[-1][3] = 1
    l44 = [[1, 46, 49, 51], [0, 2, 4, 44], leaf_base.copy()]
    l44[-1][4] = 1
    l45 = [[1, 46, 49], [0, 2, 4, 44, 51], leaf_base.copy()]
    l45[-1][5] = 1
    l46 = [[1, 46, 52], [0, 2, 4, 44, 49], leaf_base.copy()]
    l46[-1][6] = 1
    l47 = [[1, 46], [0, 2, 4, 44, 49, 52], leaf_base.copy()]
    l47[-1][7] = 1
    l48 = [[1, 50], [0, 2, 4, 44, 46], leaf_base.copy()]
    l48[-1][8] = 1
    l49 = [[1], [0, 2, 4, 44, 46, 50], leaf_base.copy()]
    l49[-1][9] = 1

    # fix here Node 6
    l50 = [[5, 6, 53, 54, 56], [0, 1, 3], leaf_base.copy()]
    l50[-1][0] = 1
    l51 = [[5, 6, 53, 54], [0, 1, 3, 56], leaf_base.copy()]
    l51[-1][1] = 1
    l52 = [[5, 6, 53, 57], [0, 1, 3, 54], leaf_base.copy()]
    l52[-1][2] = 1
    l53 = [[5, 6, 53], [0, 1, 3, 54, 57], leaf_base.copy()]
    l53[-1][3] = 1
    l54 = [[5, 6, 55, 58, 60], [0, 1, 3, 53], leaf_base.copy()]
    l54[-1][4] = 1
    l55 = [[5, 6, 55, 58], [0, 1, 3, 53, 60], leaf_base.copy()]
    l55[-1][5] = 1
    l56 = [[5, 6, 55, 61], [0, 1, 3, 53, 58], leaf_base.copy()]
    l56[-1][6] = 1
    l57 = [[5, 6, 55], [0, 1, 3, 53, 58, 61], leaf_base.copy()]
    l57[-1][7] = 1
    l58 = [[5, 6, 59], [0, 1, 3, 53, 55], leaf_base.copy()]
    l58[-1][8] = 1
    l59 = [[5, 6], [0, 1, 3, 53, 55, 59], leaf_base.copy()]
    l59[-1][9] = 1

    l60 = [[5, 62, 63, 65], [0, 1, 3, 6], leaf_base.copy()]
    l60[-1][0] = 1
    l61 = [[5, 62, 63], [0, 1, 3, 6, 65], leaf_base.copy()]
    l61[-1][1] = 1
    l62 = [[5, 62, 66], [0, 1, 3, 6, 63], leaf_base.copy()]
    l62[-1][2] = 1
    l63 = [[5, 62], [0, 1, 3, 6, 63, 66], leaf_base.copy()]
    l63[-1][3] = 1
    l64 = [[5, 64, 67, 69], [0, 1, 3, 6, 62], leaf_base.copy()]
    l64[-1][4] = 1
    l65 = [[5, 64, 67], [0, 1, 3, 6, 62, 69], leaf_base.copy()]
    l65[-1][5] = 1
    l66 = [[5, 64, 70], [0, 1, 3, 6, 62, 67], leaf_base.copy()]
    l66[-1][6] = 1
    l67 = [[5, 64], [0, 1, 3, 6, 62, 67, 70], leaf_base.copy()]
    l67[-1][7] = 1
    l68 = [[5, 68], [0, 1, 3, 6, 62, 64], leaf_base.copy()]
    l68[-1][8] = 1
    l69 = [[5], [0, 1, 3, 6, 62, 64, 68], leaf_base.copy()]
    l69[-1][9] = 1

    l70 = [[7, 71, 72, 74], [0, 1, 3, 5], leaf_base.copy()]
    l70[-1][0] = 1
    l71 = [[7, 71, 72], [0, 1, 3, 5, 74], leaf_base.copy()]
    l71[-1][1] = 1
    l72 = [[7, 71, 75], [0, 1, 3, 5, 72], leaf_base.copy()]
    l72[-1][2] = 1
    l73 = [[7, 71], [0, 1, 3, 5, 72, 75], leaf_base.copy()]
    l73[-1][3] = 1
    l74 = [[7, 73, 76, 78], [0, 1, 3, 5, 71], leaf_base.copy()]
    l74[-1][4] = 1
    l75 = [[7, 73, 76], [0, 1, 3, 5, 71, 78], leaf_base.copy()]
    l75[-1][5] = 1
    l76 = [[7, 73, 79], [0, 1, 3, 5, 71, 76], leaf_base.copy()]
    l76[-1][6] = 1
    l77 = [[7, 73], [0, 1, 3, 5, 71, 76, 79], leaf_base.copy()]
    l77[-1][7] = 1
    l78 = [[7, 77], [0, 1, 3, 5, 71, 73], leaf_base.copy()]
    l78[-1][8] = 1
    l79 = [[7], [0, 1, 3, 5, 71, 73, 77], leaf_base.copy()]
    l79[-1][9] = 1

    l80 = [[80, 81, 83], [0, 1, 3, 5, 7], leaf_base.copy()]
    l80[-1][0] = 1
    l81 = [[80, 81], [0, 1, 3, 5, 7, 83], leaf_base.copy()]
    l81[-1][1] = 1
    l82 = [[80, 84], [0, 1, 3, 5, 7, 81], leaf_base.copy()]
    l82[-1][2] = 1
    l83 = [[80], [0, 1, 3, 5, 7, 81, 84], leaf_base.copy()]
    l83[-1][3] = 1
    l84 = [[82, 85, 87], [0, 1, 3, 5, 7, 80], leaf_base.copy()]
    l84[-1][4] = 1
    l85 = [[82, 85], [0, 1, 3, 5, 7, 80, 87], leaf_base.copy()]
    l85[-1][5] = 1
    l86 = [[82, 88], [0, 1, 3, 5, 7, 80, 85], leaf_base.copy()]
    l86[-1][6] = 1
    l87 = [[82], [0, 1, 3, 5, 7, 80, 85, 88], leaf_base.copy()]
    l87[-1][7] = 1
    l88 = [[86], [0, 1, 3, 5, 7, 80, 82], leaf_base.copy()]
    l88[-1][8] = 1
    l89 = [[], [0, 1, 3, 5, 7, 80, 82, 86], leaf_base.copy()]
    l89[-1][9] = 1

    # 90 leaves

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52,
        l53, l54, l55, l56, l57, l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69,
        l70, l71,
        l72, l73, l74, l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85,
        l86, l87, l88, l89
    ]

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves

# 8 nodes, 2 node at left side
def DT_zerlings_v1_8nl():
    """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT.
    2 Nodes added to the left side. total 8 nodes
    """
    dim_in = 37
    dim_out = 10

    num_decision_nodes = 8
    num_action_nodes = 81  # 9 * 9
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 14] = 1
    B[0] = 0

    # node 1
    W[1, 1] = 1
    B[1] = 30

    # node 2:
    W[2, 0] = -1
    B[2] = -20

    # node 3:
    W[3, 1] = 1
    B[3] = 18

    # node 4:
    W[4, 0] = 1
    B[4] = 40

    # node 5:
    W[5, 0] = -1
    B[5] = -40

    # node 6:
    W[6, 1] = 1
    B[6] = 20

    #node 7:
    W[7, 1] = -1
    B[7] = -20

    # ========== Action Nodes ==========

    B[8] = -10
    B[10] = 10
    B[13] = 10
    B[15] = 10

    B[17] = -10
    B[19] = 10
    B[22] = 10
    B[24] = -10

    B[26] = -10
    B[28] = 10
    B[31] = -10
    B[34] = 10

    B[35] = 10
    B[36] = -10
    B[39] = 10

    B[44] = 10
    B[45] = 10
    B[47] = 10

    B[53] = 10
    B[54] = -10
    B[57] = -10

    B[62] = 10
    B[63] = -10
    B[66] = 10

    B[71] = 10
    B[72] = 10
    B[74] = -10

    B[80] = 10
    B[81] = 10
    B[83] = 10


    # ========== Leaves ==========
    leaf_base = [0] * dim_out

    l0 = [[0, 1, 8, 9, 11], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 1, 8, 9], [11], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 1, 8, 12], [9], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 1, 8], [9, 12], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 1, 10, 13, 15], [8], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 1, 10, 13], [8, 15], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 1, 10, 16], [8, 13], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 1, 10], [8, 13, 16], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 1, 14], [8, 10], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0, 1], [8, 10, 14], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[0, 3, 17, 18, 20], [1], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[0, 3, 17, 18], [1, 20], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[0, 3, 17, 21], [1, 18], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[0, 3, 17], [1, 18, 21], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[0, 3, 19, 22, 24], [1, 17], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[0, 3, 19, 22], [1, 17, 24], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[0, 3, 19, 25], [1, 17, 22], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[0, 3, 19], [1, 17, 22, 25], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[0, 3, 23], [1, 17, 19], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[0, 3], [1, 17, 19, 23], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[0, 26, 27, 29], [1, 3], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[0, 26, 27], [1, 3, 29], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[0, 26, 30], [1, 3, 27], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[0, 26], [1, 3, 27, 30], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[0, 28, 31, 33], [1, 3, 26], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[0, 28, 31], [1, 3, 26, 33], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[0, 28, 34], [1, 3, 26, 31], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[0, 28], [1, 3, 26, 31, 34], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[0, 32], [1, 3, 26, 28], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[0], [1, 3, 26, 28, 32], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[2, 4, 35, 36, 38], [0], leaf_base.copy()]
    l30[-1][0] = 1
    l31 = [[2, 4, 35, 36], [0, 38], leaf_base.copy()]
    l31[-1][1] = 1
    l32 = [[2, 4, 35, 39], [0, 36], leaf_base.copy()]
    l32[-1][2] = 1
    l33 = [[2, 4, 35], [0, 36, 39], leaf_base.copy()]
    l33[-1][3] = 1
    l34 = [[2, 4, 37, 40, 42], [0, 35], leaf_base.copy()]
    l34[-1][4] = 1
    l35 = [[2, 4, 37, 40], [0, 35, 42], leaf_base.copy()]
    l35[-1][5] = 1
    l36 = [[2, 4, 37, 43], [0, 35, 40], leaf_base.copy()]
    l36[-1][6] = 1
    l37 = [[2, 4, 37], [0, 35, 40, 43], leaf_base.copy()]
    l37[-1][7] = 1
    l38 = [[2, 4, 41], [0, 35, 37], leaf_base.copy()]
    l38[-1][8] = 1
    l39 = [[2, 4], [0, 35, 37, 41], leaf_base.copy()]

    l40 = [[2, 6, 44, 45, 47], [0, 4], leaf_base.copy()]
    l40[-1][0] = 1
    l41 = [[2, 6, 44, 45], [0, 4, 47], leaf_base.copy()]
    l41[-1][1] = 1
    l42 = [[2, 6, 44, 48], [0, 4, 45], leaf_base.copy()]
    l42[-1][2] = 1
    l43 = [[2, 6, 44], [0, 4, 45, 48], leaf_base.copy()]
    l43[-1][3] = 1
    l44 = [[2, 6, 46, 49, 51], [0, 4, 44], leaf_base.copy()]
    l44[-1][4] = 1
    l45 = [[2, 6, 46, 49], [0, 4, 44, 51], leaf_base.copy()]
    l45[-1][5] = 1
    l46 = [[2, 6, 46, 52], [0, 4, 44, 49], leaf_base.copy()]
    l46[-1][6] = 1
    l47 = [[2, 6, 46], [0, 4, 44, 49, 52], leaf_base.copy()]
    l47[-1][7] = 1
    l48 = [[2, 6, 50], [0, 4, 44, 46], leaf_base.copy()]
    l48[-1][8] = 1
    l49 = [[2, 6], [0, 4, 44, 46, 50], leaf_base.copy()]
    l49[-1][9] = 1

    l50 = [[2, 53, 54, 56 ], [0, 4, 6], leaf_base.copy()]
    l50[-1][0] = 1
    l51 = [[2, 53, 54], [0, 4, 6, 56], leaf_base.copy()]
    l51[-1][1] = 1
    l52 = [[2, 53, 57], [0, 4, 6, 54], leaf_base.copy()]
    l52[-1][2] = 1
    l53 = [[2, 53], [0, 4, 6, 54, 57], leaf_base.copy()]
    l53[-1][3] = 1
    l54 = [[2, 55, 58, 60], [0, 4, 6, 53], leaf_base.copy()]
    l54[-1][4] = 1
    l55 = [[2, 55, 58], [0, 4, 6, 53, 60], leaf_base.copy()]
    l55[-1][5] = 1
    l56 = [[2, 55, 61], [0, 4, 6, 53, 58], leaf_base.copy()]
    l56[-1][6] = 1
    l57 = [[2, 55], [0, 4, 6, 53, 58, 61], leaf_base.copy()]
    l57[-1][7] = 1
    l58 = [[2, 59], [0, 4, 6, 53, 55], leaf_base.copy()]
    l58[-1][8] = 1
    l59 = [[2], [0, 4, 6, 53, 55, 59], leaf_base.copy()]
    l59[-1][9] = 1

    l60 = [[5, 62, 63, 65], [0, 2], leaf_base.copy()]
    l60[-1][0] = 1
    l61 = [[5, 62, 63], [0, 2, 65], leaf_base.copy()]
    l61[-1][1] = 1
    l62 = [[5, 62, 66], [0, 2, 63], leaf_base.copy()]
    l62[-1][2] = 1
    l63 = [[5, 62], [0, 2, 63, 66], leaf_base.copy()]
    l63[-1][3] = 1
    l64 = [[5, 64, 67, 69], [0, 2, 62], leaf_base.copy()]
    l64[-1][4] = 1
    l65 = [[5, 64, 67], [0, 2, 62, 69], leaf_base.copy()]
    l65[-1][5] = 1
    l66 = [[5, 64, 70], [0, 2, 62, 67], leaf_base.copy()]
    l66[-1][6] = 1
    l67 = [[5, 64], [0, 2, 62, 67, 70], leaf_base.copy()]
    l67[-1][7] = 1
    l68 = [[5, 68], [0, 2, 62, 64], leaf_base.copy()]
    l68[-1][8] = 1
    l69 = [[5], [0, 2, 62, 64, 68], leaf_base.copy()]
    l69[-1][9] = 1

    l70 = [[7, 71, 72, 74], [0, 2, 5], leaf_base.copy()]
    l70[-1][0] = 1
    l71 = [[7, 71, 72], [0, 2, 5, 74], leaf_base.copy()]
    l71[-1][1] = 1
    l72 = [[7, 71, 75], [0, 2, 5, 72], leaf_base.copy()]
    l72[-1][2] = 1
    l73 = [[7, 71], [0, 2, 5, 72, 75], leaf_base.copy()]
    l73[-1][3] = 1
    l74 = [[7, 73, 76, 78], [0, 2, 5, 71], leaf_base.copy()]
    l74[-1][4] = 1
    l75 = [[7, 73, 76], [0, 2, 5, 71, 78], leaf_base.copy()]
    l75[-1][5] = 1
    l76 = [[7, 73, 79], [0, 2, 5, 71, 76], leaf_base.copy()]
    l76[-1][6] = 1
    l77 = [[7, 73], [0, 2, 5, 71, 76, 79], leaf_base.copy()]
    l77[-1][7] = 1
    l78 = [[7, 77], [0, 2, 5, 71, 73], leaf_base.copy()]
    l78[-1][8] = 1
    l79 = [[7], [0, 2, 5, 71, 73, 77], leaf_base.copy()]
    l79[-1][9] = 1

    l80 = [[80, 81, 83], [0, 2, 5, 7], leaf_base.copy()]
    l80[-1][0] = 1
    l81 = [[80, 81], [0, 2, 5, 7, 83], leaf_base.copy()]
    l81[-1][1] = 1
    l82 = [[80, 84], [0, 2, 5, 7, 81], leaf_base.copy()]
    l82[-1][2] = 1
    l83 = [[80], [0, 2, 5, 7, 81, 84], leaf_base.copy()]
    l83[-1][3] = 1
    l84 = [[82, 85, 87], [0, 2, 5, 7, 80], leaf_base.copy()]
    l84[-1][4] = 1
    l85 = [[82, 85], [0, 2, 5, 7, 80, 87], leaf_base.copy()]
    l85[-1][5] = 1
    l86 = [[82, 88], [0, 2, 5, 7, 80, 85], leaf_base.copy()]
    l86[-1][6] = 1
    l87 = [[82], [0, 2, 5, 7, 80, 85, 88], leaf_base.copy()]
    l87[-1][7] = 1
    l88 = [[86], [0, 2, 5, 7, 80, 82], leaf_base.copy()]
    l88[-1][8] = 1
    l89 = [[], [0, 2, 5, 7, 80, 82, 86], leaf_base.copy()]
    l89[-1][9] = 1

    # 90 leaves

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9,
        l10, l11, l12, l13, l14, l15, l16, l17, l18, l19,
        l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39,
        l40, l41, l42, l43, l44, l45, l46, l47, l48, l49,
        l50, l51, l52, l53, l54, l55, l56, l57, l58, l59,
        l60, l61, l62, l63, l64, l65, l66, l67, l68, l69,
        l70, l71, l72, l73, l74, l75, l76, l77, l78, l79,
        l80, l81, l82, l83, l84, l85, l86, l87, l88, l89
    ]

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves

# V1 10 nodes
def DT_zerlings_v1_10n():
    """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT.
   10 Nodes
    """
    dim_in = 37
    dim_out = 10

    num_decision_nodes = 10
    num_action_nodes = 99  # 11 * 9
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 14] = 1
    B[0] = 0

    # node 1
    W[1, 1] = 1
    B[1] = 30

    # node 2:
    W[2, 0] = -1
    B[2] = -20

    # node 3:
    W[3, 1] = 1
    B[3] = 18

    # node 4:
    W[4, 0] = 1
    B[4] = 40

    # node 5:
    W[5, 0] = -1
    B[5] = -40

    # node 6:
    W[6, 1] = 1
    B[6] = 20

    #node 7:
    W[7, 1] = -1
    B[7] = -20

    W[8, 0] = 1
    B[8] = 30

    W[9, 1] = -1
    B[9] = 20

    # ========== Action Nodes ==========
    B[10] = -10
    B[12] = 10
    B[15] = 10
    B[17] = 10

    B[28] = -10
    B[30] = 10
    B[33] = -10
    B[36] = 10

    B[37] = -10
    B[39] = 10
    B[42] = -10
    B[45] = -10

    B[46] = -10
    B[48] = 10
    B[52] = 10

    B[55] = 10
    B[56] = -10
    B[59] = 10

    B[64] = 10
    B[65] = 10
    B[67] = 10

    B[73] = 10
    B[74] = -10
    B[77] = 10

    B[82] = 10
    B[83] = -10
    B[86] = 10

    B[91] = 10
    B[92] = 10
    B[94] = -10

    B[100] = 10
    B[101] = 10
    B[103] = 10


    # ========== Leaves ==========
    leaf_base = [0] * dim_out

    l0 = [[0, 1, 3, 10, 11, 13], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 1, 3, 10, 11], [13], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 1, 3, 10, 14], [11], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 1, 3, 10], [11, 14], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 1, 3, 12, 15, 17], [10], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 1, 3, 12, 15], [10, 17], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 1, 3, 12, 18], [10, 15], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 1, 3, 12], [10, 15, 18], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 1, 3, 16], [10, 12], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0, 1, 3], [10, 12, 16], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[0, 1, 19, 20, 22], [3], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[0, 1, 19, 20], [3, 22], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[0, 1, 19, 23], [3, 20], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[0, 1, 19], [3, 20, 23], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[0, 1, 21, 24, 26], [3, 19], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[0, 1, 21, 24], [3, 19, 26], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[0, 1, 21, 27], [3, 19, 24], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[0, 1, 21], [3, 19, 24, 27], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[0, 1, 25], [3, 19, 21], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[0, 1], [3, 19, 21, 25], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[0, 4, 7, 28, 29, 31], [1], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[0, 4, 7, 28, 29], [1, 31], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[0, 4, 7, 28, 32], [1, 29], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[0, 4, 7, 28], [1, 29, 32], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[0, 4, 7, 30, 33, 35], [1, 28], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[0, 4, 7, 30, 33], [1, 28, 35], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[0, 4, 7, 30, 36], [1, 28, 33], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[0, 4, 7, 30], [1, 28, 33, 36], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[0, 4, 34], [1, 28, 30], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[0, 4], [1, 28, 30, 34], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[0, 4, 37, 38, 40], [1, 7], leaf_base.copy()]
    l30[-1][0] = 1
    l31 = [[0, 4, 37, 38], [1, 7, 40], leaf_base.copy()]
    l31[-1][1] = 1
    l32 = [[0, 4, 37, 41], [1, 7, 38], leaf_base.copy()]
    l32[-1][2] = 1
    l33 = [[0, 4, 37], [1, 7, 38, 41], leaf_base.copy()]
    l33[-1][3] = 1
    l34 = [[0, 4, 39, 42, 44], [1, 7, 37], leaf_base.copy()]
    l34[-1][4] = 1
    l35 = [[0, 4, 39, 42], [1, 7, 37, 44], leaf_base.copy()]
    l35[-1][5] = 1
    l36 = [[0, 4, 39, 45], [1, 7, 37, 42], leaf_base.copy()]
    l36[-1][6] = 1
    l37 = [[0, 4, 39], [1, 7, 37, 42, 45], leaf_base.copy()]
    l37[-1][7] = 1
    l38 = [[0, 4, 43], [1, 7, 37, 39], leaf_base.copy()]
    l38[-1][8] = 1
    l39 = [[0, 4], [1, 7, 37, 39, 43], leaf_base.copy()]
    l39[-1][9] = 1

    l40 = [[0, 46, 47, 49], [1, 4], leaf_base.copy()]
    l40[-1][0] = 1
    l41 = [[0, 46, 47], [1, 4, 49], leaf_base.copy()]
    l41[-1][1] = 1
    l42 = [[0, 46, 50], [1, 4, 47], leaf_base.copy()]
    l42[-1][2] = 1
    l43 = [[0, 46], [1, 4, 47, 50], leaf_base.copy()]
    l43[-1][3] = 1
    l44 = [[0, 48, 51, 53], [1, 4, 46], leaf_base.copy()]
    l44[-1][4] = 1
    l45 = [[0, 48, 51], [1, 4, 46, 53], leaf_base.copy()]
    l45[-1][5] = 1
    l46 = [[0, 48, 54], [1, 4, 46, 51], leaf_base.copy()]
    l46[-1][6] = 1
    l47 = [[0, 48], [1, 4, 46, 51, 54], leaf_base.copy()]
    l47[-1][7] = 1
    l48 = [[0, 52], [1, 4, 46, 48], leaf_base.copy()]
    l48[-1][8] = 1
    l49 = [[0], [1, 4, 46, 48, 52], leaf_base.copy()]
    l49[-1][9] = 1

    l50 = [[2, 5, 55, 56, 58], [0], leaf_base.copy()]
    l50[-1][0] = 1
    l51 = [[2, 5, 55, 56], [0, 58], leaf_base.copy()]
    l51[-1][1] = 1
    l52 = [[2, 5, 55, 59], [0, 56], leaf_base.copy()]
    l52[-1][2] = 1
    l53 = [[2, 5, 55], [0, 56, 59], leaf_base.copy()]
    l53[-1][3] = 1
    l54 = [[2, 5, 57, 60, 62], [0, 55], leaf_base.copy()]
    l54[-1][4] = 1
    l55 = [[2, 5, 57, 60], [0, 55, 62], leaf_base.copy()]
    l55[-1][5] = 1
    l56 = [[2, 5, 57, 63], [0, 55, 60], leaf_base.copy()]
    l56[-1][6] = 1
    l57 = [[2, 5, 57], [0, 55, 60, 63], leaf_base.copy()]
    l57[-1][7] = 1
    l58 = [[2, 5, 61], [0, 55, 57], leaf_base.copy()]
    l58[-1][8] = 1
    l59 = [[2, 5], [0, 55, 57, 61], leaf_base.copy()]
    l59[-1][9] = 1

    l60 = [[2, 8, 64, 65, 67], [0, 5], leaf_base.copy()]
    l60[-1][0] = 1
    l61 = [[2, 8, 64, 65], [0, 5, 67], leaf_base.copy()]
    l61[-1][1] = 1
    l62 = [[2, 8, 64, 68], [0, 5, 65], leaf_base.copy()]
    l62[-1][2] = 1
    l63 = [[2, 8, 64], [0, 5, 65, 68], leaf_base.copy()]
    l63[-1][3] = 1
    l64 = [[2, 8, 66, 69, 71], [0, 5, 64], leaf_base.copy()]
    l64[-1][4] = 1
    l65 = [[2, 8, 66, 69], [0, 5, 64, 71], leaf_base.copy()]
    l65[-1][5] = 1
    l66 = [[2, 8, 66, 72], [0, 5, 64, 69], leaf_base.copy()]
    l66[-1][6] = 1
    l67 = [[2, 8, 66], [0, 5, 64, 69, 72], leaf_base.copy()]
    l67[-1][7] = 1
    l68 = [[2, 8, 70], [0, 5, 64, 66], leaf_base.copy()]
    l68[-1][8] = 1
    l69 = [[2, 8], [0, 5, 64, 66, 70], leaf_base.copy()]
    l69[-1][9] = 1

    l70 = [[2, 73, 74, 76], [0, 5, 8], leaf_base.copy()]
    l70[-1][0] = 1
    l71 = [[2, 73, 74], [0, 5, 8, 76], leaf_base.copy()]
    l71[-1][1] = 1
    l72 = [[2, 73, 77], [0, 5, 8, 74], leaf_base.copy()]
    l72[-1][2] = 1
    l73 = [[2, 73], [0, 5, 8, 74, 77], leaf_base.copy()]
    l73[-1][3] = 1
    l74 = [[2, 75, 78, 80], [0, 5, 8, 73], leaf_base.copy()]
    l74[-1][4] = 1
    l75 = [[2, 75, 78], [0, 5, 8, 73, 80], leaf_base.copy()]
    l75[-1][5] = 1
    l76 = [[2, 75, 81], [0, 5, 8, 73, 78], leaf_base.copy()]
    l76[-1][6] = 1
    l77 = [[2, 75], [0, 5, 8, 73, 78, 81], leaf_base.copy()]
    l77[-1][7] = 1
    l78 = [[2, 79], [0, 5, 8, 73, 75], leaf_base.copy()]
    l78[-1][8] = 1
    l79 = [[2], [0, 5, 8, 73, 75, 79], leaf_base.copy()]
    l79[-1][9] = 1

    l80 = [[6, 82, 83, 85], [0, 2], leaf_base.copy()]
    l80[-1][0] = 1
    l81 = [[6, 82, 83], [0, 2, 85], leaf_base.copy()]
    l81[-1][1] = 1
    l82 = [[6, 82, 86], [0, 2, 83], leaf_base.copy()]
    l82[-1][2] = 1
    l83 = [[6, 82], [0, 2, 83, 86], leaf_base.copy()]
    l83[-1][3] = 1
    l84 = [[6, 84, 87, 89], [0, 2, 82], leaf_base.copy()]
    l84[-1][4] = 1
    l85 = [[6, 84, 87], [0, 2, 82, 89], leaf_base.copy()]
    l85[-1][5] = 1
    l86 = [[6, 84, 90], [0, 2, 82, 87], leaf_base.copy()]
    l86[-1][6] = 1
    l87 = [[6, 84], [0, 2, 82, 87, 90], leaf_base.copy()]
    l87[-1][7] = 1
    l88 = [[6, 88], [0, 2, 82, 84], leaf_base.copy()]
    l88[-1][8] = 1
    l89 = [[6], [0, 2, 82, 84, 88], leaf_base.copy()]
    l89[-1][9] = 1

    l90 = [[9, 91, 92, 94], [0, 2, 6], leaf_base.copy()]
    l90[-1][0] = 1
    l91 = [[9, 91, 92], [0, 2, 6, 94], leaf_base.copy()]
    l91[-1][1] = 1
    l92 = [[9, 91, 95], [0, 2, 6, 92], leaf_base.copy()]
    l92[-1][2] = 1
    l93 = [[9, 91], [0, 2, 6, 92, 95], leaf_base.copy()]
    l93[-1][3] = 1
    l94 = [[9, 93, 96, 98], [0, 2, 6, 91], leaf_base.copy()]
    l94[-1][4] = 1
    l95 = [[9, 93, 96], [0, 2, 6, 91, 98], leaf_base.copy()]
    l95[-1][5] = 1
    l96 = [[9, 93, 99], [0, 2, 6, 91, 96], leaf_base.copy()]
    l96[-1][6] = 1
    l97 = [[9, 93], [0, 2, 6, 91, 96, 99], leaf_base.copy()]
    l97[-1][7] = 1
    l98 = [[9, 97], [0, 2, 6, 91, 93], leaf_base.copy()]
    l98[-1][8] = 1
    l99 = [[9], [0, 2, 6, 91, 93, 97], leaf_base.copy()]
    l99[-1][9] = 1

    l100 = [[100, 101, 103], [0, 2, 6, 9], leaf_base.copy()]
    l100[-1][0] = 1
    l101 = [[100, 101], [0, 2, 6, 9, 103], leaf_base.copy()]
    l101[-1][1] = 1
    l102 = [[100, 104], [0, 2, 6, 9, 101], leaf_base.copy()]
    l102[-1][2] = 1
    l103 = [[100], [0, 2, 6, 9, 101, 104], leaf_base.copy()]
    l103[-1][3] = 1
    l104 = [[102, 105, 107], [0, 2, 6, 9, 100], leaf_base.copy()]
    l104[-1][4] = 1
    l105 = [[102, 105], [0, 2, 6, 9, 100, 107], leaf_base.copy()]
    l105[-1][5] = 1
    l106 = [[102, 108], [0, 2, 6, 9, 100, 105], leaf_base.copy()]
    l106[-1][6] = 1
    l107 = [[102], [0, 2, 6, 9, 100, 105, 108], leaf_base.copy()]
    l107[-1][7] = 1
    l108 = [[106], [0, 2, 6, 9, 100, 102], leaf_base.copy()]
    l108[-1][8] = 1
    l109 = [[], [0, 2, 6, 9, 100, 102, 106], leaf_base.copy()]
    l109[-1][9] = 1

    # 110 leaves

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69, l70, l71,
        l72, l73, l74, l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85,
        l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96, l97, l98, l99,
        l100, l101, l102, l103, l104, l105, l106, l107, l108, l109
    ]

    return dim_in, dim_out, num_decision_nodes, num_action_nodes, W, B, init_leaves


def DT_zerlings_v1_31n():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 37
    dim_out = 10
    num_nodes = 31

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 10

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -10

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -10

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 10

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -10

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -10

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -10

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -10

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -10

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -10

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -10

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 10

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -10

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -10

    # node 14: -x0 - 0.2 > 0
    W[14, 0] = 1
    B[14] = 10

    # node 15: -x1 + 1.1 > 0
    W[15, 1] = -1
    B[15] = 10

    # node 16: -x3 - 0.2 > 0
    W[16, 3] = -1
    B[16] = -10

    # node 17: x5 - 0.1 > 0
    W[17, 5] = 1
    B[17] = -10

    # node 18: -x5 + 0.1 > 0
    W[18, 5] = -1
    B[18] = 10

    # node 19: x6 + x7 - 0.9 > 0
    W[19, 6] = 1
    W[19, 7] = 1
    B[19] = -10

    # node 20: -x5 -0.1 > 0
    W[20, 5] = -1
    B[20] = -10

    # node 21: x6 + x7 - 0.9 > 0
    W[21, 6] = 1
    W[21, 7] = 1
    B[21] = -10

    # node 22: x6 + x7 - 0.9 > 0
    W[22, 6] = 1
    W[22, 7] = 1
    B[22] = -10

    # node 23: x0 - 0.2 > 0
    W[23, 0] = 1
    B[23] = -10

    # node 24: x6 + x7 - 0.9 > 0
    W[24, 6] = 1
    W[24, 7] = 1
    B[24] = -10

    # node 25: x0 - 0.2 > 0
    W[25, 0] = 1
    B[25] = -10

    # node 26: x5 + 0.1 > 0
    W[26, 5] = 1
    B[26] = 10

    # node 27: -x0 - 0.2 > 0
    W[27, 0] = -1
    B[27] = -10

    # node 28: -x0 - 0.2 > 0
    W[28, 0] = -1
    B[28] = -10

    # node 29: -x0 - 0.2 > 0
    W[29, 0] = 1
    B[29] = 10

    # node 30: x5 + 0.1 > 0
    W[30, 5] = 1
    B[30] = 10

    # 2. Define leaf nodes [T] [F] [Action]
    leaf_base = [0] * dim_out
    l0 = [[0, 1, 3, 7, 15], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 1, 3, 7], [15], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 1, 3, 16], [7], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 1, 3], [7, 16], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 1, 8, 17], [3], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 1, 8], [3, 17], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 1, 18], [3, 8], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 1], [3, 8, 18], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 4, 9, 19], [1], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0, 4, 9], [1, 19], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[0, 4, 20], [1, 9], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[0, 4], [1, 9, 20], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[0, 10, 21], [1, 4], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[0, 10], [1, 4, 21], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[0, 22], [1, 4, 10], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[0], [1, 4, 10, 22], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[2, 5, 11, 23], [0], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[2, 5, 11], [0, 23], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[2, 5, 24], [0, 11], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[2, 5], [0, 11, 24], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[2, 12, 25], [0, 5], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[2, 12], [0, 5, 25], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[2, 26], [0, 5, 12], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[2], [0, 5, 12, 26], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[6, 13, 27], [0, 2], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[6, 13], [0, 2, 27], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[6, 28], [0, 2, 13], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[6], [0, 2, 13, 28], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[14, 29], [0, 2, 6], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[14], [0, 2, 6, 29], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[30], [0, 2, 6, 14], leaf_base.copy()]
    l30[-1][8] = 1
    l31 = [[], [0, 2, 6, 14, 30], leaf_base.copy()]
    l31[-1][9] = 1  # No Move

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31
    ]

    return dim_in, dim_out, W, B, init_leaves


# def DT_zerlings_l2_0():
#     """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT."""
#     dim_in = 37
#     dim_out = 10

#     num_decision_nodes = 6
#     num_action_nodes = 63  # 7 * 9
#     num_nodes = num_decision_nodes + num_action_nodes

#     # 1. Define nodes of the tree of the form X.Wt + B > 0
#     # State: X = [x0, x1]
#     W = np.zeros((num_nodes, dim_in))  # each row represents a node
#     B = np.zeros(num_nodes)  # Biases of each node

#     # node 0
#     W[0, 14] = 1
#     B[0] = 0

#     # node 1
#     W[1, 1] = 1
#     B[1] = 30

#     # node 2:
#     W[2, 0] = -1
#     B[2] = -20

#     # node 3:
#     W[3, 1] = 1
#     B[3] = 18

#     # node 4:
#     W[4, 0] = 1
#     B[4] = 40

#     # node 5:
#     W[5, 0] = -1
#     B[5] = -40

#     # ========== Action Nodes ==========
#     W[6, 2] = 1
#     B[6] = 1
#     W[7, 3] = 1
#     B[7] = 1

#     W[15, 0] = 1
#     B[15] = 1
#     W[16, 1] = -1
#     B[16] = 1
#     W[19, 0] = 1
#     B[19] = 1

#     W[24, 0] = 1
#     B[24] = 1
#     W[25, 1] = -1
#     B[25] = -1
#     W[28, 3] = 1
#     B[28] = -1

#     W[33, 0] = 1
#     B[33] = -1
#     W[36, 1] = -1
#     B[36] = 1
#     W[41, 0] = 1
#     B[41] = 1

#     B[34] = -1
#     B[38] = 1
#     B[45] = 1

#     B[51] = -1
#     B[54] = 1
#     B[59] = -1

#     B[52] = -1
#     B[56] = 1
#     B[63] = -1

#     # ========== Leaves ==========
#     leaf_base = [0] * dim_out

#     l0 = [[0, 6, 7, 9], [], leaf_base.copy()]
#     l0[-1][0] = 1
#     l1 = [[0, 6, 7], [9], leaf_base.copy()]
#     l1[-1][1] = 1
#     l2 = [[0, 6, 10], [7], leaf_base.copy()]
#     l2[-1][2] = 1
#     l3 = [[0, 6], [7, 10], leaf_base.copy()]
#     l3[-1][3] = 1
#     l4 = [[0, 8, 11], [6], leaf_base.copy()]
#     l4[-1][4] = 1
#     l5 = [[0, 8], [6, 11], leaf_base.copy()]
#     l5[-1][5] = 1
#     l6 = [[0, 12, 13], [6, 8], leaf_base.copy()]
#     l6[-1][6] = 1
#     l7 = [[0, 12], [6, 8, 13], leaf_base.copy()]
#     l7[-1][7] = 1
#     l8 = [[0, 14], [6, 8, 12], leaf_base.copy()]
#     l8[-1][8] = 1
#     l9 = [[0], [6, 8, 12, 14], leaf_base.copy()]
#     l9[-1][9] = 1

#     l10 = [[1, 2, 15, 16, 18], [0], leaf_base.copy()]
#     l10[-1][0] = 1
#     l11 = [[1, 2, 15, 16], [0, 18], leaf_base.copy()]
#     l11[-1][1] = 1
#     l12 = [[1, 2, 15, 19], [0, 16], leaf_base.copy()]
#     l12[-1][2] = 1
#     l13 = [[1, 2, 15], [0, 16, 19], leaf_base.copy()]
#     l13[-1][3] = 1
#     l14 = [[1, 2, 17, 20], [0, 15], leaf_base.copy()]
#     l14[-1][4] = 1
#     l15 = [[1, 2, 17], [0, 15, 20], leaf_base.copy()]
#     l15[-1][5] = 1
#     l16 = [[1, 2, 21, 22], [0, 15, 17], leaf_base.copy()]
#     l16[-1][6] = 1
#     l17 = [[1, 2, 21], [0, 15, 17, 22], leaf_base.copy()]
#     l17[-1][7] = 1
#     l18 = [[1, 2, 23], [0, 15, 17, 21], leaf_base.copy()]
#     l18[-1][8] = 1
#     l19 = [[1, 2], [0, 15, 17, 21, 23], leaf_base.copy()]
#     l19[-1][9] = 1

#     l20 = [[3, 24, 25, 27], [
#         0,
#         1,
#     ], leaf_base.copy()]
#     l20[-1][0] = 1
#     l21 = [[3, 24, 25], [0, 1, 27], leaf_base.copy()]
#     l21[-1][1] = 1
#     l22 = [[3, 24, 28], [0, 1, 25], leaf_base.copy()]
#     l22[-1][2] = 1
#     l23 = [[3, 24], [0, 1, 25, 28], leaf_base.copy()]
#     l23[-1][3] = 1
#     l24 = [[3, 26, 29], [0, 1, 24], leaf_base.copy()]
#     l24[-1][4] = 1
#     l25 = [[3, 26], [0, 1, 24, 29], leaf_base.copy()]
#     l25[-1][5] = 1
#     l26 = [[3, 30, 31], [0, 1, 24, 26], leaf_base.copy()]
#     l26[-1][6] = 1
#     l27 = [[3, 30], [0, 1, 24, 26, 31], leaf_base.copy()]
#     l27[-1][7] = 1
#     l28 = [[3, 32], [0, 1, 24, 26, 30], leaf_base.copy()]
#     l28[-1][8] = 1
#     l29 = [[3], [0, 1, 24, 26, 30, 32], leaf_base.copy()]
#     l29[-1][9] = 1

#     l30 = [[1, 4, 33, 35, 39], [0, 2], leaf_base.copy()]
#     l30[-1][0] = 1
#     l31 = [[1, 4, 33, 35], [0, 2, 39], leaf_base.copy()]
#     l31[-1][1] = 1
#     l32 = [[1, 4, 33, 40], [0, 2, 35], leaf_base.copy()]
#     l32[-1][2] = 1
#     l33 = [[1, 4, 33], [0, 2, 35, 40], leaf_base.copy()]
#     l33[-1][3] = 1
#     l34 = [[1, 4, 36, 41], [0, 2, 33], leaf_base.copy()]
#     l34[-1][4] = 1
#     l35 = [[1, 4, 36], [0, 2, 33, 41], leaf_base.copy()]
#     l35[-1][5] = 1
#     l36 = [[1, 4, 42, 47], [0, 2, 33, 36], leaf_base.copy()]
#     l36[-1][6] = 1
#     l37 = [[1, 4, 42], [0, 2, 33, 36, 47], leaf_base.copy()]
#     l37[-1][7] = 1
#     l38 = [[1, 4, 48], [0, 2, 33, 36, 42], leaf_base.copy()]
#     l38[-1][8] = 1
#     l39 = [[1, 4], [0, 2, 33, 36, 42, 48], leaf_base.copy()]
#     l39[-1][9] = 1

#     l40 = [[1, 34, 37, 43], [0, 2, 4], leaf_base.copy()]
#     l40[-1][0] = 1
#     l41 = [[1, 34, 37], [0, 2, 4, 43], leaf_base.copy()]
#     l41[-1][1] = 1
#     l42 = [[1, 34, 44], [0, 2, 4, 37], leaf_base.copy()]
#     l42[-1][2] = 1
#     l43 = [[1, 34], [0, 2, 4, 37, 44], leaf_base.copy()]
#     l43[-1][3] = 1
#     l44 = [[1, 38, 45], [0, 2, 4, 34], leaf_base.copy()]
#     l44[-1][4] = 1
#     l45 = [[1, 38], [0, 2, 4, 34, 45], leaf_base.copy()]
#     l45[-1][5] = 1
#     l46 = [[1, 46, 49], [0, 2, 4, 34, 38], leaf_base.copy()]
#     l46[-1][6] = 1
#     l47 = [[1, 46], [0, 2, 4, 34, 38, 49], leaf_base.copy()]
#     l47[-1][7] = 1
#     l48 = [[1, 50], [0, 2, 4, 34, 38, 46], leaf_base.copy()]
#     l48[-1][8] = 1
#     l49 = [[1], [0, 2, 4, 34, 38, 46, 50], leaf_base.copy()]
#     l49[-1][9] = 1

#     l50 = [[5, 51, 53, 57], [0, 1, 3], leaf_base.copy()]
#     l50[-1][0] = 1
#     l51 = [[5, 51, 53], [0, 1, 3, 57], leaf_base.copy()]
#     l51[-1][1] = 1
#     l52 = [[5, 51, 58], [0, 1, 3, 53], leaf_base.copy()]
#     l52[-1][2] = 1
#     l53 = [[5, 51], [0, 1, 3, 53, 58], leaf_base.copy()]
#     l53[-1][3] = 1
#     l54 = [[5, 54, 59], [0, 1, 3, 51], leaf_base.copy()]
#     l54[-1][4] = 1
#     l55 = [[5, 54], [0, 1, 3, 51, 59], leaf_base.copy()]
#     l55[-1][5] = 1
#     l56 = [[5, 60, 65], [0, 1, 3, 51, 54], leaf_base.copy()]
#     l56[-1][6] = 1
#     l57 = [[5, 60], [0, 1, 3, 51, 54, 65], leaf_base.copy()]
#     l57[-1][7] = 1
#     l58 = [[5, 66], [0, 1, 3, 51, 54, 60], leaf_base.copy()]
#     l58[-1][8] = 1
#     l59 = [[5], [0, 1, 3, 51, 54, 60, 66], leaf_base.copy()]
#     l59[-1][9] = 1

#     l60 = [[52, 55, 61], [0, 1, 3, 5], leaf_base.copy()]
#     l60[-1][0] = 1
#     l61 = [[52, 55], [0, 1, 3, 5, 61], leaf_base.copy()]
#     l61[-1][1] = 1
#     l62 = [[52, 62], [0, 1, 3, 5, 55], leaf_base.copy()]
#     l62[-1][2] = 1
#     l63 = [[52], [0, 1, 3, 5, 55, 62], leaf_base.copy()]
#     l63[-1][3] = 1
#     l64 = [[56, 63], [0, 1, 3, 5, 52], leaf_base.copy()]
#     l64[-1][4] = 1
#     l65 = [[56], [0, 1, 3, 5, 52, 63], leaf_base.copy()]
#     l65[-1][5] = 1
#     l66 = [[64, 67], [0, 1, 3, 5, 52, 56], leaf_base.copy()]
#     l66[-1][6] = 1
#     l67 = [[64], [0, 1, 3, 5, 52, 56, 67], leaf_base.copy()]
#     l67[-1][7] = 1
#     l68 = [[68], [0, 1, 3, 5, 52, 56, 64], leaf_base.copy()]
#     l68[-1][8] = 1
#     l69 = [[], [0, 1, 3, 5, 52, 56, 64, 68], leaf_base.copy()]
#     l69[-1][9] = 1

#     # 70 leaves

#     init_leaves = [
#         l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
#         l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
#         l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
#         l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
#         l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69
#     ]

#     return dim_in, dim_out, W, B, init_leaves


def DT_zerlings_l2():
    """DT for find and defeat Zerlings. Returns node and leaf weights which represent a DT.
    8 nodes. 2 nodes are appended to right most node
    """
    dim_in = 37
    dim_out = 10

    num_decision_nodes = 8
    num_action_nodes = 81  # 9 * 9
    num_nodes = num_decision_nodes + num_action_nodes

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0
    W[0, 14] = 1
    B[0] = 0

    # node 1
    W[1, 1] = 1
    B[1] = 30

    # node 2:
    W[2, 0] = -1
    B[2] = -20

    # node 3:
    W[3, 1] = 1
    B[3] = 18

    # node 4:
    W[4, 0] = 1
    B[4] = 40

    # node 5:
    W[5, 0] = -1
    B[5] = -40

    # node 6:
    W[6, 1] = 1
    B[6] = 20

    #node 7:
    W[7, 1] = -1
    B[7] = -20

    # ========== Action Nodes ==========
    B[8] = -10
    B[10] = 10
    B[13] = 10
    B[15] = 10

    B[17] = 10
    B[18] = -10
    B[21] = 10

    # new addition after training
    B[26] = 10
    B[27] = -10
    B[30] = 10

    B[35] = 10
    B[36] = 10
    B[38] = 10


    B[44] = 10
    B[45] = -10
    B[48] = -10 # used to be 10 (new addition to the training)

    B[53] = 10
    B[54] = 10
    B[56] = -10

    B[62] = 10
    B[63] = -10
    B[66] = -10

    B[71] = 10
    B[72] = 10
    B[74] = 10

    B[80] = 10
    B[81] = -10
    B[84] = 10
    # ========== Leaves ==========
    leaf_base = [0] * dim_out

    l0 = [[0, 8, 9, 11], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 8, 9], [11], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 8, 12], [9], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 8], [9, 12], leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 10, 13, 15], [8], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 10, 13], [8, 15], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 10, 16], [8, 13], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 10], [8, 13, 16], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 14], [8, 10], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0], [8, 10, 14], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[1, 2, 17, 18, 20], [0], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[1, 2, 17, 18], [0, 20], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[1, 2, 17, 21], [0, 18], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[1, 2, 17], [0, 18, 21], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[1, 2, 19, 22, 24], [0, 17], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[1, 2, 19, 22], [0, 17, 24], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[1, 2, 19, 25], [0, 17, 22], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[1, 2, 19], [0, 17, 22, 25], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[1, 2, 23], [0, 17, 19], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[1, 2], [0, 17, 19, 23], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[3, 26, 27, 29], [0, 1], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[3, 26, 27], [0, 1, 29], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[3, 26, 30], [0, 1, 27], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[3, 26], [0, 1, 27, 30], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[3, 28, 31, 33], [0, 1, 26], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[3, 28, 31], [0, 1, 26, 33], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[3, 28, 34], [0, 1, 26, 31], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[3, 28], [0, 1, 26, 31, 34], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[3, 32], [0, 1, 26, 28], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[3], [0, 1, 26, 28, 32], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[1, 4, 35, 36, 38], [0, 2], leaf_base.copy()]
    l30[-1][0] = 1
    l31 = [[1, 4, 35, 36], [0, 2, 38], leaf_base.copy()]
    l31[-1][1] = 1
    l32 = [[1, 4, 35, 39], [0, 2, 36], leaf_base.copy()]
    l32[-1][2] = 1
    l33 = [[1, 4, 35], [0, 2, 36, 39], leaf_base.copy()]
    l33[-1][3] = 1
    l34 = [[1, 4, 37, 40, 42], [0, 2, 35], leaf_base.copy()]
    l34[-1][4] = 1
    l35 = [[1, 4, 37, 40], [0, 2, 35, 42], leaf_base.copy()]
    l35[-1][5] = 1
    l36 = [[1, 4, 37, 43], [0, 2, 35, 40], leaf_base.copy()]
    l36[-1][6] = 1
    l37 = [[1, 4, 37], [0, 2, 35, 40, 43], leaf_base.copy()]
    l37[-1][7] = 1
    l38 = [[1, 4, 41], [0, 2, 35, 37], leaf_base.copy()]
    l38[-1][8] = 1
    l39 = [[1, 4], [0, 2, 35, 37, 41], leaf_base.copy()]
    l39[-1][9] = 1

    l40 = [[1, 44, 45, 47], [0, 2, 4], leaf_base.copy()]
    l40[-1][0] = 1
    l41 = [[1, 44, 45], [0, 2, 4, 47], leaf_base.copy()]
    l41[-1][1] = 1
    l42 = [[1, 44, 48], [0, 2, 4, 45], leaf_base.copy()]
    l42[-1][2] = 1
    l43 = [[1, 44], [0, 2, 4, 45, 48], leaf_base.copy()]
    l43[-1][3] = 1
    l44 = [[1, 46, 49, 51], [0, 2, 4, 44], leaf_base.copy()]
    l44[-1][4] = 1
    l45 = [[1, 46, 49], [0, 2, 4, 44, 51], leaf_base.copy()]
    l45[-1][5] = 1
    l46 = [[1, 46, 52], [0, 2, 4, 44, 49], leaf_base.copy()]
    l46[-1][6] = 1
    l47 = [[1, 46], [0, 2, 4, 44, 49, 52], leaf_base.copy()]
    l47[-1][7] = 1
    l48 = [[1, 50], [0, 2, 4, 44, 46], leaf_base.copy()]
    l48[-1][8] = 1
    l49 = [[1], [0, 2, 4, 44, 46, 50], leaf_base.copy()]
    l49[-1][9] = 1

    # fix here Node 6
    l50 = [[5, 6, 53, 54, 56], [0, 1, 3], leaf_base.copy()]
    l50[-1][0] = 1
    l51 = [[5, 6, 53, 54], [0, 1, 3, 56], leaf_base.copy()]
    l51[-1][1] = 1
    l52 = [[5, 6, 53, 57], [0, 1, 3, 54], leaf_base.copy()]
    l52[-1][2] = 1
    l53 = [[5, 6, 53], [0, 1, 3, 54, 57], leaf_base.copy()]
    l53[-1][3] = 1
    l54 = [[5, 6, 55, 58, 60], [0, 1, 3, 53], leaf_base.copy()]
    l54[-1][4] = 1
    l55 = [[5, 6, 55, 58], [0, 1, 3, 53, 60], leaf_base.copy()]
    l55[-1][5] = 1
    l56 = [[5, 6, 55, 61], [0, 1, 3, 53, 58], leaf_base.copy()]
    l56[-1][6] = 1
    l57 = [[5, 6, 55], [0, 1, 3, 53, 58, 61], leaf_base.copy()]
    l57[-1][7] = 1
    l58 = [[5, 6, 59], [0, 1, 3, 53, 55], leaf_base.copy()]
    l58[-1][8] = 1
    l59 = [[5, 6], [0, 1, 3, 53, 55, 59], leaf_base.copy()]
    l59[-1][9] = 1

    l60 = [[5, 62, 63, 65], [0, 1, 3, 6], leaf_base.copy()]
    l60[-1][0] = 1
    l61 = [[5, 62, 63], [0, 1, 3, 6, 65], leaf_base.copy()]
    l61[-1][1] = 1
    l62 = [[5, 62, 66], [0, 1, 3, 6, 63], leaf_base.copy()]
    l62[-1][2] = 1
    l63 = [[5, 62], [0, 1, 3, 6, 63, 66], leaf_base.copy()]
    l63[-1][3] = 1
    l64 = [[5, 64, 67, 69], [0, 1, 3, 6, 62], leaf_base.copy()]
    l64[-1][4] = 1
    l65 = [[5, 64, 67], [0, 1, 3, 6, 62, 69], leaf_base.copy()]
    l65[-1][5] = 1
    l66 = [[5, 64, 70], [0, 1, 3, 6, 62, 67], leaf_base.copy()]
    l66[-1][6] = 1
    l67 = [[5, 64], [0, 1, 3, 6, 62, 67, 70], leaf_base.copy()]
    l67[-1][7] = 1
    l68 = [[5, 68], [0, 1, 3, 6, 62, 64], leaf_base.copy()]
    l68[-1][8] = 1
    l69 = [[5], [0, 1, 3, 6, 62, 64, 68], leaf_base.copy()]
    l69[-1][9] = 1

    l70 = [[7, 71, 72, 74], [0, 1, 3, 5], leaf_base.copy()]
    l70[-1][0] = 1
    l71 = [[7, 71, 72], [0, 1, 3, 5, 74], leaf_base.copy()]
    l71[-1][1] = 1
    l72 = [[7, 71, 75], [0, 1, 3, 5, 72], leaf_base.copy()]
    l72[-1][2] = 1
    l73 = [[7, 71], [0, 1, 3, 5, 72, 75], leaf_base.copy()]
    l73[-1][3] = 1
    l74 = [[7, 73, 76, 78], [0, 1, 3, 5, 71], leaf_base.copy()]
    l74[-1][4] = 1
    l75 = [[7, 73, 76], [0, 1, 3, 5, 71, 78], leaf_base.copy()]
    l75[-1][5] = 1
    l76 = [[7, 73, 79], [0, 1, 3, 5, 71, 76], leaf_base.copy()]
    l76[-1][6] = 1
    l77 = [[7, 73], [0, 1, 3, 5, 71, 76, 79], leaf_base.copy()]
    l77[-1][7] = 1
    l78 = [[7, 77], [0, 1, 3, 5, 71, 73], leaf_base.copy()]
    l78[-1][8] = 1
    l79 = [[7], [0, 1, 3, 5, 71, 73, 77], leaf_base.copy()]
    l79[-1][9] = 1

    l80 = [[80, 81, 83], [0, 1, 3, 5, 7], leaf_base.copy()]
    l80[-1][0] = 1
    l81 = [[80, 81], [0, 1, 3, 5, 7, 83], leaf_base.copy()]
    l81[-1][1] = 1
    l82 = [[80, 84], [0, 1, 3, 5, 7, 81], leaf_base.copy()]
    l82[-1][2] = 1
    l83 = [[80], [0, 1, 3, 5, 7, 81, 84], leaf_base.copy()]
    l83[-1][3] = 1
    l84 = [[82, 85, 87], [0, 1, 3, 5, 7, 80], leaf_base.copy()]
    l84[-1][4] = 1
    l85 = [[82, 85], [0, 1, 3, 5, 7, 80, 87], leaf_base.copy()]
    l85[-1][5] = 1
    l86 = [[82, 88], [0, 1, 3, 5, 7, 80, 85], leaf_base.copy()]
    l86[-1][6] = 1
    l87 = [[82], [0, 1, 3, 5, 7, 80, 85, 88], leaf_base.copy()]
    l87[-1][7] = 1
    l88 = [[86], [0, 1, 3, 5, 7, 80, 82], leaf_base.copy()]
    l88[-1][8] = 1
    l89 = [[], [0, 1, 3, 5, 7, 80, 82, 86], leaf_base.copy()]
    l89[-1][9] = 1

    # 90 leaves

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42, l43,
        l44, l45, l46, l47, l48, l49, l50, l51, l52, l53, l54, l55, l56, l57,
        l58, l59, l60, l61, l62, l63, l64, l65, l66, l67, l68, l69, l70, l71,
        l72, l73, l74, l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85,
        l86, l87, l88, l89
    ]

    return dim_in, dim_out, W, B, init_leaves


def DT_zerlings_minimal():
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    dim_in = 37
    dim_out = 10
    num_nodes = 31

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    W = np.zeros((num_nodes, dim_in))  # each row represents a node
    B = np.zeros(num_nodes)  # Biases of each node

    # node 0: -x1 + 1.1 > 0
    W[0, 1] = -1
    B[0] = 10

    # node 1: -x3 - 0.2 > 0
    W[1, 3] = -1
    B[1] = -10

    # node 2: x5 - 0.1 > 0
    W[2, 5] = 1
    B[2] = -10

    # node 3: -x5 + 0.1 > 0
    W[3, 5] = -1
    B[3] = 10

    # node 4: x6 + x7 - 0.9 > 0
    W[4, 6] = 1
    W[4, 7] = 1
    B[4] = -10

    # node 5: -x5 -0.1 > 0
    W[5, 5] = -1
    B[5] = -10

    # node 6: x6 + x7 - 0.9 > 0
    W[6, 6] = 1
    W[6, 7] = 1
    B[6] = -10

    # node 7: x6 + x7 - 0.9 > 0
    W[7, 6] = 1
    W[7, 7] = 1
    B[7] = -10

    # node 8: x0 - 0.2 > 0
    W[8, 0] = 1
    B[8] = -10

    # node 9: x6 + x7 - 0.9 > 0
    W[9, 6] = 1
    W[9, 7] = 1
    B[9] = -10

    # node 10: x0 - 0.2 > 0
    W[10, 0] = 1
    B[10] = -10

    # node 11: x5 + 0.1 > 0
    W[11, 5] = 1
    B[11] = 10

    # node 12: -x0 - 0.2 > 0
    W[12, 0] = -1
    B[12] = -10

    # node 13: -x0 - 0.2 > 0
    W[13, 0] = -1
    B[13] = -10

    # node 14: -x0 - 0.2 > 0
    W[14, 0] = 1
    B[14] = 10

    # node 15: -x1 + 1.1 > 0
    W[15, 1] = -1
    B[15] = 10

    # node 16: -x3 - 0.2 > 0
    W[16, 3] = -1
    B[16] = -10

    # node 17: x5 - 0.1 > 0
    W[17, 5] = 1
    B[17] = -10

    # node 18: -x5 + 0.1 > 0
    W[18, 5] = -1
    B[18] = 10

    # node 19: x6 + x7 - 0.9 > 0
    W[19, 6] = 1
    W[19, 7] = 1
    B[19] = -10

    # node 20: -x5 -0.1 > 0
    W[20, 5] = -1
    B[20] = -10

    # node 21: x6 + x7 - 0.9 > 0
    W[21, 6] = 1
    W[21, 7] = 1
    B[21] = -10

    # node 22: x6 + x7 - 0.9 > 0
    W[22, 6] = 1
    W[22, 7] = 1
    B[22] = -10

    # node 23: x0 - 0.2 > 0
    W[23, 0] = 1
    B[23] = -10

    # node 24: x6 + x7 - 0.9 > 0
    W[24, 6] = 1
    W[24, 7] = 1
    B[24] = -10

    # node 25: x0 - 0.2 > 0
    W[25, 0] = 1
    B[25] = -10

    # node 26: x5 + 0.1 > 0
    W[26, 5] = 1
    B[26] = 10

    # node 27: -x0 - 0.2 > 0
    W[27, 0] = -1
    B[27] = -10

    # node 28: -x0 - 0.2 > 0
    W[28, 0] = -1
    B[28] = -10

    # node 29: -x0 - 0.2 > 0
    W[29, 0] = 1
    B[29] = 10

    # node 30: x5 + 0.1 > 0
    W[30, 5] = 1
    B[30] = 10

    # 2. Define leaf nodes [T] [F] [Action]
    leaf_base = [0] * dim_out
    l0 = [[0, 1, 3, 7, 15], [], leaf_base.copy()]
    l0[-1][0] = 1
    l1 = [[0, 1, 3, 7], [15], leaf_base.copy()]
    l1[-1][1] = 1
    l2 = [[0, 1, 3, 16], [7], leaf_base.copy()]
    l2[-1][2] = 1
    l3 = [[0, 1, 3], [7, 16],  leaf_base.copy()]
    l3[-1][3] = 1
    l4 = [[0, 1, 8, 17], [3], leaf_base.copy()]
    l4[-1][4] = 1
    l5 = [[0, 1, 8], [3, 17], leaf_base.copy()]
    l5[-1][5] = 1
    l6 = [[0, 1, 18], [3, 8], leaf_base.copy()]
    l6[-1][6] = 1
    l7 = [[0, 1], [3, 8, 18], leaf_base.copy()]
    l7[-1][7] = 1
    l8 = [[0, 4, 9, 19], [1], leaf_base.copy()]
    l8[-1][8] = 1
    l9 = [[0, 4, 9], [1, 19], leaf_base.copy()]
    l9[-1][9] = 1

    l10 = [[0, 4, 20], [1, 9], leaf_base.copy()]
    l10[-1][0] = 1
    l11 = [[0, 4], [1, 9, 20], leaf_base.copy()]
    l11[-1][1] = 1
    l12 = [[0, 10, 21], [1, 4], leaf_base.copy()]
    l12[-1][2] = 1
    l13 = [[0, 10], [1, 4, 21], leaf_base.copy()]
    l13[-1][3] = 1
    l14 = [[0, 22], [1, 4, 10], leaf_base.copy()]
    l14[-1][4] = 1
    l15 = [[0], [1, 4, 10, 22], leaf_base.copy()]
    l15[-1][5] = 1
    l16 = [[2, 5, 11, 23], [0], leaf_base.copy()]
    l16[-1][6] = 1
    l17 = [[2, 5, 11], [0, 23], leaf_base.copy()]
    l17[-1][7] = 1
    l18 = [[2, 5, 24], [0, 11], leaf_base.copy()]
    l18[-1][8] = 1
    l19 = [[2, 5], [0, 11, 24], leaf_base.copy()]
    l19[-1][9] = 1

    l20 = [[2, 12, 25], [0, 5], leaf_base.copy()]
    l20[-1][0] = 1
    l21 = [[2, 12], [0, 5, 25], leaf_base.copy()]
    l21[-1][1] = 1
    l22 = [[2, 26], [0, 5, 12], leaf_base.copy()]
    l22[-1][2] = 1
    l23 = [[2], [0, 5, 12, 26], leaf_base.copy()]
    l23[-1][3] = 1
    l24 = [[6, 13, 27], [0, 2], leaf_base.copy()]
    l24[-1][4] = 1
    l25 = [[6, 13], [0, 2, 27], leaf_base.copy()]
    l25[-1][5] = 1
    l26 = [[6, 28], [0, 2, 13], leaf_base.copy()]
    l26[-1][6] = 1
    l27 = [[6], [0, 2, 13, 28], leaf_base.copy()]
    l27[-1][7] = 1
    l28 = [[14, 29], [0, 2, 6], leaf_base.copy()]
    l28[-1][8] = 1
    l29 = [[14], [0, 2, 6, 29], leaf_base.copy()]
    l29[-1][9] = 1

    l30 = [[30], [0, 2, 6, 14], leaf_base.copy()]
    l30[-1][8] = 1
    l31 = [[], [0, 2, 6, 14, 30], leaf_base.copy()]
    l31[-1][9] = 1 # No Move

    init_leaves = [
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15,
        l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29,
        l30, l31
    ]


    return dim_in, dim_out, W, B, init_leaves
