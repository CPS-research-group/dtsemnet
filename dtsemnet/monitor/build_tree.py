'''Visualize and Build Tree from DTNet'''

import numpy as np
import pydot

# make them equal size null with -1
class Node:
    '''Defines a node in the tree'''
    def __init__(self, node_num):
        self.left = None
        self.right = None
        self.node_num = node_num
        self.w = None
        self.c = None
        self.is_leaf = False
        self.action = None

    def __repr__(self):
        if self.left != None:
            print_left = self.left.node_num
        else:
            print_left = None

        if self.right != None:
            print_right = self.right.node_num
        else:
            print_right = None

        if self.is_leaf:
            print_node = self.action
        else:
            print_node = self.node_num
        return f"Left: {print_left}, Right: {print_right}, Node: {print_node}"


def build_tree(dt_agent):
    '''Builds a tree from the policy agent'''
    Tree = dict()
    leaf_init_information = dt_agent.init_leaves
    # insert leaf
    for ln in range(len(leaf_init_information)):
        left = leaf_init_information[ln][0]
        right = leaf_init_information[ln][1]
        req_len = max(len(left), len(right))
        total_len = len(left) + len(right)
        l = np.pad(np.sort(left)[::-1], (0, req_len - len(left) + 1),
                constant_values=-1)
        r = np.pad(np.sort(right)[::-1], (0, req_len - len(right) + 1),
                constant_values=-1)

        i = 0
        prev_node = None
        while i < total_len:
            if l[0] > r[0]:
                # pop the first element
                val = l[0]
                l = np.delete(l, 0)

                if i == 0:
                    #create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)
                    leaf_node = Node(None)
                    leaf_node.action = leaf_init_information[ln][-1]
                    Tree[val].left = leaf_node
                    leaf_node.is_leaf = True

                    # assign leaf to left of that node
                    prev_node = val

                else:
                    # check if node exists else create a new node

                    if val not in Tree.keys():
                        Tree[val] = Node(val)

                    Tree[val].left = Tree[prev_node]
                    # assign leaf to left of that node
                    prev_node = val

            else:
                # pop the first element
                val = r[0]
                r = np.delete(r, 0)

                if i == 0:
                    #create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)
                    leaf_node = Node(None)  # create a leaf node
                    leaf_node.action = leaf_init_information[ln][-1]
                    Tree[val].right = leaf_node

                    leaf_node.is_leaf = True
                    # assign leaf to right of that node
                    prev_node = val
                else:
                    # check if node exists else create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)

                    Tree[val].right = Tree[prev_node]
                    # assign leaf to left of that node
                    prev_node = val
            i += 1
    return Tree


# Inorder traversal of tree
def print_binary_tree(node):
    '''Prints the tree in inorder traversal'''
    if node == None:
        return
    if node.is_leaf:
        print(node.action)
        return

    print_binary_tree(node.left)

    print(node.node_num)

    print_binary_tree(node.right)


# Define a function to create a pydot graph of the binary tree
def get_equation(node_weight, node_bias):
    eqn = ''
    for i, wt in enumerate(node_weight):
        if wt != 0:
            eqn += f"{str(round(wt, 1))} * x{i} + "
        if i % 2 == 0:
            eqn += f" \n "
    eqn += f"{str(round(node_bias, 1))} > 0"
    return eqn

def get_actionname(action):
    if action == 0:
        return f'Do \n Nothing'
    elif action == 1:
        return f'Fire \n Left'
    elif action == 2:
        return f'Fire \n Main'
    elif action == 3:
        return f'Fire \n Right'

def viz_tree(node, node_weights, node_biases, graph=None):
    '''Creates a pydot graph of the binary tree
    tree_graph = viz_tree(Tree[0])
    tree_graph.write_png('dtnet_tree_lunar.png')
    '''
    col = ['red', 'green', 'skyblue', 'yellow']
    if graph is None:
        graph_attributes = {
                    'graph_type': 'graph',
                    'nodesep': '0.1',
                    'ranksep': '0.2',
                    'splines': 'true',  # Use orthogonal edge routing
                    'overlap': 'false',
                    'ratio': '0.2',
                    'margin': '0',
                    'pad': '0',
                    'compound': 'true',
                    'newrank': 'true',
                }
        graph = pydot.Dot(**graph_attributes)
    
    
    if node.node_num == 0:
        root_node = pydot.Node(str(0),
                            label=get_equation(node_weights[0], node_biases[0]),
                            fontsize=14,
                            style='filled',
                            fillcolor='lightblue',
                            shape='rectangle',
                            fontname='Arial-Bold', 
                            width=2.5,
                            height=1.4,
                            margin='0',
                            fixedsize=True)
        graph.add_node(root_node)


    if node.left is not None:
        if node.left.node_num is None:
            use = f"A{np.argmax(node.left.action)}_N{node.node_num}"
            left_node = pydot.Node(str(use),
                                   label = f'{np.argmax(node.left.action)}',
                                fontsize=16,
                                    style='filled',
                                    fillcolor= 'orange', #col[np.argmax(node.left.action)],
                                    shape='rectangle',
                                    fontname='Arial-Bold', 
                                    width=1,
                                    height=1,
                                    margin='0',
                                    fixedsize=True)
        else:
            use = f"{node.left.node_num}"
            left_node = pydot.Node(str(use),
                                label=get_equation(node_weights[node.left.node_num], node_biases[node.left.node_num]),
                                fontsize=14,
                            style='filled',
                            fillcolor='lightblue',
                            shape='rectangle',
                            fontname='Arial-Bold', 
                            width=2.5,
                            height=1.4,
                            margin='0',
                            fixedsize=True)
        graph.add_node(left_node)
        if node.node_num == 0:
            graph.add_edge(pydot.Edge(str(node.node_num), str(use), label='True', fontsize=20, fontname='Arial-Bold'))
        else:
            graph.add_edge(pydot.Edge(str(node.node_num), str(use)))
        viz_tree(node.left,node_weights=node_weights, node_biases=node_biases ,graph=graph)

    if node.right is not None:
        if node.right.node_num is None:
            use = f"A{np.argmax(node.right.action)}_N{node.node_num}"
            right_node = pydot.Node(str(use),
                                    label = f'{np.argmax(node.right.action)}', 
                                    fontsize=16,
                                    style='filled',
                                    fillcolor= 'orange', #np.argmax(node.right.action)],
                                    shape='rectangle',
                                    fontname='Arial-Bold', 
                                    width=1,
                                    height=1,
                                    margin='0',
                                    fixedsize=True)
        else:
            use = f"{node.right.node_num}"
            right_node = pydot.Node(str(use),
                                    label=get_equation(node_weights[node.right.node_num], node_biases[node.right.node_num]),
                                    fontsize=14,
                                    style='filled',
                                    fillcolor='lightblue',
                                    shape='rectangle',
                                    fontname='Arial-Bold', 
                                    width=2.5,
                                    height=1.4,
                                    margin='0',
                                    fixedsize=True)
        graph.add_node(right_node)
        if node.node_num == 0:
            graph.add_edge(pydot.Edge(str(node.node_num), str(use), label='False', fontsize=20, fontname='Arial-Bold'))
        else:
            graph.add_edge(pydot.Edge(str(node.node_num), str(use)))
        viz_tree(node.right, node_weights=node_weights, node_biases=node_biases, graph=graph)

    return graph




# Simple approach
# Loop over all leaf nodes and check for their required conditions
# leaf_init_information is enough to get path and leaf node


# def prolo_dt_agent_action(state_input, policy_agent):
#     agent = policy_agent.action_network
#     action_prob = agent.action_probs.detach().cpu().numpy()

#     def check_condition(ix, condition="T"):
#         "Check if alpha(wx-b) >= 0 is True or False"
#         # convert to numpy
#         # alpha = agent.alpha.detach().numpy()
#         w = agent.layers[ix].detach().cpu().numpy()
#         b = agent.comparators[ix].detach().cpu().numpy()
#         # print('node value',np.dot(w, state_input) - b)
#         if (np.dot(w, state_input) - b) > 0:
#             if condition == "T":
#                 return True
#             else:
#                 return False

#         else:
#             if condition == "T":
#                 return False
#             else:
#                 return True

#     action = None
#     for ix, leaf in enumerate(agent.leaf_init_information):
#         true_conditions = [
#             check_condition(ix, condition="T") for ix in leaf[0]
#         ]
#         false_conditions = [
#             check_condition(ix, condition="F") for ix in leaf[1]
#         ]  # All False conditions must be False
#         # print(true_conditions, false_conditions)
#         if all(true_conditions) and all(false_conditions):
#             action = (np.argmax(action_prob[ix]))
#             #action = (np.argmax(leaf[-1]))
#             # print("action", action)

#     return action

def prolo_build_tree(agent):
    Tree = dict()
    leaf_init_information = agent.leaf_init_information
    # insert leaf
    for ln in range(len(leaf_init_information)):
        left = leaf_init_information[ln][0]
        right = leaf_init_information[ln][1]
        req_len = max(len(left), len(right))
        total_len = len(left) + len(right)
        l = np.pad(np.sort(left)[::-1], (0, req_len - len(left) + 1),
                constant_values=-1)
        r = np.pad(np.sort(right)[::-1], (0, req_len - len(right) + 1),
                constant_values=-1)

        i = 0
        prev_node = None
        while i < total_len:
            if l[0] > r[0]:
                # pop the first element
                val = l[0]
                l = np.delete(l, 0)

                if i == 0:
                    #create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)
                    leaf_node = Node(None)
                    leaf_node.action = leaf_init_information[ln][-1]
                    Tree[val].left = leaf_node
                    leaf_node.is_leaf = True

                    # assign leaf to left of that node
                    prev_node = val

                else:
                    # check if node exists else create a new node

                    if val not in Tree.keys():
                        Tree[val] = Node(val)

                    Tree[val].left = Tree[prev_node]
                    # assign leaf to left of that node
                    prev_node = val

            else:
                # pop the first element
                val = r[0]
                r = np.delete(r, 0)

                if i == 0:
                    #create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)
                    leaf_node = Node(None)  # create a leaf node
                    leaf_node.action = leaf_init_information[ln][-1]
                    Tree[val].right = leaf_node

                    leaf_node.is_leaf = True
                    # assign leaf to right of that node
                    prev_node = val
                else:
                    # check if node exists else create a new node
                    if val not in Tree.keys():
                        Tree[val] = Node(val)

                    Tree[val].right = Tree[prev_node]
                    # assign leaf to left of that node
                    prev_node = val
            i += 1

    return Tree