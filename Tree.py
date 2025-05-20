import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, lambdify, Add
import numpy as np

class Tree:
    def __init__(self, root):
        self.root = root

    def evaluate(self):
        return self.root.evaluate()

    class Node:
        def __init__(self, func, args, label=None):
            self.func = func
            self.args = args
            self.label = label or "func"
            self.children = []

        def add_child(self, child_node):
            self.children.append(child_node)

        def evaluate(self):
            child_values = [child.evaluate() for child in self.children]
            return self.func(*child_values)

    class Leaf:
        def __init__(self, value):
            self.value = value

        def evaluate(self):
            return self.value



    def plot(self):
        G = nx.DiGraph()
        labels = {}

        def add_node(node, parent=None):
            if isinstance(node, self.Leaf):
                node_id = f"Leaf:{node.value}"
                G.add_node(node_id)
                labels[node_id] = str(node.value)
            else:
                node_id = f"Node:{id(node)}"
                G.add_node(node_id)
                labels[node_id] = node.label

            if parent is not None:
                G.add_edge(parent, node_id)

            if not isinstance(node, self.Leaf):
                for child in node.children:
                    add_node(child, node_id)

        add_node(self.root)

        def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
            pos = {}
            def _hierarchy_pos(G, root, left, right, vert_loc, xcenter, pos, parent=None):
                pos[root] = (xcenter, vert_loc)
                children = list(G.successors(root))
                if len(children) != 0:
                    dx = (right - left) / len(children)
                    nextx = left + dx / 2
                    for child in children:
                        pos = _hierarchy_pos(G, child, nextx - dx / 2, nextx + dx / 2,
                                             vert_loc - vert_gap, nextx, pos, root)
                        nextx += dx
                return pos
            return _hierarchy_pos(G, root, 0, width, vert_loc, xcenter, pos)

        root_id = [n for n in G.nodes if G.in_degree(n) == 0][0]
        pos = hierarchy_pos(G, root=root_id)
        nx.draw(G, pos, with_labels=True, labels=labels,
                node_size=2000, node_color='skyblue',
                font_size=10, font_weight='bold')
        plt.show()