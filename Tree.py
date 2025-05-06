import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, lambdify, Add
import numpy as np

class Tree:
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

    def __init__(self, root):
        self.root = root

    def evaluate(self):
        return self.root.evaluate()


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


n = 2  # Number of variables

xs = symbols(f'x1:{n+1}')
print(xs)

mul = np.prod(xs)
add = sum(xs)
sub = xs[0] - Add(*xs[1:])

func_mul = lambdify(xs, mul, modules='numpy')
func_add = lambdify(xs, add, modules='numpy')
func_sub = lambdify(xs, sub, modules='numpy')

root = Tree.Node(func_mul, ['x1', 'x2'], label="x1 * x2")
node2 = Tree.Node(func_add, ['x1', 'x2'], label="x1 + x2")
node3 = Tree.Node(func_sub, ['x1', 'x2'], label="x1 - x2")
node4 = Tree.Node(func_mul, ['x1', 'x2'], label="x1 * x2")
node5 = Tree.Node(func_add, ['x1', 'x2'], label="x1 + x2")
node6 = Tree.Node(func_sub, ['x1', 'x2'], label="x1 - x2")
node7 = Tree.Node(func_mul, ['x1', 'x2'], label="x1 * x2")


leaf1 = Tree.Leaf(5)
leaf2 = Tree.Leaf(3)
leaf3 = Tree.Leaf(2)
leaf4 = Tree.Leaf(1)
leaf5 = Tree.Leaf(4)
leaf6 = Tree.Leaf(6)
leaf7 = Tree.Leaf(7)
leaf8 = Tree.Leaf(8)

root.add_child(node2)
root.add_child(node3)
node2.add_child(node4)
node2.add_child(node5)
node3.add_child(node6)
node3.add_child(node7)

node4.add_child(leaf1)
node4.add_child(leaf2)
node5.add_child(leaf3)
node5.add_child(leaf4)
node6.add_child(leaf5)
node6.add_child(leaf6)
node7.add_child(leaf7)
node7.add_child(leaf8)
tree = Tree(root)

# Evaluate and plot
print(f"Tree evaluation result: {tree.evaluate()}")
tree.plot()
