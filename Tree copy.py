import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, lambdify, Add
import numpy as np
from memory_tracker import MemoryTracker

class Tree:
    def __init__(self, root):
        self.root = root

    def evaluate(self):
        return self.root.evaluate()
    
    def evaluate_with_memory(self, m):
        return self.root.evaluate_with_memory(m)

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

        def evaluate_with_memory(self, m):
            # Evaluate children with memory
            child_values = [child.evaluate_with_memory(m) for child in self.children]
            # Map child values to the expected arguments
            arg_values = {arg: value for arg, value in zip(self.args, child_values)}
            # Compute the result
            result = self.func(**arg_values)
            # Store the result in memory if the node has a label
            if self.label:
                # m.store(self.label, result)
                if self.label not in m.variable_storage:
                    m.store(self.label, 0)
                else:
                    m.update(self.label, result)
            return result

    class Leaf:
        def __init__(self, value):
            self.value = value

        def evaluate(self):
            return self.value

        def evaluate_with_memory(self, m):
            # If the value is a variable name, load it from memory
            if isinstance(self.value, str):
                if self.value not in m.variable_storage:
                    m.store(self.value, 0)  # Default to 0 if not in memory
                return m.load(self.value)
            else:
                # If the value is a constant, return it directly
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
        
        
# from memory_tracker import MemoryTracker
# from sympy import symbols, lambdify

# def test_evaluate_with_memory():
#     # Define a simple tree
#     x1, x2 = symbols('x1 x2')
#     func_add = lambdify([x1, x2], x1 + x2, modules='numpy')

#     # Create the tree structure
#     root = Tree.Node(func_add, ['x1', 'x2'], label="x1 + x2")
#     leaf1 = Tree.Leaf("x1")
#     leaf2 = Tree.Leaf("x2")

#     root.add_child(leaf1)
#     root.add_child(leaf2)

#     # Initialize memory tracker
#     m = MemoryTracker()
#     m.store("x1", 3)
#     m.store("x2", 4)

#     # Create the tree and evaluate with memory
#     tree = Tree(root)
#     result = tree.evaluate_with_memory(m)
#     print(m.dump())

#     # Assertions
#     assert result == 7  # 3 + 4 = 7
#     assert m.load("x1") == 3
#     assert m.load("x2") == 4
#     assert m.load("x1 + x2") == 7

#     print("Test passed!")
    
# if __name__ == "__main__":
#     test_evaluate_with_memory()