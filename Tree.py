import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, lambdify, Add
import numpy as np
import galois
from memory_tracker import MemoryTracker

class Tree:
    def __init__(self, root):
        self.root = root

    def evaluate(self):
        return self.root.evaluate()
    
    def evaluate_with_memory(self, m):
        p = 17 # TODO acutally compute value
        bitsize = 5 # TODO actually compute value
        GF = galois.GF(p)
        registers = ['R1', 'R2', 'R3']
        [m.store(label+'_'+str(i), GF(0)) for i in range(bitsize) for label in registers]
        return self.root.evaluate_with_memory(m, registers, bitsize, GF)
    
    def evaluate_with_3_node_limit(self, memory_tracker):
        return self.root.evaluate_with_memory_limit(memory_tracker)

    

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

        def evaluate_with_memory(self, m, registers, bitsize, GF, inverse=False):
            """
            First element in registers is our target
            Second is Rl, Third is Rr
            """
            """for 𝑗 = 1 . . . 𝑚 do
                for 𝑐 ∈ {ℓ, 𝑟 }, 𝑖 = 1 . . . ⌈log 𝑘⌉ do
                    𝑅𝑐,𝑖 ← 𝜔 𝑗 · 𝑅𝑐,𝑖
                𝑃ℓ , 𝑃𝑟
                for 𝑖 = 1 . . . ⌈log 𝑘⌉ do
                    𝑅𝑢,𝑖 ← 𝑅𝑢,𝑖 − 𝑞𝑢,𝑖 (𝑅ℓ , 𝑅𝑟 )
                𝑃 −1ℓ , 𝑃 −1𝑟
                for 𝑐 ∈ {ℓ, 𝑟 }, 𝑖 = 1 . . . ⌈log 𝑘⌉ do
                    𝑅𝑐,𝑖 ← 𝜔 − 𝑗 · 𝑅𝑐,𝑖
            """
            # TODO base case without any children.

            orderofrootofunity = 5 # TODO
            rootofunity= GF(1) # TODO
            for j in range(1, orderofrootofunity + 1): # todo memory track all the loop variables
                for c in registers[1:]:
                    for i in range(bitsize):
                        curlabel = c + '_' + str(i)
                        m.update(curlabel, (rootofunity ** j) * m.load(curlabel))
                assert len(self.children) == 2
                self.children[0].evaluate_with_memory(m, registers[1:] + registers[:1]) 
                self.children[1].evaluate_with_memory(m, registers[2:] + registers[:2]) 
                for i in range(bitsize):
                    curlabel = registers[0] + '_' + str(i)
                    labell = registers[1] + '_' + str(i)
                    labelr = registers[2] + '_' + str(i)
                    if not inverse:
                        m.update(curlabel, m.load(curlabel) - self.func(m.load(labell), m.load(labelr)))
                    else:
                        m.update(curlabel, m.load(curlabel) + self.func(m.load(labell), m.load(labelr)))
                self.children[0].evaluate_with_memory(m, registers[1:] + registers[:1], inverse=True) 
                self.children[1].evaluate_with_memory(m, registers[2:] + registers[:2], inverse=True) 
                
                for c in registers[1:]:
                    for i in range(bitsize):
                        curlabel = c + '_' + str(i)
                        m.update(curlabel, (rootofunity ** -j) * m.load(curlabel))
            ## TODO actually compute result from the registers
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
                # If the value is a constant, store it in memory with a unique label
                label = f"Leaf:{self.value}"
                if label not in m.variable_storage:
                    m.store(label, self.value)
                return m.load(label)



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