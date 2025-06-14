import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, lambdify, Add
import numpy as np
import galois
import math
from memory_tracker import MemoryTracker


class Tree:
    recursiveCallCount = 0

    def __init__(self, root):
        self.root = root
        
    def _max_leaf_value(self):
        def find_max(node):
            if isinstance(node, self.Leaf):
                return int(node.value)
            return max(find_max(child) for child in node.children)
        return find_max(self.root)


    def evaluate(self):
        return self.root.evaluate()
    
    def evaluate_with_memory(self, m):
        return self.root.evaluate_with_memory(m)
    
    def evaluate_with_3_node_limit(self, m):
        p = 2**4
        max_val = self._max_leaf_value()
        bitsize = 3
        print(f"Bitsize for GF: {bitsize} (max_val={max_val})")
        GF = galois.GF(p)
        registers = ['R0', 'Rl', 'Rr']
        if isinstance(self.root, self.Leaf):
            return self.root.evaluate_with_memory(m)
        for i in range(bitsize):
            for label in registers:
                m.store(f'{label}_{i}', GF(0))
        return self.root.evaluate_with_memory_limit(m, registers, bitsize, GF)

    class Node:
        def __init__(self, func, args, label=None, isroot=False):
            self.func = func
            self.args = args
            self.label = label or "func"
            self.children = []
            self.isroot = isroot

        def add_child(self, child_node):
            self.children.append(child_node)

        def evaluate(self):
            child_values = [child.evaluate() for child in self.children]
            return self.func(*child_values)

        def evaluate_with_memory(self, m):
            child_values = [child.evaluate_with_memory(m) for child in self.children]
            arg_values = {arg: value for arg, value in zip(self.args, child_values)}
            result = self.func(**arg_values)
            if self.label not in m.variable_storage:
                m.store(self.label, result)
            else:
                m.update(self.label, result)
            return result
                        
        def evaluate_with_memory_limit(self, m, registers, bitsize, GF, inverse=False):
            Tree.recursiveCallCount += 1

            # If both children are leaves, process their bits
            if all(isinstance(child, Tree.Leaf) for child in self.children):
                left_val = int(self.children[0].value)
                right_val = int(self.children[1].value)
                for i in range(bitsize):
                    left_bit = (left_val >> i) & 1
                    right_bit = (right_val >> i) & 1
                    bit_val = self.func(left_bit, right_bit) % GF.characteristic
                    reg = f"{registers[0]}_{i}"
                    cur_val = m.load(reg)
                    delta = GF(bit_val) if not inverse else -GF(bit_val)
                    m.update(reg, cur_val + delta)
                return

            n = bitsize - 1
            order = GF.order - 1
            omega = GF.primitive_root_of_unity(order)
            assert pow(omega, order) == 1, "omega must be a primitive root of unity"

            for j in range(1, order + 1):
                # Rotate child registers
                for c in registers[1:]:
                    for i in range(bitsize):
                        reg = f"{c}_{i}"
                        m.update(reg, omega**j * m.load(reg))

                # Evaluate children once each
                self.children[0].evaluate_with_memory_limit(m, registers[1:] + registers[:1], bitsize, GF)
                self.children[1].evaluate_with_memory_limit(m, registers[2:] + registers[:2], bitsize, GF)

                # Update parent register with polynomial func
                for i in range(bitsize):
                    Rl = m.load(f"{registers[1]}_{i}")
                    Rr = m.load(f"{registers[2]}_{i}")
                    output_reg = f"{registers[0]}_{i}"
                    current_val = m.load(output_reg)
                    poly_val = self.func(Rl, Rr)
                    m.update(output_reg, current_val - poly_val)

                # Undo child
                self.children[0].evaluate_with_memory_limit(m, registers[1:] + registers[:1], bitsize, GF, inverse=True)
                self.children[1].evaluate_with_memory_limit(m, registers[2:] + registers[:2], bitsize, GF, inverse=True)

                # Undo rotation
                for c in registers[1:]:
                    for i in range(bitsize):
                        reg = f"{c}_{i}"
                        m.update(reg, omega**(-j) * m.load(reg))

            return m.load(f"{registers[0]}_0")

    class Leaf:
        def __init__(self, value, label=None):
            self.value = value
            self.label = label if label is not None else str(value)

        def evaluate(self):
            return self.value

        # Bitwise update for leaves (GF and int supported)
        def evaluate_with_memory_limit(self, m, registers, bitsize, GF, inverse=False):
            reg_prefix = registers[0]
            # If value is a label, load from memory; else, use value
            if isinstance(self.value, str):
                val = m.load(self.value)
            else:
                val = GF(self.value)
                val = int(val)
            for i in range(bitsize):
                bit_val = (val >> i) & 1
                reg = f"{reg_prefix}_{i}"
                prev = m.load(reg)
                delta = GF(bit_val) if not inverse else -GF(bit_val)
                m.update(reg, prev + delta)
                print(f"[Leaf] {self.value} {'+' if not inverse else '-'} {bit_val} -> {reg} = {prev + delta}")
            return val

        # Alternative version: update whole value (not bitwise)
        def evaluate_with_memory_limit(self, m, registers, bitsize, GF, inverse=False):
            reg_prefix = registers[0]
            if isinstance(self.value, str):
                val = m.load(self.value)
            else:
                val = GF(self.value)
            for i in range(bitsize):
                reg = f"{reg_prefix}_{i}"
                prev = m.load(reg)
                delta = val if not inverse else -val
                m.update(reg, prev + delta)
                print(f"[Leaf] {self.value} {'+' if not inverse else '-'} {val} -> {reg} = {prev + delta}")
            return val

    def plot(self):
        G = nx.DiGraph()
        labels = {}

        def add_node(node, parent=None):
            node_id = f"Leaf:{node.label}" if isinstance(node, self.Leaf) else f"Node:{id(node)}"
            G.add_node(node_id)
            labels[node_id] = str(node.value) if isinstance(node, self.Leaf) else node.label
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
                if children:
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


def test_tree_evaluate_with_3_node_limit():
    x, y = symbols('x y')
    xor_poly = x + y - 2*x*y
    func_xor = lambdify([x, y], xor_poly, modules='numpy')

    # Build tree using only XOR nodes
    root = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2", isroot=True)
    # Example: add more nodes/leaves for a larger tree
    # node2 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    # node3 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    # leaf1 = Tree.Leaf(5)
    # leaf2 = Tree.Leaf(3)
    # leaf3 = Tree.Leaf(2)
    # leaf4 = Tree.Leaf(1)

    # root.add_child(node2)
    # root.add_child(node3)
    # node2.add_child(leaf1)
    # node2.add_child(leaf2)
    # node3.add_child(leaf3)
    # node3.add_child(leaf4)
    
    # More complex tree example
    node2 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    node3 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    node4 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    node5 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    node6 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")
    node7 = Tree.Node(func_xor, ['x1', 'x2'], label="x1 XOR x2")

    leaf1 = Tree.Leaf(0b01, label="leaf1")
    leaf2 = Tree.Leaf(0b10, label="leaf2")
    leaf3 = Tree.Leaf(0b01, label="leaf3")
    leaf4 = Tree.Leaf(0b11, label="leaf4")
    leaf5 = Tree.Leaf(0b00, label="leaf5")
    leaf6 = Tree.Leaf(0b01, label="leaf6")
    leaf7 = Tree.Leaf(0b10, label="leaf7")
    leaf8 = Tree.Leaf(0b01, label="leaf8")

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

    m = MemoryTracker()

    print("Evaluating full tree with memory-limited protocol:")
    result = tree.evaluate_with_3_node_limit(m)
    # result_actual = tree.evaluate_with_memory(m)
    print
    print(f"Tree evaluation result: {result}")
    print("Memory dump:", m.dump())
    print("Memory summary:", m.summary())
    
    num_leaves = 8
    log_n = math.ceil(math.log2(num_leaves))
    loglog_n = math.ceil(math.log2(log_n))
    theoretical_bound = log_n * loglog_n

    summary = m.summary()
    print(f"max_bits used: {summary['max_bits']}")
    print(f"Theoretical O(log n * log log n): {theoretical_bound}")
    
    print(f"Recursive call count: {Tree.recursiveCallCount}")

    try:
        tree.plot()
    except Exception as e:
        assert False, f"Plotting raised an exception: {e}"

if __name__ == "__main__":
    test_tree_evaluate_with_3_node_limit()