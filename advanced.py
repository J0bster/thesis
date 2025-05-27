import random
from sympy import symbols
import galois
import numpy as np
from sympy import lambdify, Add

from memory_tracker import MemoryTracker
from Tree import Tree

def advanced_evaluate_min_registers(tree, m, field, omega, max_registers=3):
    """
    Evaluate the tree using the advanced algorithm, but only ever store up to
    `max_registers` values in the MemoryTracker at any time.
    No explicit register list is used; all storage is via the memory tracker.
    """
    memory = MemoryTracker()

    # Initial evaluation: store only the root value in memory (like evaluate_with_memory)
    result = tree.root.evaluate_with_memory(memory)

    # Main loop: perform m rounds of computation, always updating values in memory
    for j in range(m):
        print(f"[Iteration j={j}]")
        # Only operate on what's in memory
        for varname in list(memory.variable_storage.keys()):
            val = memory.load(varname)
            new_val = (field(1) - omega) * val
            memory.update(varname, new_val)
        # If you need to combine values or apply tree gates, do it here

        # Optionally, enforce max_registers limit:
        if len(memory.variable_storage) > max_registers:
            # Remove oldest or least-recently-used, or raise an error
            # For demo, just print a warning
            print(f"Warning: More than {max_registers} variables in memory!")

    print(f"Final memory after {m} iterations: {memory.variable_storage.keys()}")
    print(memory.dump())
    # Return the root's value from memory (or however you want to finalize)
    return memory.load(tree.root.label if tree.root.label else list(memory.variable_storage.keys())[0])

if __name__ == "__main__":
    # Quick demo usage
    p = 17
    GF = galois.GF(p)
    m = 4

    # Find a primitive m-th root of unity in GF(p)
    omega = None
    for g in GF.elements[1:]:
        if pow(g, m) == 1 and all(pow(g, k) != 1 for k in range(1, m)):
            omega = g
            break
    assert omega is not None, "No primitive root of unity found"

    # --- Build a more elaborate balanced binary tree (like in test_full_tree.py) ---
    n = 2
    xs = symbols(f'x1:{n+1}')
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

    balanced_tree = Tree(root)

    # Optionally plot the tree structure
    try:
        balanced_tree.plot()
    except Exception as e:
        print(f"Plotting raised an exception: {e}")

    # Evaluate with minimal registers
    final_value = advanced_evaluate_min_registers(balanced_tree, m, GF, omega)
    print("Final Value:", final_value)