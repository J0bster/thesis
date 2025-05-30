from Tree import Tree
from sympy import symbols, lambdify, Add
import numpy as np

def test_tree_evaluation_and_plot():
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

    tree = Tree(root)

    # Test the plot function
    print(f"Tree evaluation result: {tree.evaluate()}")
    
    # try:
    #     tree.plot()
    # except Exception as e:
    #     assert False, f"Plotting raised an exception: {e}"
    
    assert tree.evaluate() == -1044, "Tree evaluation did not return the expected result."
        
if __name__ == "__main__":
    test_tree_evaluation_and_plot()
    print("All tests passed!")