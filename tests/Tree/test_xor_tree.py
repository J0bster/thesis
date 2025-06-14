import pytest
from sympy import symbols, lambdify
from Tree import Tree
from memory_tracker import MemoryTracker

@pytest.fixture
def xor_func():
    x, y = symbols('x y')
    xor_poly = x + y - 2 * x * y
    return lambdify([x, y], xor_poly, modules='numpy')

def expected_xor(*values):
    from functools import reduce
    return reduce(lambda x, y: x ^ y, values)

def run_xor_tree(tree, bitsize=2):
    m = MemoryTracker()
    tree.evaluate_with_3_node_limit(m)
    bits = [int(m.load(f"R0_{i}")) for i in range(bitsize)]
    value = sum((b << i) for i, b in enumerate(bits))
    return value

def test_xor_tree_2_leaves(xor_func):
    leaf1 = Tree.Leaf(1)
    leaf2 = Tree.Leaf(0)
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(leaf1)
    root.add_child(leaf2)
    tree = Tree(root)
    assert run_xor_tree(tree, bitsize=1) == 1

def test_xor_tree_4_leaves(xor_func):
    a, b, c, d = 1, 2, 3, 0
    n1 = Tree.Node(xor_func, ['x1', 'x2'])
    n1.add_child(Tree.Leaf(a))
    n1.add_child(Tree.Leaf(b))
    n2 = Tree.Node(xor_func, ['x1', 'x2'])
    n2.add_child(Tree.Leaf(c))
    n2.add_child(Tree.Leaf(d))
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(n1)
    root.add_child(n2)
    tree = Tree(root)
    assert run_xor_tree(tree, bitsize=2) == expected_xor(a, b, c, d)


def test_xor_tree_8_leaves(xor_func):
    vals = [0b01, 0b10, 0b01, 0b11, 0b00, 0b01, 0b10, 0b01]
    n1 = Tree.Node(xor_func, ['x1', 'x2']); n1.add_child(Tree.Leaf(vals[0])); n1.add_child(Tree.Leaf(vals[1]))
    n2 = Tree.Node(xor_func, ['x1', 'x2']); n2.add_child(Tree.Leaf(vals[2])); n2.add_child(Tree.Leaf(vals[3]))
    n3 = Tree.Node(xor_func, ['x1', 'x2']); n3.add_child(Tree.Leaf(vals[4])); n3.add_child(Tree.Leaf(vals[5]))
    n4 = Tree.Node(xor_func, ['x1', 'x2']); n4.add_child(Tree.Leaf(vals[6])); n4.add_child(Tree.Leaf(vals[7]))
    n5 = Tree.Node(xor_func, ['x1', 'x2']); n5.add_child(n1); n5.add_child(n2)
    n6 = Tree.Node(xor_func, ['x1', 'x2']); n6.add_child(n3); n6.add_child(n4)
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(n5)
    root.add_child(n6)
    tree = Tree(root)
    assert run_xor_tree(tree, bitsize=2) == expected_xor(*vals)


def test_xor_tree_8_leaves_zeros(xor_func):
    vals = [0,0,0,0,0,0,0,0]
    n1 = Tree.Node(xor_func, ['x1', 'x2']); n1.add_child(Tree.Leaf(vals[0])); n1.add_child(Tree.Leaf(vals[1]))
    n2 = Tree.Node(xor_func, ['x1', 'x2']); n2.add_child(Tree.Leaf(vals[2])); n2.add_child(Tree.Leaf(vals[3]))
    n3 = Tree.Node(xor_func, ['x1', 'x2']); n3.add_child(Tree.Leaf(vals[4])); n3.add_child(Tree.Leaf(vals[5]))
    n4 = Tree.Node(xor_func, ['x1', 'x2']); n4.add_child(Tree.Leaf(vals[6])); n4.add_child(Tree.Leaf(vals[7]))
    n5 = Tree.Node(xor_func, ['x1', 'x2']); n5.add_child(n1); n5.add_child(n2)
    n6 = Tree.Node(xor_func, ['x1', 'x2']); n6.add_child(n3); n6.add_child(n4)
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(n5)
    root.add_child(n6)
    tree = Tree(root)
    assert run_xor_tree(tree, bitsize=2) == expected_xor(*vals)
    
def test_xor_tree_4_leaves_3bits(xor_func):
    # 3-bit values: 0b110 (6), 0b101 (5), 0b011 (3), 0b001 (1)
    vals = [0b110, 0b101, 0b011, 0b001]
    # Build tree: ((a ^ b) ^ (c ^ d))
    leaf1, leaf2, leaf3, leaf4 = (Tree.Leaf(v) for v in vals)
    n1 = Tree.Node(xor_func, ['x1', 'x2']); n1.add_child(leaf1); n1.add_child(leaf2)
    n2 = Tree.Node(xor_func, ['x1', 'x2']); n2.add_child(leaf3); n2.add_child(leaf4)
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(n1)
    root.add_child(n2)
    tree = Tree(root)
    expected = expected_xor(*vals)
    assert run_xor_tree(tree, bitsize=3) == expected

def test_xor_tree_8_leaves_3bits(xor_func):
    # 8 leaves: perfect binary tree, height 3
    vals = [0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111, 0b000]
    # Level 1
    leafs = [Tree.Leaf(v) for v in vals]
    n1 = Tree.Node(xor_func, ['x1', 'x2']); n1.add_child(leafs[0]); n1.add_child(leafs[1])
    n2 = Tree.Node(xor_func, ['x1', 'x2']); n2.add_child(leafs[2]); n2.add_child(leafs[3])
    n3 = Tree.Node(xor_func, ['x1', 'x2']); n3.add_child(leafs[4]); n3.add_child(leafs[5])
    n4 = Tree.Node(xor_func, ['x1', 'x2']); n4.add_child(leafs[6]); n4.add_child(leafs[7])
    # Level 2
    n5 = Tree.Node(xor_func, ['x1', 'x2']); n5.add_child(n1); n5.add_child(n2)
    n6 = Tree.Node(xor_func, ['x1', 'x2']); n6.add_child(n3); n6.add_child(n4)
    # Root
    root = Tree.Node(xor_func, ['x1', 'x2'], isroot=True)
    root.add_child(n5)
    root.add_child(n6)
    tree = Tree(root)
    expected = expected_xor(*vals)
    assert run_xor_tree(tree, bitsize=3) == expected
