from Tree import Tree

# Test to check if a tree can be initialized with a root node
def test_tree_initialization():
    root = Tree.Leaf(10)
    tree = Tree(root)
    assert tree.root == root
    assert tree.evaluate() == 10

# Test to check if a node can evaluate its children
def test_node_evaluation():
    add_func = lambda x, y: x + y
    root = Tree.Node(add_func, ['x1', 'x2'], label="x1 + x2")
    child1 = Tree.Leaf(5)
    child2 = Tree.Leaf(3)
    root.add_child(child1)
    root.add_child(child2)
    tree = Tree(root)
    assert tree.evaluate() == 8

# Test to check if a tree with multiple levels evaluates correctly
def test_tree_evaluation():
    mul_func = lambda x, y: x * y
    add_func = lambda x, y: x + y

    root = Tree.Node(mul_func, ['x1', 'x2'], label="x1 * x2")
    node1 = Tree.Node(add_func, ['x1', 'x2'], label="x1 + x2")
    node2 = Tree.Node(add_func, ['x1', 'x2'], label="x1 + x2")
    leaf1 = Tree.Leaf(2)
    leaf2 = Tree.Leaf(3)
    leaf3 = Tree.Leaf(4)
    leaf4 = Tree.Leaf(5)

    root.add_child(node1)
    root.add_child(node2)
    node1.add_child(leaf1)
    node1.add_child(leaf2)
    node2.add_child(leaf3)
    node2.add_child(leaf4)

    tree = Tree(root)
    # (2 + 3) * (4 + 5) = 5 * 9 = 45
    assert tree.evaluate() == 45

# Test to check if a leaf node evaluates to its value
def test_leaf_evaluation():
    leaf = Tree.Leaf(7)
    assert leaf.evaluate() == 7

# Test to check if the plot function runs without errors
def test_tree_plot():
    root = Tree.Leaf(1)
    tree = Tree(root)
    try:
        tree.plot()
    except Exception as e:
        assert False, f"Plotting raised an exception: {e}"