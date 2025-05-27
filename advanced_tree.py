def evaluate_tree_rotation(node, m, field, poly_expr, variables, memory, reg_names=("r1", "r2", "r3")):
    """
    Evaluate a binary tree using rotation trick and at most 3 fixed registers.
    Each node's function is applied via the polynomial masking trick.
    """
    R1, R2, R3 = reg_names

    # Leaf: encode tau + x into a register
    if isinstance(node, Tree.Leaf):
        tau = field(np.random.randint(0, field.order))
        noisy = tau + field(node.value)
        memory.update(R1, noisy)
        return R1, tau, field(node.value)  # Return register, tau, and x

    # Step 1: Recursively evaluate left and right child
    reg_left, tau_left, x_left = evaluate_tree_rotation(node.children[0], m, field, poly_expr, variables, memory, reg_names)
    memory.update(R1, memory.load(reg_left))

    reg_right, tau_right, x_right = evaluate_tree_rotation(node.children[1], m, field, poly_expr, variables, memory, reg_names)
    memory.update(R2, memory.load(reg_right))

    # Step 2: Use the rotation trick to recover f(x1, x2)
    def poly_func(vals):
        return poly_expr.subs({variables[i]: vals[i] for i in range(len(variables))})

    omega = None
    for g in field.elements[1:]:
        if pow(g, m) == 1 and all(pow(g, i) != 1 for i in range(1, m)):
            omega = g
            break
    assert omega is not None, "No primitive root of unity found"

    total = field(0)
    for j in range(m):
        omega_j = omega ** j
        rot1 = omega_j * (tau_left + x_left)
        rot2 = omega_j * (tau_right + x_right)
        y = poly_func([rot1, rot2])
        total += field(int(y) % field.order)

    result = total / field(m)
    memory.update(R3, result)
    return R3, field(0), result  # New value becomes x for parent
