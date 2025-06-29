import time
import random
import galois
import math

def get_max_leaf_val(bitsize):
    return 2**bitsize - 1

def show_progress(current, total, prefix="Progress"):
    bar_len = 30
    filled_len = int(bar_len * current // total)
    bar = "#" * filled_len + "-" * (bar_len - filled_len)
    print(f"{prefix}: [{bar}] {current}/{total} trees tested", end="\r")
    if current == total:
        print()

def compute_field(max_bitsize):
    k_max = 2**max_bitsize
    log_k = math.ceil(math.log2(k_max))
    field_order = 2**math.ceil(math.log2(2**log_k + 2))
    GF = galois.GF(field_order)
    return GF

def compute_expected_result(leaf_values, func_name):
    from functools import reduce
    if func_name.lower() == "xor":
        return reduce(lambda x, y: x ^ y, leaf_values)
    elif func_name.lower() == "and":
        return reduce(lambda x, y: x & y, leaf_values)
    else:
        raise ValueError(f"Unknown func_name: {func_name}")

def extract_tree_output(memory, bitsize):
    bits = [int(memory.get(f"R0_{i}", 0)) for i in range(bitsize)]
    return sum((b << i) for i, b in enumerate(bits))

def make_balanced_tree(depth, func, leaf_value_fn):
    if depth == 0:
        return Tree.Leaf(leaf_value_fn())
    left = make_balanced_tree(depth-1, func, leaf_value_fn)
    right = make_balanced_tree(depth-1, func, leaf_value_fn)
    node = Tree.Node(func, ['x1', 'x2'], label=func.__name__)
    node.add_child(left)
    node.add_child(right)
    return node

def speed_test(tree_depths, func, func_name, bitsize, GF, trials=3, random_trees=3):
    results = []
    max_leaf_val = get_max_leaf_val(bitsize)
    print(f"\nUsing GF({GF.order}) for bitsize={bitsize}")
    for d in tree_depths:
        print(f"\nTesting depth {d} ({2**d} leaves) with function '{func_name}'")
        num_leaves = 2 ** d
        for trial_tree in range(random_trees):
            leaf_values = [random.randint(0, max_leaf_val) for _ in range(num_leaves)]
            leaf_val_gen = iter(leaf_values).__next__
            root = make_balanced_tree(d, func, leaf_val_gen)
            tree = Tree(root, bitsize, GF)
            times = []
            max_bits_list = []
            tree_outputs = []
            for t in range(trials):
                m = MemoryTracker()
                start = time.time()
                tree.evaluate_with_3_node_limit(m)
                end = time.time()
                memory = m.dump()
                tree_output = extract_tree_output(memory, bitsize)
                tree_outputs.append(tree_output)
                times.append(end - start)
                max_bits_list.append(m.summary()['max_bits'])
            avg_time = sum(times) / len(times)
            avg_max_bits = sum(max_bits_list) / len(max_bits_list)
            result = tree_outputs[0]
            expected = compute_expected_result(leaf_values, func_name)
            is_correct = (result == expected)

            if not is_correct:
                print(f"WARNING: Tree output mismatch at depth={d}, bitsize={bitsize}, trial={trial_tree}, trial_run={t}:\n"
                      f"  Got      : {result}\n"
                      f"  Expected : {expected}\n"
                      f"  Leaves   : {leaf_values}")

            results.append({
                "depth": d,
                "num_leaves": num_leaves,
                "bitsize": bitsize,
                "field": f"GF({GF.order})",
                "func": func_name,
                "avg_time": avg_time,
                "avg_max_bits": avg_max_bits,
                "result": result,
                "expected": expected,
                "correct": is_correct,
                "trial_tree": trial_tree,
                "trial_run": t,
                "leaf_values": leaf_values
            })
            show_progress(trial_tree + 1, random_trees, prefix=f"Depth {d} ({func_name}, {bitsize} bits)")
    return results

if __name__ == "__main__":
    from memory_tracker import MemoryTracker
    from Tree import Tree

    depths = list(range(5, 6))
    bitsizes = [2, 3, 4]
    trials = 1
    random_trees = 3

    all_results = []

    max_bitsize = max(bitsizes)
    GF = compute_field(max_bitsize)

    for bitsize in bitsizes:
        xor_func = lambda x1, x2: x1 ^ x2
        xor_func.__name__ = "xor"
        xor_results = speed_test(depths, xor_func, "xor", bitsize, GF, trials, random_trees)
        all_results += xor_results

        and_func = lambda x1, x2: x1 & x2
        and_func.__name__ = "and"
        and_results = speed_test(depths, and_func, "and", bitsize, GF, trials, random_trees)
        all_results += and_results

    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv("tree_experiment_results.csv", index=False)
    except ImportError:
        pass