from memory_tracker import MemoryTracker

def fibonacci(i, m):
    # Base cases
    if i == 0:
        if f'f_{i}' not in m.variable_storage:
            m.store(f'f_{i}', 0)
        else:
            m.update(f'f_{i}', 0)
        return 0
    if i == 1:
        if f'f_{i}' not in m.variable_storage:
            m.store(f'f_{i}', 1)
        else:
            m.update(f'f_{i}', 1)
        return 1

    # If already computed, reuse it
    if f'f_{i}' in m.variable_storage:
        return m.load(f'f_{i}')
    
    v1 = fibonacci(i - 1, m)
    if f'f_{i-1}' not in m.variable_storage:
        m.store(f'f_{i-1}', v1)
    else:
        m.update(f'f_{i-1}', v1)
    v2 = fibonacci(i - 2, m)
    if f'f_{i-2}' not in m.variable_storage:
        m.store(f'f_{i-2}', v2)
    else:
        m.update(f'f_{i-2}', v2)
    
    result = v1 + v2
    if f'f_{i}' not in m.variable_storage:
        m.store(f'f_{i}', result)
    else:
        m.update(f'f_{i}', result)
    
    return result

if __name__ == "__main__":
    m = MemoryTracker()
    n = 8  # Calculate Fibonacci(8)
    result = fibonacci(n, m)

    print(f"Fibonacci({n}) = {result}")
    print("Memory summary:", m.summary())
    print("Final memory state:", m.dump())