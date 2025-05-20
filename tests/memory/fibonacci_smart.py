from memory_tracker import MemoryTracker

def fibonacci(i, m):
    """Fibonacci function with memory tracking. Using at max 3 registers."""
    # Base cases
    var1 = 0
    var2 = 1
    m.store('var1', var1)
    m.store('var2', var2)
    
    for _ in range(2, i + 1):
        next_var = var1 + var2
        m.update('var1', var1)
        m.update('var2', var2)
        var1, var2 = var2, next_var
        m.update('var2', var2)
    
    return var2
    
if __name__ == "__main__":
    m = MemoryTracker()
    n = 8  # Calculate Fibonacci(8)
    result = fibonacci(n, m)

    print(f"Fibonacci({n}) = {result}")
    print("Memory summary:", m.summary())
    print("Final memory state:", m.dump())