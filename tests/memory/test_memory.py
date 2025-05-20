from memory_tracker import MemoryTracker

def test_bit_size():
    tracker = MemoryTracker()
    assert tracker._get_bit_size(0) == 1  # 0 requires 1 bit
    assert tracker._get_bit_size(1) == 1  # 1 requires 1 bit
    assert tracker._get_bit_size(2) == 2  # 2 requires 2 bits
    assert tracker._get_bit_size(3) == 2  # 3 requires 2 bits
    assert tracker._get_bit_size(4) == 3  # 4 requires 3 bits
    assert tracker._get_bit_size(255) == 8  # 255 requires 8 bits
    assert tracker._get_bit_size(256) == 9  # 256 requires 9 bits
    assert tracker._get_bit_size(3.14) == 64  # Assume 64 bits for floats

def test_store_and_load():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    tracker.store("b", 3.14)
    assert tracker.load("a") == 10
    assert tracker.load("b") == 3.14

def test_store_duplicate_variable():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    try:
        tracker.store("a", 20)
        assert False, "Expected ValueError for duplicate variable"
    except ValueError:
        pass

def test_update_variable():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    tracker.update("a", 20)
    assert tracker.load("a") == 20

def test_erase_variable():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    tracker.erase("a")
    try:
        tracker.load("a")
        assert False, "Expected KeyError for erased variable"
    except KeyError:
        pass

def test_summary():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    tracker.store("b", 255)
    tracker.update("a", 256)
    tracker.erase("b")
    summary = tracker.summary()
    assert summary["current_bits"] == 9  # Only "a" with 9 bits remains
    assert summary["max_bits"] >= 17  # At some point, "a" (9 bits) + "b" (8 bits) existed
    assert summary["num_variables"] == 1  # Only "a" remains

def test_dump():
    tracker = MemoryTracker()
    tracker.store("a", 10)
    tracker.store("b", 20)
    dump = tracker.dump()
    assert dump == {"a": 10, "b": 20}