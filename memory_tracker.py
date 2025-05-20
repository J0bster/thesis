import numpy as np

class MemoryTracker:
    def __init__(self):
        self.variable_storage = {}
        self.sizes_in_bits = {}
        self.current_num_bits = 0
        self.max_num_bits = 0
        self.max_num_variables = 0

    def _get_bit_size(self, value):
        if isinstance(value, int):
            return int(np.ceil(np.log2(value + 1))) if value > 0 else 1
        elif isinstance(value, float):
            return 64
        else:
            return 32

    def store(self, varname, value):
        if varname in self.variable_storage:
            raise ValueError(f"Variable '{varname}' already exists. Use update() instead.")
        
        size = self._get_bit_size(value)
        self.variable_storage[varname] = value
        self.current_num_bits += size
        self.max_num_bits = max(self.max_num_bits, self.current_num_bits)
        self.max_num_variables = max(self.max_num_variables, len(self.variable_storage))

    def update(self, varname, new_value):
        if varname not in self.variable_storage:
            raise KeyError(f"Variable '{varname}' does not exist. Use store() instead.")
        
        old_size = self._get_bit_size(self.variable_storage[varname])
        new_size = self._get_bit_size(new_value)
        self.variable_storage[varname] = new_value
        self.current_num_bits += (new_size - old_size)
        self.max_num_bits = max(self.max_num_bits, self.current_num_bits)

    def load(self, varname):
        if varname not in self.variable_storage:
            raise KeyError(f"Variable '{varname}' does not exist.")
        return self.variable_storage[varname]

    def erase(self, varname):
        if varname in self.variable_storage:
            size = self._get_bit_size(self.variable_storage[varname])
            del self.variable_storage[varname]
            self.current_num_bits -= size

    def summary(self):
        return {
            "current_bits": self.current_num_bits,
            "max_bits": self.max_num_bits,
            "num_variables": len(self.variable_storage),
            "max_num_variables": self.max_num_variables,
        }

    def dump(self):
        return dict(self.variable_storage)