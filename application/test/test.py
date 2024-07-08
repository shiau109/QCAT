import numpy as np

# Original array
array = np.array([1, 2, 3, 4, 5])

# Number of zeros to add
num_zeros = 5

# Extend the array with zeros
extended_array = np.pad(array, (1, num_zeros), 'constant', constant_values=0)

print("Original array:", array)
print("Extended array:", extended_array)