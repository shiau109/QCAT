import numpy as np

# Example 2D array
array = np.array([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])

# Find the indices of the maximum values along each column
max_indices = np.argmax(array, axis=0)
max_val = np.max(array, axis=1)

# Print the indices
print(max_indices, max_val)
