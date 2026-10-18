# Creating the arrays.
# Checking for the dimesions of the arrays

import numpy as np

array = np.array(
    [[["A", "B", "C"], ["D", "E", "F"]], [["G", "H", "I"], ["J", "K", "L"]]]
)
# array *= 2

print(array.ndim)
print(array.shape)
print(array[1, 1, 2])

name_all = array[0, 0, 0] + array[1, 1, 2] + array[1, 1, 2]
print(name_all)

beak = array[0, 0, 1] + array[0, 1, 1] + array[0, 0, 0] + array[1, 1, 1]
print(beak)
