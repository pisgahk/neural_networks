import torch

# CREATING TENSORS.

# 1. Direct creation from data
data = [ [1,2,3], [4,5,6]]
my_tensor = torch.tensor(data)
print(my_tensor)

# 2. Creating from a desired shape.
shape = (2,3)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)

print(f"random tensor: \n {random}")

# 3. Creating by mimicking another tensor.


