import torch
import numpy as np
## create tensor
# 1.from nunmpy
# torch.from_numpy(np.array([1,2,3]))
# 2.using tensor()
# torch.tensor([[1,2,3],[3,4,5]])
# 3.uniform distribution
# a = torch.rand(3, 3) # shape (3,3) torch.randperm(2) permutaion
# torch.rand_like(a) # create a tensor whose shape is like a
# torch.randint(1, 10, (3, 3)) # create with int within range 1-10
# 4.normal db
# torch.randn(3,3)
# 5. for same number
# torch.full([3,3], 7)
# 6.range number like range()
# torch.arange(0, 10, 2) # where 2 is step
# torch.linspace(0,10,steps=5) # from 0 to 10, split 5 parts equally
# 7. high dimension
# torch.ones(3,3) zeros(3,3) eye(3,3)
# index ... & index_select
# tensor.index_select(2, torch.arange(8)) tensor[..., 0:28:2]

if __name__ == '__main__':
    print(torch.randperm(10))