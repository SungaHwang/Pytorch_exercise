import inspect
import os
import torch
from torch import nn

details = inspect.getsource(nn.Module)
print(details)

fname = 'nn.Module_details.py'

with open(fname, 'w') as f:
    f.write('data = [{}]'.format(details))