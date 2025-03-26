import torch
import torch.utils.cpp_extension
torch.ops.load_library(torch.utils.cpp_extension.load('upfirdn2d_plugin', 'src/stylegan2/torch_utils/ops/upfirdn2d.cpp'))
print("upfirdn2d_plugin compiled successfully!")
