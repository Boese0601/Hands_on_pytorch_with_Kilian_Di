
# input X
# Ht = fai( Xt Wxh + Ht-1 Whh+ bh)
# Output O
# Ot = Ht Whq + bq

import torch

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
Ht = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# same
Ht_ = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0))
