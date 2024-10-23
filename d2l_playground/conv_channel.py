import torch
from utils import d2l

def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(X_i, K_i) for X_i, K_i in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim = 0)

X = torch.tensor([[[0., 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1., 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0., 1], [2, 3]], [[1., 2], [3, 4]]])
K = torch.stack((K, K + 1, K + 2), 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, w, h = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, w * h))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, w, h))

K = torch.tensor([[[1.]], [[1]]])
K = torch.stack((K, K + 1, K + 2), 0)
Y1 = corr2d_multi_in_out(X, K)
Y2 = corr2d_multi_in_out_1x1(X, K)
print(torch.abs(Y1 - Y2).sum() < 1e-6)


