import torch
from torch.autograd import Function
import torch.nn.functional as F


def norm(x):
    l2x = l2(x)
    sqrt = torch.sqrt(l2x)
    assert (sqrt.data != sqrt.data).sum() == 0
    return sqrt


def l2(x):
    return dot(x, x)


def dot(a, b):
    return (a * b).sum(dim=-1)


def transform(theta, eps=1e-5):
    # as l2t-> 0 norm->theta, div = 1
    # as l2t-> 1 norm->theta / ||l2t||, div = ||theta||
    # as l2t-> +inf norm->theta / ||l2t||, div = ||theta||
    # when l2t = 0: div = (1 + ( 0) (+1)) = 1
    # when l2t = 1: div = (1 + ( 0) (+1)) = 1
    # when l2t = 2: div = (1 + ( 1) (+1)) = 2
    # when l2t = 3: div = (1 + ( 2) (+1)) = 3
    # div = (1 + relu(l2t - 1))
    # or with ELU:
    # relu(x) = elu(x - 1) + 1
    # div = 1 + (1 + elu(l2t - 2))
    # div = 2 + elu(l2t - 2)
    # when l2t = 0: div = (2 + (-1.0)) = 1
    # when l2t = 1: div = (2 + (-0.3)) = 1.7
    # when l2t = 2: div = (2 + ( 0.0)) = 2
    # when l2t = 3: div = (2 + (+1.0)) = 3
    # theta = torch.clamp(theta, -cmax, cmax)
    l2t = torch.sqrt(l2(theta)).unsqueeze(1) + eps
    div = 2 + F.elu(l2t - 2)
    ret = theta / (div + eps)
    # l2t = torch.sqrt(l2(ret)).unsqueeze(1)
    # ret2 = ret / (F.relu(l2t - 1) + 1 + eps)
    return ret


def distance(u, v, eps=1e-5):
    diff = norm(u - v)
    alpha = torch.sqrt(1. - l2(u))
    beta = torch.sqrt(1. - l2(v))
    root = l2(u) * l2(v) - 2 * (u * v).sum(dim=1) + 1
    num = diff + torch.sqrt(root)
    div = alpha * beta
    ret = 2 * torch.log(num / div)
    return ret


def save_grad(name):
    def hook(grad):
        n = (grad != grad).sum()
        print(name, n, grad.size())
    return hook


def distance_batch(umat, vmat, eps=1e-5):
    u = umat.unsqueeze(1)
    v = vmat.unsqueeze(0)
    # u.register_hook(save_grad('u'))
    uvdiff = (u - v + eps)
    # uvdiff.register_hook(save_grad('uvdiff'))
    uvdot = (u * v).sum(dim=-1)
    # uvdot.register_hook(save_grad('uvdot'))
    diff = norm(uvdiff)
    # diff.register_hook(save_grad('diff'))
    l2u, l2v = l2(u), l2(v)
    # l2u.register_hook(save_grad('l2u'))
    assert torch.sum(l2u > 1).data[0] == 0
    assert torch.sum(l2v > 1).data[0] == 0
    alpha = torch.sqrt(1 - l2u + eps)
    beta = torch.sqrt(1 - l2v + eps)
    # alpha.register_hook(save_grad('alpha'))
    root = l2u * l2v - 2 * uvdot + 1 + eps
    num = diff + torch.sqrt(root)
    # num.register_hook(save_grad('num'))
    div = alpha * beta
    ret = 2 * torch.log(num / div)
    # ret.register_hook(save_grad('ret'))
    return ret
