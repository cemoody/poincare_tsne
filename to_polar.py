import torch
import poincare
import numpy as np
from torch.autograd import Variable
import pandas as pd

import torch.nn.functional as F
import torch

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
    l2t = torch.sqrt(l2(theta)).unsqueeze(1) + eps
    div = 2 + F.elu(l2t - 2)
    ret = theta / (div + eps)
    return ret

def distance_batch(umat, vmat, eps=1e-5):
    u = umat.unsqueeze(1)
    v = vmat.unsqueeze(0)
    uvdiff = (u - v + eps)
    uvdot = (u * v).sum(dim=-1)
    diff = norm(uvdiff)
    l2u, l2v = l2(u), l2(v)
    alpha = torch.sqrt(1 - l2u + eps)
    beta = torch.sqrt(1 - l2v + eps)
    root = l2u * l2v - 2 * uvdot + 1 + eps
    num = diff + torch.sqrt(root)
    div = alpha * beta
    ret = 2 * torch.log(num / div)
    return ret

def radius_diff_batch(umat, vmat):
    u = umat.unsqueeze(1)
    v = vmat.unsqueeze(0)
    nu = norm(u)
    nv = norm(v)
    return nu - nv


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

mu = np.load("./model.npz")['mu']
lv = np.load("./model.npz")['lv']
y = np.load('input.npz')['y']
img = np.load('input.npz')['pos']

mu = Variable(torch.from_numpy(mu))
lv = Variable(torch.from_numpy(lv))

rho, phi = cart2pol(pmu[:, 0], pmu[:, 1])
eta = np.log(1 - rho)
idx = (phi < -1.0) & (phi > -0.4)

full = pd.DataFrame(dict(rho=rho, phi=phi, x=pmu[:, 0], y=pmu[:, 1], label=y))
full['eta'] = np.log(1 - full.rho)
full['radius_zscore'] = (full.rho.mean() - full.rho)/full.rho.std()
full['radius_zscore_clipped'] = np.clip(df.radius_zscore, -4, 4)
