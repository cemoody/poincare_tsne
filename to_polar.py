import torch
import poincare
import numpy as np
from torch.autograd import Variable
import pandas as pd

import torch.nn.functional as F
import torch


def l2(x):
    return dot(x, x)


def dot(a, b):
    return (a * b).sum(dim=-1)


def transform(theta, eps=1e-5):
    l2t = torch.sqrt(l2(theta)).unsqueeze(1) + eps
    div = 2 + F.elu(l2t - 2)
    ret = theta / (div + eps)
    return ret


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

mu = np.load("./model.npz")['mu']
lv = np.load("./model.npz")['lv']
url = np.load('model.npz')['url']
svid= np.load('model.npz')['svid']
loc = ['/data/svid/svid_{:d}.jpg'.format(s) for s in svid]
mu = Variable(torch.from_numpy(mu))
lv = Variable(torch.from_numpy(lv))
pmu = transform(mu)
pmu = pmu.data.numpy()

rho, phi = cart2pol(pmu[:, 0], pmu[:, 1])

full = pd.DataFrame(dict(rho=rho, phi=phi, x=pmu[:, 0], y=pmu[:, 1],
                         url=url, loc=loc))
full['eta'] = np.log(full.rho.max() - full.rho + 1e-12)
full['radius_zscore'] = (full.rho.mean() - full.rho)/full.rho.std()
full['radius_zscore_clipped'] = np.clip(full.radius_zscore, -4, 4)
lim = full[full.eta > -15]
lim[['eta', 'phi', 'loc']].to_csv('polar.csv', index=False, header=False)

