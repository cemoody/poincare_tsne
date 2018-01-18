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


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def find_parent(row, thresh=5):
    distances = dist[row.name].copy()
    distances[rdiff[row.name] > -1e-9] = 1e8
    # distances[distances < thresh] = 1e8
    index = np.argsort(distances)[1]
    parent = full.iloc[full.index == index].iloc[0]
    dpr = distances[index]
    return parent, dpr



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

mu = np.load("./model.npz")['mu']
lv = np.load("./model.npz")['lv']
url = np.load('model.npz')['url']
svid= np.load('model.npz')['svid']
loc = ['/data/svid/svid_{:d}.jpg'.format(s) for s in svid]
mu = Variable(torch.from_numpy(mu))
lv = Variable(torch.from_numpy(lv))
pmu = transform(mu)
dist = distance_batch(pmu, pmu).data.numpy()
rdiff = radius_diff_batch(pmu, pmu).data.numpy()
pmu = pmu.data.numpy()


rho, phi = cart2pol(pmu[:, 0], pmu[:, 1])

full = pd.DataFrame(dict(rho=rho, phi=phi, x=pmu[:, 0], y=pmu[:, 1],
                         url=url, loc=loc))
full['eta'] = np.log(full.rho.max() - full.rho + 1e-12)
full['radius_zscore'] = (full.rho.mean() - full.rho)/full.rho.std()
full['radius_zscore_clipped'] = np.clip(full.radius_zscore, -4, 4)

percentile = np.argsort(full.rho, axis=0) / (len(full) * 1.0)
full['rho_percentile'] = percentile
full['rho_sqpercentile'] = np.sqrt(percentile)
full['rho_p4percentile'] = percentile**(0.20)
xsp, ysp = pol2cart(full.rho_sqpercentile, full.phi)
xp, yp = pol2cart(full.rho_percentile, full.phi)
xp4, yp4 = pol2cart(full.rho_p4percentile, full.phi)
full['sqpercentile_x'] = xsp
full['sqpercentile_y'] = ysp
full['percentile_x'] = xp
full['percentile_y'] = yp
full['percentile_xp4'] = xp4
full['percentile_yp4'] = yp4

# lim = full[full.eta > -15]
full[['eta', 'phi', 'loc']].to_csv('polar.csv', index=False, header=False)
full[['rho_percentile', 'phi', 'loc']].to_csv('percentile.csv', index=False, header=False)
full[['rho_sqpercentile', 'phi', 'loc']].to_csv('sqpercentile.csv', index=False, header=False)
full[['x', 'y', 'loc']].to_csv('xy.csv', index=False, header=False)
full[['sqpercentile_x', 'sqpercentile_y', 'loc']].to_csv('sqpercentile_xy.csv', index=False, header=False)
full[['percentile_x', 'percentile_y', 'loc']].to_csv('percentile_xy.csv', index=False, header=False)
full[['percentile_xp4', 'percentile_yp4', 'loc']].to_csv('percentile_xyp4.csv', index=False, header=False)

xs = []
ys = []
for _, row in full.iterrows():
    parent, dpr = find_parent(row)
    if dpr < np.inf:
        xs.append(parent.percentile_xp4)
        ys.append(parent.percentile_yp4)
    else:
        xs.append(None)
        ys.append(None)
full['percentile_p4_parent_x'] = xs
full['percentile_p4_parent_y'] = ys
full.to_csv('full.csv', index=False)
