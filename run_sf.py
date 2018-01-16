from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse

from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path
import glob
import torch
from torch import nn
from wrapper import Wrapper
# from tsne import TSNE
from poincare import transform
from vtsne import VTSNE

from sklearn.externals.joblib import Memory
from sklearn.neighbors import NearestNeighbors

mem = Memory('cache')



@mem.cache
def preprocess(perplexity=30, metric='euclidean', limit=6000):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    fns = glob.glob("/data/svid/*")
    svids = [fn.split('/')[-1].split('_')[1].split('.')[0] for fn in fns]
    svids = set(int(s) for s in svids)
    df = (pd.read_pickle("/data/svid_vgg.pd")
            .drop_duplicates('style_variant_id'))
    df = df[df.style_variant_id.isin(svids)].head(limit)
    url = df.qinfra_url
    svid = df.style_variant_id
    X = df.values[:, 2:].astype('float32')
    X = np.ascontiguousarray(X)
    digits = datasets.load_digits(n_class=6)
    n_points = X.shape[0]
    distances2 = pairwise_distances(X, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij2d = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij2d = squareform(pij2d)
    i, j = np.indices(pij2d.shape)
    i = i.ravel()
    j = j.ravel()
    pij = pij2d.ravel().astype('float32')
    pij = pij.astype('float32')
    i = i.astype('int64')
    j = j.astype('int64')
    # Remove self-indices
    idx = i != j
    idx &= pij > 1e-16
    i, j, pij = i[idx], j[idx], pij[idx]
    return n_points, pij, None, None, i, j, url, svid


@mem.cache
def preprocess_inexact_nmslib(perplexity=30, limit=6000):
    import nmslib
    df = (pd.read_pickle("/data/svid_vgg.pd")
            .drop_duplicates('style_variant_id')
            .head(limit))
    X = df.values[:, 2:].astype('float32')
    X = np.ascontiguousarray(X)
    n_samples = X.shape[0]
    # Cpmpute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    k = min(n_samples - 1, int(3. * perplexity + 1))

    # Find the nearest neighbors for every point
    index = nmslib.init(space='l2', method='hnsw')
    for i, x in enumerate(X):
        nmslib.addDataPoint(index, i, x.tolist())
    nmslib.createIndex(index, {'efConstruction': 400, 'M': 32, 'post': 2})
    nmslib.setQueryTimeParams(index, dict(ef=800))

    # query for the nearest neighbours of the first datapoint
    tups = index.knnQueryBatch(X, k=perplexity * 3 + 1)

    trunc_tups = []
    for neighbor, dist in tups:
        if dist[0] < 1e-6:
            dist = dist[1:]
            neighbor = neighbor[1:]
        else:
            dist = dist[:-1]
            neighbor = neighbor[:-1]
        trunc_tups.append((neighbor, dist))
    neighbors_nn = np.array([t[0] for t in trunc_tups])
    distances_nn = np.array([t[1] for t in trunc_tups])

    l2 = np.sqrt(np.sum((X - X[0])**2.0, axis=1))
    nn = np.argsort(l2)

    # knn return the euclidean distance but we need it squared
    # to be consistent with the 'exact' method. Note that the
    # the method was derived using the euclidean method as in the
    # input space. Not sure of the implication of using a different
    # metric.
    distances_nn **= 2

    # compute the joint probability distribution for the input space
    P = manifold.t_sne._joint_probabilities_nn(distances_nn, neighbors_nn,
                                               perplexity, True)
    Poo = P.tocoo()
    i = Poo.row
    j = Poo.col
    pij = Poo.data

    # Remove self-indices
    idx = i != j
    idx &= pij > 1e-16
    i, j, pij = i[idx], j[idx], pij[idx]
    return n_samples, pij, None, None, i, j


cuda = True
if os.getenv('GPU', False):
     torch.cuda.set_device(int(os.getenv('GPU', '1')))
draw_ellipse = False
n_points, pij, y, pos, i, j, url, svid = preprocess(limit=3000)
# n_points, pij, y, pos, i, j = preprocess_inexact_nmslib()

n_topics = 2
n_dim = 2
batchsize = 4096 * 2
print(n_points, n_dim, n_topics, len(pij))

par = VTSNE(n_points, n_topics, n_dim)
if os.path.exists("model.pt"):
    par.load_state_dict((torch.load('model.pt')))
if cuda:
    par = nn.DataParallel(par)
wrap = Wrapper(par, batchsize=batchsize, epochs=1, cuda=cuda)
for itr in range(500):
    wrap.fit(pij, i, j)

    # Visualize the results
    model = par.module
    embed = model.logits.weight.cpu().data.numpy()
    f = plt.figure()
    if not draw_ellipse:
        plt.scatter(embed[:, 0], embed[:, 1], alpha=0.05, s=2.0, lw=0.0)
        plt.axis('off')
        plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    else:
        # Visualize with ellipses
        var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
        ax = plt.gca()
        for xy, (w, h), c in zip(embed, var, y):
            e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
            e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
            e.set_alpha(0.5)
            ax.add_artist(e)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        plt.axis('off')
        plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    model = model.cpu()
    p = transform(transform(model.logits_mu.weight)).data.numpy()
    np.savez("model", mu=model.logits_mu.weight.data.numpy(),
             lv=model.logits_lv.weight.data.numpy(), p=p,
             y=y, img=pos, url=url, svid=svid)
    torch.save(model.state_dict(), 'model.pt')
    if cuda:
        model = model.cuda()

