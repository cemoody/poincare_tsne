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
from ptsne import PTSNE

from sklearn.decomposition import PCA
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
    pca = PCA(n_components=600)
    X = pca.fit_transform(X)
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


cuda = True
if os.getenv('GPU', False):
     torch.cuda.set_device(int(os.getenv('GPU', '1')))
draw_ellipse = False
n_points, pij, y, pos, i, j, url, svid = preprocess(limit=3000)

n_topics = 2
n_dim = 2
batchsize = 4096 * 2
print(n_points, n_dim, n_topics, len(pij))

par = PTSNE(n_points, n_dim)
if os.path.exists("model.pt"):
    par.load_state_dict((torch.load('model.pt')))
if cuda:
    par = nn.DataParallel(par)
wrap = Wrapper(par, batchsize=batchsize, epochs=1, cuda=cuda)
for itr in range(500):
    wrap.fit(pij, i, j)

    # Visualize the results
    model = par.module
    model = model.cpu()
    r, _, vec = model.emb.sample(noise=0)
    embed = (r.unsqueeze(-1) * vec).data.numpy()
    f = plt.figure()
    if not draw_ellipse:
        plt.scatter(embed[:, 0], embed[:, 1], alpha=0.05, s=2.0, lw=0.0)
        plt.axis('off')
        plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
        plt.close(f)
    np.savez("model", mu=embed, url=url, svid=svid)
    torch.save(model.state_dict(), 'model.pt')
    if cuda:
        model = model.cuda()
