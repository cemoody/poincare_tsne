import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np

from poincare import transform
from poincare import distance
from poincare import distance_batch


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    z = eps.mul(std).add_(mu)
    kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld).mul_(-0.5)
    return z, kld


class VTSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim):
        super(VTSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)
        self.n_points = n_points
        self.n_dim = n_dim
        self.logits_lv.weight.data.add_(5)
    
    @property
    def logits(self):
        return self.logits_mu

    def sample_logits(self, i=None):
        if i is None:
            return reparametrize(self.logits_mu.weight,
                                 self.logits_lv.weight)
        else:
            return reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward_student_poincare(self, pij, i, j):
        # Get  for all points
        x, loss_kldrp = self.sample_logits()
        p = transform(transform(x))
        # Compute partition function
        dkl = distance_batch(p, p)
        n_diagonal = dkl.size()[0]
        # The sums over all elements k and l, but in reality we want to
        # skip summing over the diagonal, so we correct by subtracting
        # the diagonal off
        part = ((1 + dkl**2.0).pow(-1.0)).sum() - n_diagonal
        # Compute the numerator
        xi, _ = self.sample_logits(i)
        xj, _ = self.sample_logits(j)
        pi = transform(transform(xi))
        pj = transform(transform(xj))
        dij = distance(pi, pj)
        num = (1. + dij**2.0).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        # Compute KLD(pij || qij)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute sum of likelihood and regularization terms
        loss = loss_kld.sum() + loss_kldrp.sum() * 1e-7
        return loss

    def forward_poincare(self, pij, i, j):
        # Get  for all points
        x, loss_kldrp = self.sample_logits()
        p = transform(transform(x))
        # 
        return loss

    def __call__(self, *args):
        return self.forward(*args)
